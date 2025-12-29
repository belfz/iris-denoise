use image::{DynamicImage, ImageBuffer, Luma, Rgb};
use imageproc::filter::gaussian_blur_f32;

/// Experimental astrophotography-oriented denoiser.
///
/// Goals:
/// - Preserve nebulosity / faint structures by keeping luminance detail.
/// - Protect stars by re-injecting positive high-frequency luminance.
/// - Target color (chrominance) noise more aggressively than luminance noise.
pub fn denoise_image_experimental(image: DynamicImage, strength: u8) -> DynamicImage {
    let rgb = image.to_rgb8();
    let (w, h) = rgb.dimensions();

    // Tunables derived from strength (1-5).
    let (sigma_luma, sigma_chroma) = match strength {
        1 => (0.6, 1.0),
        2 => (0.8, 1.2),
        3 => (1.0, 1.4),
        4 => (1.15, 1.55),
        _ => (1.3, 1.7),
    };
    let sigma_star = 1.2;
    let star_boost = 0.45;

    // Build Y, Cb, Cr planes as f32.
    let mut y_plane: ImageBuffer<Luma<f32>, Vec<f32>> = ImageBuffer::new(w, h);
    let mut cb_plane: ImageBuffer<Luma<f32>, Vec<f32>> = ImageBuffer::new(w, h);
    let mut cr_plane: ImageBuffer<Luma<f32>, Vec<f32>> = ImageBuffer::new(w, h);

    for (x, y, pixel) in rgb.enumerate_pixels() {
        let r = pixel[0] as f32;
        let g = pixel[1] as f32;
        let b = pixel[2] as f32;

        // ITU-R BT.601 luma and chroma (offset chroma by +128).
        let yv = 0.299 * r + 0.587 * g + 0.114 * b;
        let cb = (b - yv) * 0.564 + 128.0;
        let cr = (r - yv) * 0.713 + 128.0;

        y_plane.put_pixel(x, y, Luma([yv]));
        cb_plane.put_pixel(x, y, Luma([cb]));
        cr_plane.put_pixel(x, y, Luma([cr]));
    }

    // Soften luminance slightly.
    let y_soft = gaussian_blur_f32(&y_plane, sigma_luma);
    // Extract high-frequency luminance to protect stars/edges.
    let y_star = gaussian_blur_f32(&y_plane, sigma_star);

    // Smooth chroma more aggressively to knock down color noise.
    let cb_smooth = gaussian_blur_f32(&cb_plane, sigma_chroma);
    let cr_smooth = gaussian_blur_f32(&cr_plane, sigma_chroma);

    let mut out = ImageBuffer::new(w, h);
    for (x, y, pixel) in out.enumerate_pixels_mut() {
        let y_orig = y_plane.get_pixel(x, y)[0];
        let y_base = y_soft.get_pixel(x, y)[0];
        let y_hp = (y_orig - y_star.get_pixel(x, y)[0]).clamp(-12.0, 18.0);

        // Re-inject positive high-frequency detail to preserve stars,
        // allow a small amount of negative to avoid halos.
        let y_final = y_base + star_boost * y_hp.max(0.0) + 0.2 * y_hp.min(0.0);

        let cb = cb_smooth.get_pixel(x, y)[0];
        let cr = cr_smooth.get_pixel(x, y)[0];

        // Reconstruct RGB from YCbCr.
        let r = y_final + 1.403 * (cr - 128.0);
        let b = y_final + 1.773 * (cb - 128.0);
        let g = y_final - 0.714 * (cr - 128.0) - 0.344 * (cb - 128.0);

        *pixel = Rgb([
            r.clamp(0.0, 255.0).round() as u8,
            g.clamp(0.0, 255.0).round() as u8,
            b.clamp(0.0, 255.0).round() as u8,
        ]);
    }

    DynamicImage::ImageRgb8(out)
}

