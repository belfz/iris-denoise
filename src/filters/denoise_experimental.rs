use image::{DynamicImage, ImageBuffer, Luma, Rgb};
use imageproc::filter::gaussian_blur_f32;

/// Experimental astrophotography-oriented denoiser (multi-scale, chroma-heavy).
///
/// Goals:
/// - Preserve nebulosity and faint dust by keeping medium-scale structure.
/// - Keep stars: positive high-frequency detail is retained more than negative.
/// - Hit color noise hard while leaving luminance detail comparatively intact.
pub fn denoise_image_experimental(image: DynamicImage, strength: u8) -> DynamicImage {
    let rgb = image.to_rgb8();
    let (w, h) = rgb.dimensions();

    // Tunables per strength (1-5).
    // sigma_fine: small blur for fine detail separation.
    // sigma_base: larger blur for coarse base.
    // sigma_chroma: strong chroma smoothing (applied twice with blend).
    // fine_thresh/coarse_thresh: soft-thresholds for detail suppression.
    // gains: asymmetric to favor positive (star) detail over negative (noise).
    let (
        sigma_fine,
        sigma_base,
        sigma_chroma,
        fine_thresh,
        coarse_thresh,
        fine_gain_pos,
        fine_gain_neg,
        coarse_gain,
        base_blend,
    ) = match strength {
        1 => (0.9, 1.8, 2.4, 2.2, 1.1, 0.85, 0.25, 0.80, 0.78),
        2 => (1.05, 2.1, 2.7, 2.7, 1.3, 0.90, 0.26, 0.82, 0.78),
        3 => (1.20, 2.4, 3.1, 3.2, 1.5, 0.95, 0.28, 0.85, 0.79),
        4 => (1.35, 2.7, 3.4, 3.8, 1.8, 1.00, 0.30, 0.88, 0.80),
        _ => (1.50, 3.0, 3.8, 4.4, 2.0, 1.05, 0.32, 0.90, 0.80),
    };

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

    // Multi-scale luminance: fine and base.
    let y_blur_fine = gaussian_blur_f32(&y_plane, sigma_fine);
    let y_blur_base = gaussian_blur_f32(&y_blur_fine, sigma_base);

    // Strong chroma smoothing with two-stage blur to better remove color speckle.
    let sigma_chroma2 = sigma_chroma * 1.2;
    let cb_blur1 = gaussian_blur_f32(&cb_plane, sigma_chroma);
    let cb_blur2 = gaussian_blur_f32(&cb_blur1, sigma_chroma2);
    let cb_smooth = blend_planes(&cb_blur1, &cb_blur2, 0.5);

    let cr_blur1 = gaussian_blur_f32(&cr_plane, sigma_chroma);
    let cr_blur2 = gaussian_blur_f32(&cr_blur1, sigma_chroma2);
    let cr_smooth = blend_planes(&cr_blur1, &cr_blur2, 0.5);

    let mut out = ImageBuffer::new(w, h);
    for (x, y, pixel) in out.enumerate_pixels_mut() {
        let y_orig = y_plane.get_pixel(x, y)[0];
        let fine = y_orig - y_blur_fine.get_pixel(x, y)[0];
        let coarse = y_blur_fine.get_pixel(x, y)[0] - y_blur_base.get_pixel(x, y)[0];

        // Soft-threshold details.
        let fine_d = soft_shrink(fine, fine_thresh);
        let coarse_d = soft_shrink(coarse, coarse_thresh);

        // Asymmetric gains: preserve stars (positive), suppress noise (negative).
        let fine_term = fine_d.max(0.0) * fine_gain_pos + fine_d.min(0.0) * fine_gain_neg;
        let coarse_term = coarse_d * coarse_gain;

        // Base luminance blended with original to stabilize brightness and keep nebulosity glow.
        let base = base_blend * y_blur_base.get_pixel(x, y)[0] + (1.0 - base_blend) * y_orig;
        let y_final = base + coarse_term + fine_term;

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

fn soft_shrink(v: f32, thresh: f32) -> f32 {
    let mag = v.abs();
    if mag <= thresh {
        0.0
    } else {
        v.signum() * (mag - thresh)
    }
}

fn blend_planes(
    a: &ImageBuffer<Luma<f32>, Vec<f32>>,
    b: &ImageBuffer<Luma<f32>, Vec<f32>>,
    alpha: f32,
) -> ImageBuffer<Luma<f32>, Vec<f32>> {
    let mut out = ImageBuffer::new(a.width(), a.height());
    for (x, y, pix) in out.enumerate_pixels_mut() {
        let va = a.get_pixel(x, y)[0];
        let vb = b.get_pixel(x, y)[0];
        *pix = Luma([alpha * va + (1.0 - alpha) * vb]);
    }
    out
}
