use image::{DynamicImage, ImageBuffer, Luma, Rgb};
use imageproc::filter::gaussian_blur_f32;

pub fn sharpen_image(image: DynamicImage, strength: u8) -> DynamicImage {
    let rgb = image.to_rgb8();

    // Tuned RGB unsharp mask parameters per strength level.
    let (sigma, amount) = match strength {
        1 => (0.5, 0.30),
        2 => (0.6, 0.45),
        3 => (0.8, 0.60),
        4 => (1.0, 0.80),
        _ => (1.2, 1.00), // strength 5+
    };

    let blurred = gaussian_blur_f32(&rgb, sigma);
    let mut out = ImageBuffer::new(rgb.width(), rgb.height());

    for (x, y, pixel) in out.enumerate_pixels_mut() {
        let orig = rgb.get_pixel(x, y);
        let blur = blurred.get_pixel(x, y);

        let mut sharpened = [0u8; 3];
        for c in 0..3 {
            let o = orig[c] as f32;
            let b = blur[c] as f32;
            let val = o + amount * (o - b);
            sharpened[c] = val.clamp(0.0, 255.0).round() as u8;
        }

        *pixel = Rgb(sharpened);
    }

    DynamicImage::ImageRgb8(out)
}

pub fn sharpen_image_luma(image: DynamicImage, strength: u8) -> DynamicImage {
    let rgb = image.to_rgb8();
    let (w, h) = rgb.dimensions();

    // Luma-only unsharp mask tuned for the experimental denoiser output.
    let (sigma, amount) = match strength {
        1 => (0.45, 0.60),
        2 => (0.55, 0.80),
        3 => (0.70, 1.00),
        4 => (0.85, 1.15),
        _ => (0.95, 1.30),
    };

    // Build YCbCr planes (f32).
    let mut y_plane: ImageBuffer<Luma<f32>, Vec<f32>> = ImageBuffer::new(w, h);
    let mut cb_plane: ImageBuffer<Luma<f32>, Vec<f32>> = ImageBuffer::new(w, h);
    let mut cr_plane: ImageBuffer<Luma<f32>, Vec<f32>> = ImageBuffer::new(w, h);

    for (x, y, p) in rgb.enumerate_pixels() {
        let r = p[0] as f32;
        let g = p[1] as f32;
        let b = p[2] as f32;

        let yv = 0.299 * r + 0.587 * g + 0.114 * b;
        let cb = (b - yv) * 0.564 + 128.0;
        let cr = (r - yv) * 0.713 + 128.0;

        y_plane.put_pixel(x, y, Luma([yv]));
        cb_plane.put_pixel(x, y, Luma([cb]));
        cr_plane.put_pixel(x, y, Luma([cr]));
    }

    let y_blur = gaussian_blur_f32(&y_plane, sigma);
    let mut out = ImageBuffer::new(w, h);

    for (x, y, pixel) in out.enumerate_pixels_mut() {
        let y_orig = y_plane.get_pixel(x, y)[0];
        let y_b = y_blur.get_pixel(x, y)[0];
        let y_sharp = y_orig + amount * (y_orig - y_b);

        let cb = cb_plane.get_pixel(x, y)[0];
        let cr = cr_plane.get_pixel(x, y)[0];

        // Reconstruct RGB from YCbCr.
        let r = y_sharp + 1.403 * (cr - 128.0);
        let b = y_sharp + 1.773 * (cb - 128.0);
        let g = y_sharp - 0.714 * (cr - 128.0) - 0.344 * (cb - 128.0);

        *pixel = Rgb([
            r.clamp(0.0, 255.0).round() as u8,
            g.clamp(0.0, 255.0).round() as u8,
            b.clamp(0.0, 255.0).round() as u8,
        ]);
    }

    DynamicImage::ImageRgb8(out)
}
