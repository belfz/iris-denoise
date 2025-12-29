use image::{DynamicImage, ImageBuffer, Rgb};
use imageproc::filter::gaussian_blur_f32;

pub fn sharpen_image(image: DynamicImage, strength: u8) -> DynamicImage {
    let rgb = image.to_rgb8();

    // Tuned unsharp mask parameters per strength level.
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

