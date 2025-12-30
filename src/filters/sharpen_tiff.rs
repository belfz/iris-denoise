use image::{DynamicImage, Rgb};
use imageproc::filter::gaussian_blur_f32;

/// Unsharp mask for TIFF (RGB16-capable) while preserving bit depth.
pub fn sharpen_image_tiff(image: DynamicImage, strength: u8) -> DynamicImage {
    let rgb = image.to_rgb16();

    let (sigma, amount) = match strength {
        1 => (0.35, 0.30),
        2 => (0.45, 0.45),
        3 => (0.60, 0.60),
        4 => (0.75, 0.80),
        _ => (0.90, 1.00),
    };

    let blurred = gaussian_blur_f32(&rgb, sigma);
    let mut out = rgb.clone();

    for (x, y, pixel) in out.enumerate_pixels_mut() {
        let orig = rgb.get_pixel(x, y);
        let blur = blurred.get_pixel(x, y);

        let mut sharpened = [0u16; 3];
        for c in 0..3 {
            let o = orig[c] as f32;
            let b = blur[c] as f32;
            let val = o + amount * (o - b);
            sharpened[c] = val.clamp(0.0, u16::MAX as f32).round() as u16;
        }

        *pixel = Rgb(sharpened);
    }

    DynamicImage::ImageRgb16(out)
}
