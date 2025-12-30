use image::{DynamicImage, ImageBuffer, Rgb};
use imageproc::filter::gaussian_blur_f32;
use rayon::prelude::*;

/// Denoise TIFF (RGB16-capable) while preserving bit depth.
pub fn denoise_image_tiff(image: DynamicImage, strength: u8) -> DynamicImage {
    let rgb = image.to_rgb16();

    let (median_kernel, sigma) = match strength {
        1 => (1, 0.20),
        2 => (2, 0.35),
        3 => (3, 0.50),
        4 => (4, 0.65),
        _ => (5, 0.80),
    };

    let median_pass = median_filter_parallel_u16(&rgb, median_kernel);
    let blurred = gaussian_blur_f32(&median_pass, sigma);

    DynamicImage::ImageRgb16(blurred)
}

fn median_filter_parallel_u16(
    img: &ImageBuffer<Rgb<u16>, Vec<u16>>,
    kernel: u32,
) -> ImageBuffer<Rgb<u16>, Vec<u16>> {
    let w = img.width() as usize;
    let h = img.height() as usize;
    let r = (kernel / 2) as isize;

    let rows: Vec<Vec<[u16; 3]>> = (0..h)
        .into_par_iter()
        .map(|y| {
            let mut row = Vec::with_capacity(w);
            for x in 0..w {
                let mut med = [0u16; 3];
                for c in 0..3 {
                    let mut values = Vec::with_capacity((kernel * kernel) as usize);
                    for ky in -r..=r {
                        let yy = (y as isize + ky).clamp(0, h as isize - 1) as usize;
                        for kx in -r..=r {
                            let xx = (x as isize + kx).clamp(0, w as isize - 1) as usize;
                            values.push(img[(xx as u32, yy as u32)][c]);
                        }
                    }
                    values.sort_unstable();
                    med[c] = values[values.len() / 2];
                }
                row.push(med);
            }
            row
        })
        .collect();

    let mut out = vec![0u16; w * h * 3];
    for (y, row) in rows.into_iter().enumerate() {
        for (x, pix) in row.into_iter().enumerate() {
            let idx = (y * w + x) * 3;
            out[idx..idx + 3].copy_from_slice(&pix);
        }
    }

    ImageBuffer::from_raw(w as u32, h as u32, out).expect("buffer size matches image dimensions")
}
