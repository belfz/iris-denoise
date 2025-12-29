use std::path::{Path, PathBuf};
use std::{env, process};

use image::{DynamicImage, ImageBuffer, ImageError, ImageFormat, ImageReader, Rgb};
use imageproc::filter::gaussian_blur_f32;
use rayon::prelude::*;

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let (input_path, output_path, strength) = parse_args()?;

    let image = load_image(&input_path)?;
    let denoised = denoise_image(image, strength);
    denoised.save(&output_path)?;

    println!("Denoised image written to {}", output_path.display());
    Ok(())
}

fn parse_args() -> Result<(PathBuf, PathBuf, u8), String> {
    let mut args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        return Err(usage());
    }

    let input_path = PathBuf::from(args.remove(0));
    let mut output_path: Option<PathBuf> = None;
    let mut strength: Option<u8> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--strength" | "-s" => {
                let val = args
                    .get(i + 1)
                    .ok_or_else(|| "missing value after --strength".to_string())?;
                strength = Some(parse_strength(val)?);
                i += 2;
            }
            "--output" | "-o" => {
                let val = args
                    .get(i + 1)
                    .ok_or_else(|| "missing value after --output".to_string())?;
                output_path = Some(PathBuf::from(val));
                i += 2;
            }
            other => {
                // Positional fallbacks: first extra = output, second extra = strength number.
                if output_path.is_none() && !other.chars().all(|c| c.is_ascii_digit()) {
                    output_path = Some(PathBuf::from(other));
                } else if strength.is_none() {
                    strength = Some(parse_strength(other)?);
                }
                i += 1;
            }
        }
    }

    let strength = strength.unwrap_or(3);
    let output_path = output_path.unwrap_or_else(|| default_output_path(&input_path, strength));

    Ok((input_path, output_path, strength))
}

fn usage() -> String {
    "Usage: denoise <input.png> [output.png] [--strength 1-5]\n       denoise <input.png> [-o out.png] [--strength 1-5]"
        .to_string()
}

fn parse_strength(raw: &str) -> Result<u8, String> {
    let val: u8 = raw
        .parse()
        .map_err(|_| "strength must be a number between 1 and 5".to_string())?;
    if (1..=5).contains(&val) {
        Ok(val)
    } else {
        Err("strength must be between 1 (mild) and 5 (strong)".to_string())
    }
}

fn default_output_path(input: &Path, strength: u8) -> PathBuf {
    let parent = input.parent().unwrap_or_else(|| Path::new("."));
    let stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("image");

    parent.join(format!("{stem}_denoised_strength-{strength}.png"))
}

fn load_image(path: &Path) -> Result<DynamicImage, ImageError> {
    // Try magic-byte guessing first; fall back to extension-based lookup for robustness.
    let reader = ImageReader::open(path)?;
    let reader = match reader.with_guessed_format() {
        Ok(r) => r,
        Err(_) => {
            let format = ImageFormat::from_path(path)?;
            let mut r = ImageReader::open(path)?;
            r.set_format(format);
            r
        }
    };

    reader.decode()
}

fn denoise_image(image: DynamicImage, strength: u8) -> DynamicImage {
    let rgb = image.to_rgb8();

    // Strength scales filter size and blur sigma (gentler curve).
    let (median_kernel, sigma) = match strength {
        1 => (1, 0.10),
        2 => (2, 0.18),
        3 => (3, 0.30),
        4 => (4, 0.45),
        _ => (5, 0.60), // strength 5+
    };

    // Median filter knocks down hot pixels and isolated noise (parallelized).
    let median_pass = median_filter_parallel(&rgb, median_kernel);

    // Gentle Gaussian blur further smooths background noise without
    // eliminating small details.
    let blurred = gaussian_blur_f32(&median_pass, sigma);

    DynamicImage::ImageRgb8(blurred)
}

fn median_filter_parallel(img: &ImageBuffer<Rgb<u8>, Vec<u8>>, kernel: u32) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let w = img.width() as usize;
    let h = img.height() as usize;
    let r = (kernel / 2) as isize;

    let rows: Vec<Vec<[u8; 3]>> = (0..h)
        .into_par_iter()
        .map(|y| {
            let mut row = Vec::with_capacity(w);
            for x in 0..w {
                let mut med = [0u8; 3];
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

    let mut out = vec![0u8; w * h * 3];
    for (y, row) in rows.into_iter().enumerate() {
        for (x, pix) in row.into_iter().enumerate() {
            let idx = (y * w + x) * 3;
            out[idx..idx + 3].copy_from_slice(&pix);
        }
    }

    ImageBuffer::from_raw(w as u32, h as u32, out).expect("buffer size matches image dimensions")
}
