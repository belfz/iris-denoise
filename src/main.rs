use std::path::{Path, PathBuf};
use std::process;

use clap::Parser;
use image::{DynamicImage, ImageBuffer, ImageError, ImageFormat, ImageReader, Rgb};
use imageproc::filter::gaussian_blur_f32;
use rayon::prelude::*;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input image path
    #[arg(value_name = "INPUT")]
    input: PathBuf,

    /// Output image path (defaults to auto-named when omitted)
    #[arg(short, long, value_name = "OUTPUT")]
    output: Option<PathBuf>,

    /// Denoise strength 1-5 (mild to strong)
    #[arg(
        short,
        long,
        value_name = "STRENGTH",
        default_value_t = 3,
        value_parser = clap::value_parser!(u8).range(1..=5)
    )]
    strength: u8,

    /// Enable sharpening; optional strength 1-5, defaults to 3 when flag is given without a value
    #[arg(
        long,
        value_name = "STRENGTH",
        num_args = 0..=1,
        default_missing_value = "3",
        value_parser = clap::value_parser!(u8).range(1..=5)
    )]
    sharpen: Option<u8>,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let output_path =
        args.output
            .unwrap_or_else(|| default_output_path(&args.input, args.strength, args.sharpen));

    let image = load_image(&args.input)?;
    let denoised = denoise_image(image, args.strength);
    let final_image = if let Some(sharpen_strength) = args.sharpen {
        sharpen_image(denoised, sharpen_strength)
    } else {
        denoised
    };
    final_image.save(&output_path)?;

    println!(
        "Processed image written to {}",
        output_path.display()
    );
    Ok(())
}

fn default_output_path(input: &Path, strength: u8, sharpen: Option<u8>) -> PathBuf {
    let parent = input.parent().unwrap_or_else(|| Path::new("."));
    let stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("image");

    let sharpen_suffix = sharpen.map(|s| format!("_sharpen-{s}")).unwrap_or_default();
    parent.join(format!(
        "{stem}_denoised_strength-{strength}{sharpen_suffix}.png"
    ))
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

fn sharpen_image(image: DynamicImage, strength: u8) -> DynamicImage {
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
