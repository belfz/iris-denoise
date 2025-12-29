use std::path::{Path, PathBuf};
use std::process;

use clap::Parser;

use denoise::{
    denoise_image,
    denoise_image_experimental,
    load_image,
    sharpen_image,
};
use denoise::filters::sharpen_image_luma;

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

    /// Use the experimental astrophotography-focused denoiser
    #[arg(long)]
    experimental: bool,
}

fn main() {
    let args = Args::parse();
    if let Err(err) = run(args) {
        eprintln!("error: {err}");
        process::exit(1);
    }
}

fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    let output_path =
        args.output
            .unwrap_or_else(|| default_output_path(&args.input, args.strength, args.sharpen, args.experimental));

    let image = load_image(&args.input)?;
    let denoised = if args.experimental {
        denoise_image_experimental(image, args.strength)
    } else {
        denoise_image(image, args.strength)
    };
    let final_image = match (args.sharpen, args.experimental) {
        (Some(sharpen_strength), true) => sharpen_image_luma(denoised, sharpen_strength),
        (Some(sharpen_strength), false) => sharpen_image(denoised, sharpen_strength),
        _ => denoised,
    };
    final_image.save(&output_path)?;

    println!(
        "Processed image written to {}",
        output_path.display()
    );
    Ok(())
}

fn default_output_path(input: &Path, strength: u8, sharpen: Option<u8>, experimental: bool) -> PathBuf {
    let parent = input.parent().unwrap_or_else(|| Path::new("."));
    let stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("image");

    let experimental_suffix = if experimental { "_experimental" } else { "" };
    let sharpen_suffix = sharpen.map(|s| format!("_sharpen-{s}")).unwrap_or_default();
    parent.join(format!(
        "{stem}_denoised_strength-{strength}{experimental_suffix}{sharpen_suffix}.png"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb, RgbImage};
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(prefix: &str) -> PathBuf {
        let mut dir = std::env::temp_dir();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        dir.push(format!("{prefix}_{}_{}", std::process::id(), nanos));
        fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    fn write_test_image(path: &Path) {
        let mut img: RgbImage = ImageBuffer::new(3, 2);
        // Row 0
        img.put_pixel(0, 0, Rgb([10, 20, 30]));
        img.put_pixel(1, 0, Rgb([40, 50, 60]));
        img.put_pixel(2, 0, Rgb([90, 80, 70]));
        // Row 1 with a bright "star"
        img.put_pixel(0, 1, Rgb([200, 200, 200]));
        img.put_pixel(1, 1, Rgb([30, 10, 5]));
        img.put_pixel(2, 1, Rgb([5, 10, 30]));
        img.save(path).expect("save test image");
    }

    fn load_rgb(path: &Path) -> RgbImage {
        image::open(path).expect("open image").to_rgb8()
    }

    #[test]
    fn run_standard_denoise_no_sharpen_matches_expected() {
        let dir = temp_dir("denoise_std");
        let input = dir.join("input.png");
        let output = dir.join("out.png");
        write_test_image(&input);

        let args = Args {
            input: input.clone(),
            output: Some(output.clone()),
            strength: 3,
            sharpen: None,
            experimental: false,
        };

        run(args).expect("run standard");

        let produced = load_rgb(&output);
        let expected = denoise_image(load_image(&input).unwrap(), 3).to_rgb8();
        assert_eq!(produced.as_raw(), expected.as_raw());
    }

    #[test]
    fn run_experimental_with_luma_sharpen_matches_expected() {
        let dir = temp_dir("denoise_exp");
        let input = dir.join("input.png");
        let output = dir.join("out.png");
        write_test_image(&input);

        let args = Args {
            input: input.clone(),
            output: Some(output.clone()),
            strength: 3,
            sharpen: Some(3),
            experimental: true,
        };

        run(args).expect("run experimental");

        let produced = load_rgb(&output);
        let expected = sharpen_image_luma(
            denoise_image_experimental(load_image(&input).unwrap(), 3),
            3,
        )
        .to_rgb8();
        assert_eq!(produced.as_raw(), expected.as_raw());
    }
}
