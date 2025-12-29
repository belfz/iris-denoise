use std::path::{Path, PathBuf};
use std::process;

use clap::Parser;

use denoise::{
    denoise_image,
    denoise_image_experimental,
    load_image,
    sharpen_image,
};

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
    if let Err(err) = run() {
        eprintln!("error: {err}");
        process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let output_path =
        args.output
            .unwrap_or_else(|| default_output_path(&args.input, args.strength, args.sharpen, args.experimental));

    let image = load_image(&args.input)?;
    let denoised = if args.experimental {
        denoise_image_experimental(image, args.strength)
    } else {
        denoise_image(image, args.strength)
    };
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
