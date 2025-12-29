use std::path::{Path, PathBuf};
use std::process;

use clap::Parser;
use image::DynamicImage;

use denoise::{
    denoise_image,
    denoise_image_experimental,
    filters::sharpen_image_luma,
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
    let args = Args::parse();
    if let Err(err) = run(args) {
        eprintln!("error: {err}");
        process::exit(1);
    }
}

#[cfg_attr(test, mockall::automock)]
trait PipelineFns {
    fn denoise(&self, img: DynamicImage, strength: u8) -> DynamicImage;
    fn denoise_experimental(&self, img: DynamicImage, strength: u8) -> DynamicImage;
    fn sharpen(&self, img: DynamicImage, strength: u8) -> DynamicImage;
    fn sharpen_luma(&self, img: DynamicImage, strength: u8) -> DynamicImage;
}

struct RealPipelines;

impl PipelineFns for RealPipelines {
    fn denoise(&self, img: DynamicImage, strength: u8) -> DynamicImage {
        denoise_image(img, strength)
    }

    fn denoise_experimental(&self, img: DynamicImage, strength: u8) -> DynamicImage {
        denoise_image_experimental(img, strength)
    }

    fn sharpen(&self, img: DynamicImage, strength: u8) -> DynamicImage {
        sharpen_image(img, strength)
    }

    fn sharpen_luma(&self, img: DynamicImage, strength: u8) -> DynamicImage {
        sharpen_image_luma(img, strength)
    }
}

fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    let mut pipelines = RealPipelines;
    run_with_pipelines(args, &mut pipelines)
}

fn run_with_pipelines(
    args: Args,
    pipelines: &mut dyn PipelineFns,
) -> Result<(), Box<dyn std::error::Error>> {
    let output_path =
        args.output
            .unwrap_or_else(|| default_output_path(&args.input, args.strength, args.sharpen, args.experimental));

    let image = load_image(&args.input)?;
    let denoised = if args.experimental {
        pipelines.denoise_experimental(image, args.strength)
    } else {
        pipelines.denoise(image, args.strength)
    };
    let final_image = match (args.sharpen, args.experimental) {
        (Some(sharpen_strength), true) => pipelines.sharpen_luma(denoised, sharpen_strength),
        (Some(sharpen_strength), false) => pipelines.sharpen(denoised, sharpen_strength),
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
    use mockall::predicate::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

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

    fn create_dir_and_input() -> (PathBuf, PathBuf) {
        // create a temporary directory for the test
        let mut dir = std::env::temp_dir();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        dir.push(format!("test_{}_{}", std::process::id(), nanos));
        fs::create_dir_all(&dir).expect("create temp dir");

        // create a temporary input and output image in the directory
        let input = dir.join("input.png");
        let output = dir.join("out.png");
        
        // write the test image to the input
        write_test_image(&input);
        
        // return the input and output paths
        (input, output)
    }

    #[test]
    fn run_invokes_experimental_and_luma_sharpen_when_flagged() {
        let mut pipelines = MockPipelineFns::new();
        pipelines
            .expect_denoise_experimental()
            .times(1)
            .with(always(), eq(3))
            .returning(|img, _| img);
        pipelines
            .expect_sharpen_luma()
            .times(1)
            .with(always(), eq(2))
            .returning(|img, _| img);
        pipelines.expect_denoise().times(0);
        pipelines.expect_sharpen().times(0);

        let (input, output) = create_dir_and_input();

        let args = Args {
            input,
            output: Some(output),
            strength: 3,
            sharpen: Some(2),
            experimental: true,
        };

        run_with_pipelines(args, &mut pipelines).expect("run experimental with spies");
    }

    #[test]
    fn run_invokes_standard_without_sharpen() {
        let mut pipelines = MockPipelineFns::new();
        pipelines
            .expect_denoise()
            .times(1)
            .with(always(), eq(2))
            .returning(|img, _| img);
        pipelines.expect_sharpen().times(0);
        pipelines.expect_denoise_experimental().times(0);
        pipelines.expect_sharpen_luma().times(0);

        let (input, output) = create_dir_and_input();

        let args = Args {
            input,
            output: Some(output),
            strength: 2,
            sharpen: None,
            experimental: false,
        };

        run_with_pipelines(args, &mut pipelines).expect("run standard without sharpen");
    }

    #[test]
    fn run_invokes_standard_with_sharpen() {
        let mut pipelines = MockPipelineFns::new();
        pipelines
            .expect_denoise()
            .times(1)
            .with(always(), eq(4))
            .returning(|img, _| img);
        pipelines
            .expect_sharpen()
            .times(1)
            .with(always(), eq(2))
            .returning(|img, _| img);
        pipelines.expect_denoise_experimental().times(0);
        pipelines.expect_sharpen_luma().times(0);

        let (input, output) = create_dir_and_input();

        let args = Args {
            input,
            output: Some(output),
            strength: 4,
            sharpen: Some(2),
            experimental: false,
        };

        run_with_pipelines(args, &mut pipelines).expect("run standard with sharpen");
    }

    #[test]
    fn run_invokes_experimental_without_sharpen() {
        let mut pipelines = MockPipelineFns::new();
        pipelines
            .expect_denoise_experimental()
            .times(1)
            .with(always(), eq(3))
            .returning(|img, _| img);
        pipelines.expect_sharpen_luma().times(0);
        pipelines.expect_denoise().times(0);
        pipelines.expect_sharpen().times(0);
        pipelines.expect_sharpen_luma().times(0);

        let (input, output) = create_dir_and_input();

        let args = Args {
            input,
            output: Some(output),
            strength: 3,
            sharpen: None,
            experimental: true,
        };

        run_with_pipelines(args, &mut pipelines).expect("run experimental without sharpen");
    }
}
