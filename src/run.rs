use std::path::{Path, PathBuf};

use denoise::load_image;

use crate::cli::Args;
use crate::pipelines::{PipelineFns, RealPipelines};

pub fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    let mut pipelines = RealPipelines;
    run_with_pipelines(args, &mut pipelines)
}

pub fn run_with_pipelines(
    args: Args,
    pipelines: &mut dyn PipelineFns,
) -> Result<(), Box<dyn std::error::Error>> {
    let output_path =
        args.output
            .as_ref()
            .cloned()
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
    use crate::pipelines::MockPipelineFns;
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
        let mut dir = std::env::temp_dir();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        dir.push(format!("test_{}_{}", std::process::id(), nanos));
        fs::create_dir_all(&dir).expect("create temp dir");

        let input = dir.join("input.png");
        let output = dir.join("out.png");
        write_test_image(&input);
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

