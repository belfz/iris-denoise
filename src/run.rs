use std::path::{Path, PathBuf};

use image::ImageFormat;

use crate::cli::Args;
use crate::img_io::{ImageSourceFormat, LoadedImage, load_image_with_meta, save_image_with_format};
use crate::pipelines::{PipelineFns, StandardImagePipelines, TiffPipelines};

pub fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    let loaded = load_image_with_meta(&args.input)?;

    match loaded.format {
        ImageSourceFormat::Tiff { .. } => run_with_pipelines(args, &TiffPipelines, loaded),
        ImageSourceFormat::StandardImage(_) => run_with_pipelines(args, &StandardImagePipelines, loaded),
    }
}

fn print_loaded_details(loaded: &LoadedImage, output_path: &Path) {
    println!("Loaded format: {}", match loaded.format {
        ImageSourceFormat::Tiff { bit_depth } => format!("TIFF: {:?}", bit_depth),
        ImageSourceFormat::StandardImage(_) => format!("PNG"),
    });
    println!("Output path: {}", output_path.display());
}

fn run_with_pipelines(
    args: Args,
    pipelines: &dyn PipelineFns,
    loaded: LoadedImage,
) -> Result<(), Box<dyn std::error::Error>> {
    let output_path = choose_output_path(
        &args.input,
        args.output.as_ref(),
        args.strength,
        args.sharpen,
        args.experimental,
        &loaded.format,
    );

    print_loaded_details(&loaded, &output_path);

    let final_image = {
        let denoised = if args.experimental {
            pipelines.denoise_experimental(loaded.image, args.strength)
        } else {
            pipelines.denoise(loaded.image, args.strength)
        };
        match (args.sharpen, args.experimental) {
            (Some(sharpen_strength), true) => pipelines.sharpen_luma(denoised, sharpen_strength),
            (Some(sharpen_strength), false) => pipelines.sharpen(denoised, sharpen_strength),
            _ => denoised,
        }
    };

    save_image_with_format(&output_path, &final_image, &loaded.format)?;

    println!("Processed image written to {}", output_path.display());
    Ok(())
}

fn default_output_path(
    input: &Path,
    strength: u8,
    sharpen: Option<u8>,
    experimental: bool,
) -> PathBuf {
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

fn choose_output_path(
    input: &Path,
    user_output: Option<&PathBuf>,
    strength: u8,
    sharpen: Option<u8>,
    experimental: bool,
    format: &ImageSourceFormat,
) -> PathBuf {
    let mut base = user_output
        .cloned()
        .unwrap_or_else(|| default_output_path(input, strength, sharpen, experimental));

    let is_tiff = match format {
        ImageSourceFormat::Tiff { .. } => true,
        ImageSourceFormat::StandardImage(fmt) => *fmt == ImageFormat::Tiff,
    };
    if is_tiff {
        base.set_extension("tiff");
    }

    base
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

    fn create_dir_and_input() -> (PathBuf, PathBuf, LoadedImage) {
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
        let loaded = load_image_with_meta(&input).expect("load test image");

        (input, output, loaded)
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

        let (input, output, loaded) = create_dir_and_input();

        let args = Args {
            input,
            output: Some(output),
            strength: 3,
            sharpen: Some(2),
            experimental: true,
        };

        run_with_pipelines(args, &pipelines, loaded).expect("run experimental with sharpen");
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

        let (input, output, loaded) = create_dir_and_input();

        let args = Args {
            input,
            output: Some(output),
            strength: 2,
            sharpen: None,
            experimental: false,
        };

        run_with_pipelines(args, &pipelines, loaded).expect("run standard without sharpen");
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

        let (input, output, loaded) = create_dir_and_input();

        let args = Args {
            input,
            output: Some(output),
            strength: 4,
            sharpen: Some(2),
            experimental: false,
        };

        run_with_pipelines(args, &pipelines, loaded).expect("run standard with sharpen");
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

        let (input, output, loaded) = create_dir_and_input();

        let args = Args {
            input,
            output: Some(output),
            strength: 3,
            sharpen: None,
            experimental: true,
        };

        run_with_pipelines(args, &pipelines, loaded).expect("run experimental without sharpen");
    }
}
