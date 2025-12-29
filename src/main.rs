use std::path::{Path, PathBuf};
use std::process;
use std::sync::Arc;

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

type ImgFn = Arc<dyn Fn(DynamicImage, u8) -> DynamicImage + Send + Sync>;

#[derive(Clone)]
struct Pipelines {
    denoise: ImgFn,
    denoise_experimental: ImgFn,
    sharpen: ImgFn,
    sharpen_luma: ImgFn,
}

impl Default for Pipelines {
    fn default() -> Self {
        Self {
            denoise: Arc::new(|img, s| denoise_image(img, s)),
            denoise_experimental: Arc::new(|img, s| denoise_image_experimental(img, s)),
            sharpen: Arc::new(|img, s| sharpen_image(img, s)),
            sharpen_luma: Arc::new(|img, s| sharpen_image_luma(img, s)),
        }
    }
}

fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    run_with_pipelines(args, &Pipelines::default())
}

fn run_with_pipelines(
    args: Args,
    pipelines: &Pipelines,
) -> Result<(), Box<dyn std::error::Error>> {
    let output_path =
        args.output
            .unwrap_or_else(|| default_output_path(&args.input, args.strength, args.sharpen, args.experimental));

    let image = load_image(&args.input)?;
    let denoised = if args.experimental {
        (pipelines.denoise_experimental)(image, args.strength)
    } else {
        (pipelines.denoise)(image, args.strength)
    };
    let final_image = match (args.sharpen, args.experimental) {
        (Some(sharpen_strength), true) => (pipelines.sharpen_luma)(denoised, sharpen_strength),
        (Some(sharpen_strength), false) => (pipelines.sharpen)(denoised, sharpen_strength),
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
    use image::{DynamicImage, ImageBuffer, Rgb, RgbImage};
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

    #[test]
    fn run_invokes_experimental_and_luma_sharpen_when_flagged() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let exp_calls = Arc::new(AtomicUsize::new(0));
        let luma_calls = Arc::new(AtomicUsize::new(0));

        let exp_calls_clone = exp_calls.clone();
        let luma_calls_clone = luma_calls.clone();

        fn passthrough() -> DynamicImage {
            DynamicImage::ImageRgb8(ImageBuffer::from_fn(1, 1, |_x, _y| Rgb([0, 0, 0])))
        }

        let pipelines = Pipelines {
            denoise: Arc::new(|_img, _s| passthrough()),
            denoise_experimental: Arc::new(move |img, _s| {
                exp_calls_clone.fetch_add(1, Ordering::SeqCst);
                img
            }),
            sharpen: Arc::new(|_img, _s| passthrough()),
            sharpen_luma: Arc::new(move |img, _s| {
                luma_calls_clone.fetch_add(1, Ordering::SeqCst);
                img
            }),
        };

        let dir = temp_dir("denoise_exp_calls");
        let input = dir.join("input.png");
        let output = dir.join("out.png");
        write_test_image(&input);

        let args = Args {
            input,
            output: Some(output),
            strength: 3,
            sharpen: Some(2),
            experimental: true,
        };

        run_with_pipelines(args, &pipelines).expect("run experimental with spies");

        assert_eq!(exp_calls.load(Ordering::SeqCst), 1, "experimental denoise should be called once");
        assert_eq!(luma_calls.load(Ordering::SeqCst), 1, "luma sharpen should be called once");
    }

    #[test]
    fn run_invokes_standard_without_sharpen() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let denoise_calls = Arc::new(AtomicUsize::new(0));
        let sharpen_calls = Arc::new(AtomicUsize::new(0));

        let denoise_calls_clone = denoise_calls.clone();
        let sharpen_calls_clone = sharpen_calls.clone();

        fn passthrough() -> DynamicImage {
            DynamicImage::ImageRgb8(ImageBuffer::from_fn(1, 1, |_x, _y| Rgb([0, 0, 0])))
        }

        let pipelines = Pipelines {
            denoise: Arc::new(move |img, _s| {
                denoise_calls_clone.fetch_add(1, Ordering::SeqCst);
                img
            }),
            denoise_experimental: Arc::new(|_img, _s| passthrough()),
            sharpen: Arc::new(move |img, _s| {
                sharpen_calls_clone.fetch_add(1, Ordering::SeqCst);
                img
            }),
            sharpen_luma: Arc::new(|_img, _s| passthrough()),
        };

        let dir = temp_dir("denoise_std_calls");
        let input = dir.join("input.png");
        let output = dir.join("out.png");
        write_test_image(&input);

        let args = Args {
            input,
            output: Some(output),
            strength: 2,
            sharpen: None,
            experimental: false,
        };

        run_with_pipelines(args, &pipelines).expect("run standard without sharpen");

        assert_eq!(denoise_calls.load(Ordering::SeqCst), 1, "standard denoise should be called once");
        assert_eq!(sharpen_calls.load(Ordering::SeqCst), 0, "sharpen should not be called");
    }

    #[test]
    fn run_invokes_standard_with_sharpen() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let denoise_calls = Arc::new(AtomicUsize::new(0));
        let sharpen_calls = Arc::new(AtomicUsize::new(0));

        let denoise_calls_clone = denoise_calls.clone();
        let sharpen_calls_clone = sharpen_calls.clone();

        fn passthrough() -> DynamicImage {
            DynamicImage::ImageRgb8(ImageBuffer::from_fn(1, 1, |_x, _y| Rgb([0, 0, 0])))
        }

        let pipelines = Pipelines {
            denoise: Arc::new(move |img, _s| {
                denoise_calls_clone.fetch_add(1, Ordering::SeqCst);
                img
            }),
            denoise_experimental: Arc::new(|_img, _s| passthrough()),
            sharpen: Arc::new(move |img, _s| {
                sharpen_calls_clone.fetch_add(1, Ordering::SeqCst);
                img
            }),
            sharpen_luma: Arc::new(|_img, _s| passthrough()),
        };

        let dir = temp_dir("denoise_std_calls_sharpen");
        let input = dir.join("input.png");
        let output = dir.join("out.png");
        write_test_image(&input);

        let args = Args {
            input,
            output: Some(output),
            strength: 4,
            sharpen: Some(2),
            experimental: false,
        };

        run_with_pipelines(args, &pipelines).expect("run standard with sharpen");

        assert_eq!(denoise_calls.load(Ordering::SeqCst), 1, "standard denoise should be called once");
        assert_eq!(sharpen_calls.load(Ordering::SeqCst), 1, "sharpen should be called once");
    }

    #[test]
    fn run_invokes_experimental_without_sharpen() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let exp_calls = Arc::new(AtomicUsize::new(0));
        let luma_calls = Arc::new(AtomicUsize::new(0));

        let exp_calls_clone = exp_calls.clone();
        let luma_calls_clone = luma_calls.clone();

        fn passthrough() -> DynamicImage {
            DynamicImage::ImageRgb8(ImageBuffer::from_fn(1, 1, |_x, _y| Rgb([0, 0, 0])))
        }

        let pipelines = Pipelines {
            denoise: Arc::new(|_img, _s| passthrough()),
            denoise_experimental: Arc::new(move |img, _s| {
                exp_calls_clone.fetch_add(1, Ordering::SeqCst);
                img
            }),
            sharpen: Arc::new(|_img, _s| passthrough()),
            sharpen_luma: Arc::new(move |img, _s| {
                luma_calls_clone.fetch_add(1, Ordering::SeqCst);
                img
            }),
        };

        let dir = temp_dir("denoise_exp_calls_no_sharp");
        let input = dir.join("input.png");
        let output = dir.join("out.png");
        write_test_image(&input);

        let args = Args {
            input,
            output: Some(output),
            strength: 3,
            sharpen: None,
            experimental: true,
        };

        run_with_pipelines(args, &pipelines).expect("run experimental without sharpen");

        assert_eq!(exp_calls.load(Ordering::SeqCst), 1, "experimental denoise should be called once");
        assert_eq!(luma_calls.load(Ordering::SeqCst), 0, "luma sharpen should not be called");
    }
}
