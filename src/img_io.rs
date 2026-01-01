use std::error::Error;
use std::path::Path;

use image::{DynamicImage, ImageFormat, ImageReader};

#[derive(Clone, Copy, Debug)]
pub enum TiffBitDepth {
    U8,
    U16,
}

#[derive(Clone, Copy, Debug)]
pub enum ImageSourceFormat {
    StandardImage(ImageFormat),
    Tiff { bit_depth: TiffBitDepth },
}

pub struct LoadedImage {
    pub image: DynamicImage,
    pub format: ImageSourceFormat,
}

/// Load an image, preferring format auto-detection and tracking format/bit depth for saving.
pub fn load_image_with_meta(path: &Path) -> Result<LoadedImage, Box<dyn Error>> {
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

    let format = reader.format().ok_or(format!("Failed to get image format from {}", path.display()))?;
    let image = reader.decode()?;

    if format == ImageFormat::Tiff {
        let bit_depth = match image {
            DynamicImage::ImageRgb16(_)
            | DynamicImage::ImageRgba16(_)
            | DynamicImage::ImageLuma16(_)
            | DynamicImage::ImageLumaA16(_) => TiffBitDepth::U16,
            _ => TiffBitDepth::U8,
        };
        Ok(LoadedImage {
            image,
            format: ImageSourceFormat::Tiff { bit_depth },
        })
    } else {
        Ok(LoadedImage {
            image,
            format: ImageSourceFormat::StandardImage(format),
        })
    }
}

/// Save image respecting the source format (TIFF stays TIFF, others use their format).
pub fn save_image_with_format(
    path: &Path,
    image: &DynamicImage,
    format: &ImageSourceFormat,
) -> Result<(), Box<dyn Error>> {
    match format {
        ImageSourceFormat::StandardImage(fmt) => {
            image.to_rgb8().save_with_format(path, *fmt)?;
            Ok(())
        }
        ImageSourceFormat::Tiff { bit_depth } => {
            let out = match bit_depth {
                TiffBitDepth::U8 => DynamicImage::ImageRgb8(image.to_rgb8()),
                TiffBitDepth::U16 => DynamicImage::ImageRgb16(image.to_rgb16()),
            };
            out.save_with_format(path, ImageFormat::Tiff)?;
            Ok(())
        }
    }
}
