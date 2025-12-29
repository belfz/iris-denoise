use std::path::Path;

use image::{DynamicImage, ImageError, ImageFormat, ImageReader};

/// Load an image, preferring format auto-detection and falling back to extension lookup.
pub fn load_image(path: &Path) -> Result<DynamicImage, ImageError> {
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

