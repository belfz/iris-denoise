use image::DynamicImage;

use denoise::{
    denoise_image, denoise_image_experimental, denoise_image_tiff, filters::sharpen_image_luma,
    sharpen_image, sharpen_image_tiff,
};

#[cfg_attr(test, mockall::automock)]
pub trait PipelineFns {
    fn denoise(&self, img: DynamicImage, strength: u8) -> DynamicImage;
    fn denoise_experimental(&self, img: DynamicImage, strength: u8) -> DynamicImage;
    fn sharpen(&self, img: DynamicImage, strength: u8) -> DynamicImage;
    fn sharpen_luma(&self, img: DynamicImage, strength: u8) -> DynamicImage;
}

pub struct StandardImagePipelines;
pub struct TiffPipelines;

impl PipelineFns for StandardImagePipelines {
    fn denoise(&self, img: DynamicImage, strength: u8) -> DynamicImage {
        println!("Standard image pipelines: denoise");
        denoise_image(img, strength)
    }

    fn denoise_experimental(&self, img: DynamicImage, strength: u8) -> DynamicImage {
        println!("Standard image pipelines: denoise_experimental");
        denoise_image_experimental(img, strength)
    }

    fn sharpen(&self, img: DynamicImage, strength: u8) -> DynamicImage {
        println!("Standard image pipelines: sharpen");
        sharpen_image(img, strength)
    }

    fn sharpen_luma(&self, img: DynamicImage, strength: u8) -> DynamicImage {
        println!("Standard image pipelines: sharpen_luma");
        sharpen_image_luma(img, strength)
    }
}

impl PipelineFns for TiffPipelines {
    fn denoise(&self, img: DynamicImage, strength: u8) -> DynamicImage {
        println!("TIFF pipelines: denoise");
        denoise_image_tiff(img, strength)
    }

    fn denoise_experimental(&self, img: DynamicImage, strength: u8) -> DynamicImage {
        // For TIFF, reuse the TIFF path for experimental until a dedicated one exists.
        println!("TIFF pipelines: denoise_experimental");
        denoise_image_tiff(img, strength)
    }

    fn sharpen(&self, img: DynamicImage, strength: u8) -> DynamicImage {
        println!("TIFF pipelines: sharpen");
        sharpen_image_tiff(img, strength)
    }

    fn sharpen_luma(&self, img: DynamicImage, strength: u8) -> DynamicImage {
        // TIFF uses the same RGB16 sharpen; no separate luma sharpening path yet.
        println!("TIFF pipelines: sharpen_luma");
        sharpen_image_tiff(img, strength)
    }
}
