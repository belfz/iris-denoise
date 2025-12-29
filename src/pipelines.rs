use image::DynamicImage;

use denoise::{
    denoise_image,
    denoise_image_experimental,
    filters::sharpen_image_luma,
    sharpen_image,
};

#[cfg_attr(test, mockall::automock)]
pub trait PipelineFns {
    fn denoise(&self, img: DynamicImage, strength: u8) -> DynamicImage;
    fn denoise_experimental(&self, img: DynamicImage, strength: u8) -> DynamicImage;
    fn sharpen(&self, img: DynamicImage, strength: u8) -> DynamicImage;
    fn sharpen_luma(&self, img: DynamicImage, strength: u8) -> DynamicImage;
}

pub struct RealPipelines;

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

