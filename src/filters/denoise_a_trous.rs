use image::{DynamicImage, ImageBuffer, Luma};
use image_dwt::{
    kernels::B3SplineKernel,
    layer::{WaveletLayer, WaveletLayerBuffer},
    recompose::{OutputLayer, RecomposableWaveletLayers},
    transform::ATrousTransform,
};

pub fn denoise_a_trous(image: DynamicImage) -> DynamicImage {
    let a_trous = ATrousTransform::new(&image, 6, B3SplineKernel);
    println!("a trous!");
    let a_trous_transformed = a_trous
        .into_iter()
        .map(|layer| {
            let buffer = layer.buffer.clone();
            let pixel_scale = layer.pixel_scale;
            if let Some(ps) = pixel_scale {
                println!("pixel_scale: {:?}", ps);
            };

            if pixel_scale.is_some_and(|scale| scale < 3) {
                // denoise
                let mut new_buffer =
                    ImageBuffer::<Luma<u16>, Vec<u16>>::new(image.width(), image.height());

                match buffer {
                    WaveletLayerBuffer::Grayscale { data } => {
                        for (x, y, pixel) in new_buffer.enumerate_pixels_mut() {
                            *pixel =
                                Luma([(data[[y as usize, x as usize]] * u16::MAX as f32) as u16])
                        }
                        let mut image = DynamicImage::ImageLuma16(new_buffer).to_luma8();
                        // Bilateral filter is a de-noising filter. Apply it to the image.
                        image = imageproc::filter::bilateral_filter(&image, 10, 10., 3.);

                        // Modify the raw buffer to contain the updated pixel values after filtering.
                        let mut data = data.clone();
                        for (x, y, pixel) in image.enumerate_pixels() {
                            data[[y as usize, x as usize]] = pixel.0[0] as f32 / u8::MAX as f32;
                        }
                        WaveletLayer {
                            buffer: WaveletLayerBuffer::Grayscale { data },
                            pixel_scale,
                        }
                    }
                    WaveletLayerBuffer::Rgb { data } => {
                        for (x, y, pixel) in new_buffer.enumerate_pixels_mut() {
                            *pixel =
                                Luma([(data[[y as usize, x as usize, 0]] * u16::MAX as f32) as u16])
                        }
                        let mut image = DynamicImage::ImageLuma16(new_buffer).to_luma8();
                        // Bilateral filter is a de-noising filter. Apply it to the image.
                        image = imageproc::filter::bilateral_filter(&image, 10, 10., 3.);

                        // Modify the raw buffer to contain the updated pixel values after filtering.
                        let mut data = data.clone();
                        for (x, y, pixel) in image.enumerate_pixels() {
                            data[[y as usize, x as usize, 0]] = pixel.0[0] as f32 / u8::MAX as f32;
                        }
                        WaveletLayer {
                            buffer: WaveletLayerBuffer::Rgb { data },
                            pixel_scale,
                        }
                    }
                }
            } else {
                layer
            }
        })
        .recompose_into_image(
            image.width() as usize,
            image.height() as usize,
            OutputLayer::Rgb,
        );
    a_trous_transformed
}
