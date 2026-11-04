//! The frame check, and the buffer-to-image regions of each plane. A word row holds what
//! `copy::nv12_layout` and `copy::p010_layout` state.
//!
//! An encoder reads its input past the frame, up to the coded size. The regions below fill that
//! area with the frame's last column and last row, on the device, so it codes in few bits.

use ash::vk;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use crate::decode::{PixelFormat, PlaneForm};
use crate::encode::{EncodeConfig, EncodeFrame};
use crate::error::EncodeError;
use crate::vulkan::copy::{FrameArea, PlaneLayout, nv12_layout, p010_layout};

/// One config's plane layouts.
#[derive(Clone, Copy, Debug)]
pub(super) struct FrameLayout {
    pub(super) format: PixelFormat,
    pub(super) size: (u32, u32),
    pub(super) luma: PlaneLayout,
    pub(super) chroma: PlaneLayout,
}

/// The copy regions of each plane. Every frame of a session shares them.
#[derive(Clone, Debug)]
pub(super) struct PlaneRegions {
    pub(super) luma: Vec<vk::BufferImageCopy>,
    pub(super) chroma: Vec<vk::BufferImageCopy>,
}

impl FrameLayout {
    pub(super) fn new(config: &EncodeConfig) -> Self {
        let area = FrameArea {
            left: 0,
            top: 0,
            width: config.width,
            height: config.height,
        };
        let [luma, chroma] = match config.format {
            PixelFormat::Nv12 => nv12_layout(area),
            PixelFormat::P010 => p010_layout(area),
        };
        Self {
            format: config.format,
            size: (config.width, config.height),
            luma,
            chroma,
        }
    }

    /// `(luma, chroma)` bytes per texel.
    const fn texel_bytes(&self) -> (u64, u64) {
        match self.format {
            PixelFormat::Nv12 => (1, 2),
            PixelFormat::P010 => (2, 4),
        }
    }

    /// The regions that fill an input image of `image` luma samples from a frame's planes.
    pub(super) fn regions(&self, image: (u32, u32)) -> PlaneRegions {
        let (luma_bytes, chroma_bytes) = self.texel_bytes();
        PlaneRegions {
            luma: plane_regions(&self.luma, luma_bytes, image),
            chroma: plane_regions(&self.chroma, chroma_bytes, (image.0 / 2, image.1 / 2)),
        }
    }

    /// Refuses a frame in another form, format, size, or memory layout, by name.
    pub(super) fn check_frame<R>(&self, frame: &EncodeFrame<'_, R>) -> Result<(), EncodeError>
    where
        R: Runtime<DType = DType>,
    {
        let expected = |shape: [usize; 2]| format!("{shape:?} in {:?} words", DType::U32);
        if frame.plane_form() != PlaneForm::Words {
            return Err(EncodeError::SurfaceMismatch {
                plane: "luma",
                expected: expected(self.luma.shape),
                found: format!("{:?} in sample form", frame.y().shape()),
            });
        }
        if let Some((format, display)) = frame.word_layout()
            && (format != self.format || (display.width(), display.height()) != self.size)
        {
            return Err(EncodeError::SurfaceMismatch {
                plane: "luma",
                expected: format!("{}x{} {}", self.size.0, self.size.1, self.format),
                found: format!("{}x{} {format}", display.width(), display.height()),
            });
        }
        check_plane("luma", frame.y(), self.luma.shape)?;
        check_plane("chroma", frame.uv(), self.chroma.shape)
    }
}

/// Refuses a plane that is not a dense U32 tensor of `shape` from the start of its buffer.
fn check_plane<R>(
    plane: &'static str,
    tensor: &Tensor<R>,
    shape: [usize; 2],
) -> Result<(), EncodeError>
where
    R: Runtime<DType = DType>,
{
    if tensor.shape() != shape || tensor.dtype() != DType::U32 {
        return Err(EncodeError::SurfaceMismatch {
            plane,
            expected: format!("{shape:?} in {:?}", DType::U32),
            found: format!("{:?} in {:?}", tensor.shape(), tensor.dtype()),
        });
    }
    if !tensor.is_contiguous() || tensor.offset() != 0 {
        return Err(EncodeError::PlaneStrides {
            plane,
            strides: tensor.strides().to_vec(),
            columns: shape[1],
        });
    }
    Ok(())
}

/// One region of `plane` at image position `(x, y)`, from the buffer texel `(column, row)`.
fn region(
    plane: &PlaneLayout,
    texel_bytes: u64,
    (column, row): (u32, u32),
    (x, y): (u32, u32),
    (width, height): (u32, u32),
) -> vk::BufferImageCopy {
    let pitch = u64::from(plane.row_texels) * texel_bytes;
    vk::BufferImageCopy {
        buffer_offset: u64::from(row) * pitch + u64::from(column) * texel_bytes,
        buffer_row_length: plane.row_texels,
        buffer_image_height: 0,
        image_subresource: vk::ImageSubresourceLayers {
            aspect_mask: plane.aspect,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        },
        image_offset: vk::Offset3D {
            x: x as i32,
            y: y as i32,
            z: 0,
        },
        image_extent: vk::Extent3D {
            width,
            height,
            depth: 1,
        },
    }
}

/// The regions that fill one plane of an image of `image` texels: the plane itself, then its last
/// column, its last row, and its last texel across the rest.
fn plane_regions(
    plane: &PlaneLayout,
    texel_bytes: u64,
    image: (u32, u32),
) -> Vec<vk::BufferImageCopy> {
    let (width, height) = plane.extent;
    let (last_column, last_row) = (width - 1, height - 1);
    let mut regions = vec![region(plane, texel_bytes, (0, 0), (0, 0), (width, height))];
    for x in width..image.0 {
        regions.push(region(
            plane,
            texel_bytes,
            (last_column, 0),
            (x, 0),
            (1, height),
        ));
    }
    for y in height..image.1 {
        regions.push(region(
            plane,
            texel_bytes,
            (0, last_row),
            (0, y),
            (width, 1),
        ));
        for x in width..image.0 {
            regions.push(region(
                plane,
                texel_bytes,
                (last_column, last_row),
                (x, y),
                (1, 1),
            ));
        }
    }
    regions
}

#[cfg(test)]
mod tests {
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    use super::*;
    use crate::geometry::Dimensions;
    use crate::time::Rational;

    fn config(width: u32, height: u32, format: PixelFormat) -> EncodeConfig {
        let rate = Rational::new(30, 1).unwrap_or(Rational::ONE);
        let base = Rational::new(1, 30).unwrap_or(Rational::ONE);
        EncodeConfig::new(
            crate::encode::EncodeCodec::Hevc,
            width,
            height,
            format,
            rate,
            base,
        )
    }

    fn words(shape: [usize; 2]) -> Tensor<CpuRuntime> {
        Tensor::empty(&shape, DType::U32, &CpuDevice::new()).expect("a plane")
    }

    #[test]
    fn an_aligned_plane_takes_one_region() {
        let layout = FrameLayout::new(&config(640, 368, PixelFormat::Nv12));
        let regions = plane_regions(&layout.luma, 1, (640, 368));
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].buffer_offset, 0);
        assert_eq!(regions[0].buffer_row_length, 640);
        assert_eq!(regions[0].image_extent.height, 368);
    }

    #[test]
    fn a_layout_fills_both_planes_of_the_image() {
        let layout = FrameLayout::new(&config(640, 360, PixelFormat::Nv12));
        let regions = layout.regions((640, 368));
        assert_eq!(regions.luma.len(), 9);
        assert_eq!(regions.chroma.len(), 5);
        assert_eq!(regions.chroma[4].image_offset.y, 183);
    }

    #[test]
    fn rows_past_the_frame_repeat_its_last_row() {
        let layout = FrameLayout::new(&config(640, 360, PixelFormat::Nv12));
        let luma = plane_regions(&layout.luma, 1, (640, 368));
        assert_eq!(luma.len(), 9);
        assert_eq!(luma[1].buffer_offset, 359 * 640);
        assert_eq!(luma[1].image_offset.y, 360);
        assert_eq!(luma[8].buffer_offset, 359 * 640);
        assert_eq!(luma[8].image_offset.y, 367);
        let chroma = plane_regions(&layout.chroma, 2, (320, 184));
        assert_eq!(chroma.len(), 5);
        assert_eq!(chroma[1].buffer_offset, 179 * 640);
        assert_eq!(chroma[1].image_offset.y, 180);
        assert_eq!(
            chroma[1].image_subresource.aspect_mask,
            vk::ImageAspectFlags::PLANE_1
        );
    }

    #[test]
    fn columns_past_the_frame_repeat_its_last_column_and_the_corner_its_last_texel() {
        let layout = FrameLayout::new(&config(322, 180, PixelFormat::P010));
        let (luma_bytes, _) = layout.texel_bytes();
        let regions = plane_regions(&layout.luma, luma_bytes, (324, 182));
        // The plane, 2 columns, then for each of 2 rows the row and 2 corner texels.
        assert_eq!(regions.len(), 1 + 2 + 2 * 3);
        assert_eq!(regions[1].buffer_offset, 321 * 2);
        assert_eq!(regions[1].image_offset.x, 322);
        assert_eq!(regions[1].image_extent.height, 180);
        let corner = regions[regions.len() - 1];
        let pitch = u64::from(layout.luma.row_texels) * 2;
        assert_eq!(corner.buffer_offset, 179 * pitch + 321 * 2);
        assert_eq!((corner.image_offset.x, corner.image_offset.y), (323, 181));
        assert_eq!(
            (corner.image_extent.width, corner.image_extent.height),
            (1, 1)
        );
    }

    #[test]
    fn each_region_stays_inside_its_plane_buffer() {
        for (format, width, height) in [
            (PixelFormat::Nv12, 1366, 768),
            (PixelFormat::P010, 1918, 1078),
        ] {
            let layout = FrameLayout::new(&config(width, height, format));
            let (luma_bytes, chroma_bytes) = layout.texel_bytes();
            for (plane, bytes, image) in [
                (&layout.luma, luma_bytes, (1920, 1088)),
                (&layout.chroma, chroma_bytes, (960, 544)),
            ] {
                let buffer = (plane.shape[0] * plane.shape[1] * 4) as u64;
                let pitch = u64::from(plane.row_texels) * bytes;
                for region in plane_regions(plane, bytes, image) {
                    let rows = u64::from(region.image_extent.height);
                    let columns = u64::from(region.image_extent.width);
                    let end = region.buffer_offset + (rows - 1) * pitch + columns * bytes;
                    assert!(end <= buffer, "{format} region ends at {end} of {buffer}");
                    assert!(region.buffer_offset.is_multiple_of(bytes));
                }
            }
        }
    }

    #[test]
    fn a_word_frame_of_the_config_passes() {
        let layout = FrameLayout::new(&config(640, 360, PixelFormat::Nv12));
        let (y, uv) = (words([360, 160]), words([180, 160]));
        let display = Dimensions::new(640, 360).expect("a size");
        let frame = EncodeFrame::from_packed_planes(&y, &uv, PixelFormat::Nv12, display);
        assert_eq!(layout.check_frame(&frame), Ok(()));
    }

    #[test]
    fn a_sample_frame_another_size_or_a_short_plane_is_refused_by_name() {
        let layout = FrameLayout::new(&config(640, 360, PixelFormat::Nv12));
        let (y, uv) = (words([360, 160]), words([180, 160]));
        let samples = EncodeFrame::new(&y, &uv);
        let error = layout.check_frame(&samples).expect_err("refused");
        assert!(error.to_string().contains("sample form"), "{error}");

        let other = Dimensions::new(320, 180).expect("a size");
        let frame = EncodeFrame::from_packed_planes(&y, &uv, PixelFormat::Nv12, other);
        let error = layout.check_frame(&frame).expect_err("refused");
        assert!(error.to_string().contains("320x180"), "{error}");

        let display = Dimensions::new(640, 360).expect("a size");
        let frame = EncodeFrame::from_packed_planes(&y, &uv, PixelFormat::P010, display);
        assert!(layout.check_frame(&frame).is_err());

        let short = words([100, 160]);
        let frame = EncodeFrame::from_packed_planes(&short, &uv, PixelFormat::Nv12, display);
        assert!(matches!(
            layout.check_frame(&frame),
            Err(EncodeError::SurfaceMismatch { plane: "luma", .. })
        ));
    }

    #[test]
    fn a_narrowed_plane_is_refused_because_its_rows_have_a_pitch() {
        let layout = FrameLayout::new(&config(640, 360, PixelFormat::Nv12));
        let wide = words([360, 200]);
        let narrow = wide.narrow(1, 0, 160).expect("a view");
        let uv = words([180, 160]);
        let display = Dimensions::new(640, 360).expect("a size");
        let frame = EncodeFrame::from_packed_planes(&narrow, &uv, PixelFormat::Nv12, display);
        assert!(matches!(
            layout.check_frame(&frame),
            Err(EncodeError::PlaneStrides { plane: "luma", .. })
        ));
    }
}
