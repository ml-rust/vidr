//! The plane sizes of one config, the surface that holds them, and the frame
//! checks against them. No driver calls.

use numr::dtype::DType;
use numr::runtime::cuda::CudaRuntime;

use crate::decode::{PixelFormat, frame::chroma_shape};
use crate::encode::{EncodeConfig, EncodeFrame};
use crate::error::EncodeError;

/// Surface row alignment. NVENC needs a multiple of 4. This is the alignment
/// ffmpeg's CUDA frames use with NVENC.
const PITCH_ALIGNMENT: usize = 256;

/// The plane sizes of one config, and the surface that holds them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct SurfaceLayout {
    pub(super) format: PixelFormat,
    pub(super) width: u32,
    pub(super) height: u32,
    /// Bytes per surface row.
    pub(super) pitch: usize,
    /// `[rows, columns]` of the luma plane.
    pub(super) luma: [usize; 2],
    /// `[rows, columns]` of the chroma plane.
    pub(super) chroma: [usize; 2],
}

/// Bytes per row of a frame's two planes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PlanePitches {
    pub(super) luma: usize,
    pub(super) chroma: usize,
}

impl SurfaceLayout {
    pub(super) fn new(config: &EncodeConfig) -> Self {
        let luma = [config.height as usize, config.width as usize];
        let chroma = chroma_shape(config.width, config.height);
        let row_bytes = chroma[1].max(luma[1]) * config.format.bytes_per_sample();
        Self {
            format: config.format,
            width: config.width,
            height: config.height,
            pitch: row_bytes.next_multiple_of(PITCH_ALIGNMENT),
            luma,
            chroma,
        }
    }

    /// The surface size in bytes: Y rows, then UV rows, at one pitch.
    pub(super) const fn size_in_bytes(&self) -> usize {
        self.pitch * (self.luma[0] + self.chroma[0])
    }

    /// The pitch as NVENC's 32-bit field. A pitch past 32 bits saturates, and
    /// NVENC refuses the surface.
    pub(super) fn pitch_u32(&self) -> u32 {
        u32::try_from(self.pitch).unwrap_or(u32::MAX)
    }

    fn sample_dtype(&self) -> DType {
        self.format.sample_dtype()
    }

    /// Checks a frame against the config, and returns its row pitches.
    pub(super) fn check_frame(
        &self,
        frame: &EncodeFrame<'_, CudaRuntime>,
    ) -> Result<PlanePitches, EncodeError> {
        let (y, uv) = (frame.y(), frame.uv());
        Ok(PlanePitches {
            luma: self.check_plane("luma", y.shape(), y.strides(), y.dtype(), self.luma)?,
            chroma: self.check_plane(
                "chroma",
                uv.shape(),
                uv.strides(),
                uv.dtype(),
                self.chroma,
            )?,
        })
    }

    fn check_plane(
        &self,
        plane: &'static str,
        shape: &[usize],
        strides: &[isize],
        dtype: DType,
        expected: [usize; 2],
    ) -> Result<usize, EncodeError> {
        let wanted = self.sample_dtype();
        if shape != expected || dtype != wanted {
            return Err(EncodeError::SurfaceMismatch {
                plane,
                expected: format!("{expected:?} {wanted:?}"),
                found: format!("{shape:?} {dtype:?}"),
            });
        }
        let columns = expected[1];
        let row_stride = usize::try_from(strides[0]).ok();
        match row_stride {
            Some(row_stride) if strides[1] == 1 && row_stride >= columns => {
                Ok(row_stride * self.format.bytes_per_sample())
            }
            _ => Err(EncodeError::PlaneStrides {
                plane,
                strides: strides.to_vec(),
                columns,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::EncodeCodec;
    use crate::time::Rational;

    fn layout(width: u32, height: u32, format: PixelFormat) -> SurfaceLayout {
        SurfaceLayout::new(&EncodeConfig::new(
            EncodeCodec::Av1,
            width,
            height,
            format,
            Rational::ONE,
            Rational::ONE,
        ))
    }

    #[test]
    fn surface_rows_follow_each_other_at_an_aligned_pitch() {
        let nv12 = layout(640, 360, PixelFormat::Nv12);
        assert_eq!(nv12.pitch, 768);
        assert_eq!(nv12.chroma, [180, 640]);
        assert_eq!(nv12.size_in_bytes(), 768 * 540);
        let p010 = layout(640, 360, PixelFormat::P010);
        assert_eq!(p010.pitch, 1280);
        let odd = layout(321, 181, PixelFormat::Nv12);
        assert_eq!(odd.chroma, [91, 322]);
        assert_eq!(odd.pitch, 512);
    }

    #[test]
    fn a_dense_or_pitched_plane_gives_its_row_bytes() {
        let p010 = layout(640, 360, PixelFormat::P010);
        assert_eq!(
            p010.check_plane("luma", &[360, 640], &[640, 1], DType::U16, p010.luma),
            Ok(1280)
        );
        assert_eq!(
            p010.check_plane("luma", &[360, 640], &[704, 1], DType::U16, p010.luma),
            Ok(1408)
        );
    }

    #[test]
    fn a_wrong_shape_or_dtype_is_a_surface_mismatch() {
        let nv12 = layout(640, 360, PixelFormat::Nv12);
        for (shape, dtype) in [([360, 640], DType::U16), ([360, 642], DType::U8)] {
            let error = nv12.check_plane("luma", &shape, &[640, 1], dtype, nv12.luma);
            assert!(matches!(
                error,
                Err(EncodeError::SurfaceMismatch { plane: "luma", .. })
            ));
        }
    }

    #[test]
    fn strides_that_are_no_row_pitch_are_refused() {
        let nv12 = layout(640, 360, PixelFormat::Nv12);
        for strides in [[320, 1], [640, 2], [-640, 1]] {
            let error = nv12.check_plane("luma", &[360, 640], &strides, DType::U8, nv12.luma);
            assert!(matches!(
                error,
                Err(EncodeError::PlaneStrides { plane: "luma", .. })
            ));
        }
    }
}
