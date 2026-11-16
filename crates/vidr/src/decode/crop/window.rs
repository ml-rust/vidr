//! Cutting a frame to a crop: the plane windows and the sample-plane copy.

use numr::runtime::Runtime;
use numr::tensor::Tensor;

use super::packed::word_window;
use super::rect::CropRect;
use crate::decode::frame::{DecodedFrame, PlaneForm};
use crate::error::DecodeError;
use crate::geometry::Dimensions;

/// `frame` cut to `crop`, in dense planes on its device.
///
/// # Errors
///
/// [`DecodeError::CropOutsideFrame`] when the crop leaves no picture, and
/// [`DecodeError::CropCopy`] when a plane does not hold the documented shape
/// or its copy fails.
pub(super) fn crop_frame<R: Runtime>(
    frame: DecodedFrame<R>,
    crop: CropRect,
) -> Result<DecodedFrame<R>, DecodeError> {
    let display = frame.display();
    let Some(size) = crop.fit(display) else {
        return Err(DecodeError::CropOutsideFrame {
            crop,
            width: display.width(),
            height: display.height(),
            pts: frame.pts(),
        });
    };
    let (luma, chroma) = windows(crop, size);
    let (y, uv) = match frame.plane_form() {
        PlaneForm::Samples => (
            sample_window(frame.y(), luma, "luma")?,
            sample_window(frame.uv(), chroma, "chroma")?,
        ),
        PlaneForm::Words => {
            let sample = frame.format().bytes_per_sample();
            (
                word_window(frame.y(), luma.in_bytes(sample), "luma")?,
                word_window(frame.uv(), chroma.in_bytes(sample), "chroma")?,
            )
        }
    };
    Ok(frame.with_planes(y, uv, size))
}

/// The rows and columns of a plane a crop keeps. Columns count plane
/// elements: samples, values, or bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Window {
    pub(super) row: usize,
    pub(super) rows: usize,
    pub(super) column: usize,
    pub(super) columns: usize,
}

impl Window {
    /// The window with each column widened to `bytes` bytes.
    pub(super) const fn in_bytes(self, bytes: usize) -> Self {
        Self {
            column: self.column * bytes,
            columns: self.columns * bytes,
            ..self
        }
    }

    pub(super) fn describe(self, plane: &[usize]) -> String {
        format!(
            "rows {}..{} and columns {}..{} of a {plane:?} plane",
            self.row,
            self.row + self.rows,
            self.column,
            self.column + self.columns
        )
    }
}

/// The luma and chroma windows of a crop that keeps `size`, in samples and
/// interleaved chroma values.
///
/// The chroma window always fits a 4:2:0 chroma plane of `ceil(H / 2)` rows
/// and `2 * ceil(W / 2)` values. With `top + rows <= H` the window ends at
/// `floor(top / 2) + ceil(rows / 2)`, which is at most `ceil((top + rows) / 2)`:
/// equal for an even `top`, and `ceil((top - 1 + rows) / 2)` for an odd one.
/// That is at most `ceil(H / 2)`. Columns follow the same steps, doubled.
pub(super) fn windows(crop: CropRect, size: Dimensions) -> (Window, Window) {
    let (top, left) = (crop.top as usize, crop.left as usize);
    let (width, height) = (size.width() as usize, size.height() as usize);
    let luma = Window {
        row: top,
        rows: height,
        column: left,
        columns: width,
    };
    let chroma = Window {
        row: top / 2,
        rows: height.div_ceil(2),
        column: 2 * (left / 2),
        columns: 2 * width.div_ceil(2),
    };
    (luma, chroma)
}

pub(super) fn copy_error<R: Runtime>(
    plane: &'static str,
    tensor: &Tensor<R>,
    window: Window,
    reason: String,
) -> DecodeError {
    DecodeError::CropCopy {
        plane,
        window: window.describe(tensor.shape()),
        reason,
    }
}

/// A sample plane cut to `window`: a narrowed view, then one copy.
fn sample_window<R: Runtime>(
    plane: &Tensor<R>,
    window: Window,
    name: &'static str,
) -> Result<Tensor<R>, DecodeError> {
    plane
        .narrow(0, window.row, window.rows)
        .and_then(|view| view.narrow(1, window.column, window.columns))
        .and_then(|view| view.compact())
        .map_err(|error| copy_error(name, plane, window, error.to_string()))
}

#[cfg(test)]
mod tests {
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    use super::*;
    use crate::decode::PixelFormat;
    use crate::decode::crop::packed::byte_span;
    use crate::decode::crop::test_frames::{
        HEIGHT, UNSTATED, WIDTH, frame, plane_bytes, rect, slice, unpack,
    };

    /// Crops a frame of every format and form, and checks each plane against
    /// the source sliced by hand.
    fn check(crop: CropRect) {
        let size = crop.fit(Dimensions::new(WIDTH, HEIGHT).unwrap()).unwrap();
        let (luma, chroma) = windows(crop, size);
        for format in [PixelFormat::Nv12, PixelFormat::P010] {
            let sample = format.bytes_per_sample();
            for form in [PlaneForm::Samples, PlaneForm::Words] {
                let (input, source) = frame(format, form);
                let cropped = crop_frame(input, crop).unwrap();
                let case = format!("{crop} {format:?} {form:?}");
                assert_eq!(cropped.display(), size, "{case}");
                assert_eq!(cropped.plane_form(), form, "{case}");
                assert_eq!(cropped.pts(), 7, "{case}");
                let want_y = slice(&source.luma, WIDTH as usize, luma, sample);
                let want_uv = slice(&source.chroma, source.chroma_shape[1], chroma, sample);
                let shape = crate::decode::frame::chroma_shape(size.width(), size.height());
                assert_eq!(chroma.rows, shape[0], "{case}");
                assert_eq!(chroma.columns, shape[1], "{case}");
                for plane in [cropped.y(), cropped.uv()] {
                    assert!(plane.is_contiguous(), "{case}");
                    assert_eq!(plane.offset(), 0, "{case}");
                    assert_eq!(plane.storage().len(), plane.numel(), "{case}");
                }
                match form {
                    PlaneForm::Samples => {
                        assert_eq!(
                            cropped.y().shape(),
                            &[size.height() as usize, size.width() as usize],
                            "{case}"
                        );
                        assert_eq!(cropped.uv().shape(), &shape, "{case}");
                        assert_eq!(plane_bytes(cropped.y(), format), want_y, "{case}");
                        assert_eq!(plane_bytes(cropped.uv(), format), want_uv, "{case}");
                    }
                    PlaneForm::Words => {
                        let y_row = size.width() as usize * sample;
                        let uv_row = shape[1] * sample;
                        assert_eq!(
                            cropped.y().shape(),
                            &[size.height() as usize, y_row.div_ceil(4)],
                            "{case}"
                        );
                        assert_eq!(
                            cropped.uv().shape(),
                            &[shape[0], uv_row.div_ceil(4)],
                            "{case}"
                        );
                        assert_eq!(unpack(cropped.y(), y_row), want_y, "{case}");
                        assert_eq!(unpack(cropped.uv(), uv_row), want_uv, "{case}");
                    }
                }
            }
        }
    }

    #[test]
    fn an_even_crop_keeps_the_window() {
        check(rect(2, 2, 4, 2));
    }

    #[test]
    fn an_odd_left_shifts_chroma_by_the_rounded_down_half() {
        check(rect(0, 0, 3, 0));
        check(rect(0, 0, 1, 2));
    }

    #[test]
    fn an_odd_top_shifts_chroma_by_the_rounded_down_half() {
        check(rect(1, 0, 0, 0));
        check(rect(3, 1, 0, 0));
    }

    #[test]
    fn an_odd_width_and_height_round_chroma_up() {
        check(rect(0, 1, 0, 1));
        check(rect(1, 1, 1, 1));
        check(rect(2, 0, 5, 1));
    }

    #[test]
    fn a_crop_that_removes_nothing_keeps_the_frame() {
        check(CropRect::default());
    }

    #[test]
    fn a_crop_to_one_pixel_keeps_one_sample_and_one_chroma_pair() {
        check(rect(6, 0, 10, 0));
        check(rect(3, 3, 5, 5));
    }

    #[test]
    fn a_crop_at_the_end_of_packed_storage_takes_the_staged_copy() {
        // Luma rows of 11 bytes sit in 3 words. A left crop of 1 at the
        // bottom right reads past the last word with one wide copy.
        let (input, _) = frame(PixelFormat::Nv12, PlaneForm::Words);
        let window = windows(rect(0, 0, 1, 0), Dimensions::new(10, 7).unwrap()).0;
        let span = byte_span(input.y(), window).unwrap();
        assert!(span.wide_end(12) > input.y().storage().size_in_bytes());
        check(rect(0, 0, 1, 0));
    }

    #[test]
    fn a_crop_that_leaves_no_picture_is_refused_by_name() {
        for crop in [rect(0, 0, 11, 0), rect(4, 3, 0, 0), rect(0, 0, 0, u32::MAX)] {
            let (input, _) = frame(PixelFormat::Nv12, PlaneForm::Samples);
            assert_eq!(
                crop_frame(input, crop).unwrap_err(),
                DecodeError::CropOutsideFrame {
                    crop,
                    width: WIDTH,
                    height: HEIGHT,
                    pts: 7,
                }
            );
        }
    }

    #[test]
    fn a_plane_smaller_than_its_display_is_refused() {
        let device = CpuDevice::new();
        let display = Dimensions::new(WIDTH, HEIGHT).unwrap();
        let frame = DecodedFrame::<CpuRuntime>::from_packed_planes(
            Tensor::from_slice(&[0_u32; 4], &[2, 2], &device).unwrap(),
            Tensor::from_slice(&[0_u32; 4], &[2, 2], &device).unwrap(),
            PixelFormat::Nv12,
            0,
            display,
            UNSTATED,
        );
        let error = crop_frame(frame, rect(1, 0, 1, 0)).unwrap_err();
        assert!(
            matches!(error, DecodeError::CropCopy { plane: "luma", .. }),
            "{error}"
        );
    }
}
