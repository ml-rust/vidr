//! Decoded picture buffers, and their view as host planes.

use std::fmt;

use super::error::HevcError;
use crate::decode::cpu::dsp::inter::interp::RefPlane;
use crate::decode::cpu::dsp::inter::weight::TargetPlane;
use crate::decode::cpu::dsp::sample::Sample;
use crate::frame::HostPlanes;

/// Row starts align to this many samples, so kernels can load whole vectors.
const STRIDE_ALIGN: usize = 16;

/// One sample plane with its row stride.
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct Plane<S> {
    samples: Vec<S>,
    width: usize,
    height: usize,
    stride: usize,
}

impl<S: Sample> Plane<S> {
    /// Allocates a plane of zero samples.
    ///
    /// # Errors
    ///
    /// [`HevcError::Allocation`] for a size that overflows or does not fit.
    pub(crate) fn new(width: usize, height: usize) -> Result<Self, HevcError> {
        let size = width
            .checked_next_multiple_of(STRIDE_ALIGN)
            .and_then(|stride| stride.checked_mul(height).map(|count| (stride, count)));
        let Some((stride, count)) = size else {
            return Err(HevcError::Allocation {
                what: "plane samples",
                count: usize::MAX,
            });
        };
        let mut samples = Vec::new();
        samples
            .try_reserve_exact(count)
            .map_err(|_| HevcError::Allocation {
                what: "plane samples",
                count,
            })?;
        samples.resize(count, S::default());
        Ok(Self {
            samples,
            width,
            height,
            stride,
        })
    }

    /// Samples per row.
    pub(crate) const fn width(&self) -> usize {
        self.width
    }

    /// Rows.
    pub(crate) const fn height(&self) -> usize {
        self.height
    }

    /// Samples from one row start to the next.
    pub(crate) const fn stride(&self) -> usize {
        self.stride
    }

    /// Row `y`, or `None` below the last row.
    pub(crate) fn row(&self, y: usize) -> Option<&[S]> {
        if y >= self.height {
            return None;
        }
        let start = y * self.stride;
        self.samples.get(start..start + self.width)
    }

    /// Row `y` for writing, or `None` below the last row.
    pub(crate) fn row_mut(&mut self, y: usize) -> Option<&mut [S]> {
        if y >= self.height {
            return None;
        }
        let start = y * self.stride;
        self.samples.get_mut(start..start + self.width)
    }

    /// Every sample, stride gaps included. Row `y` starts at `y * stride`.
    pub(crate) fn samples_mut(&mut self) -> &mut [S] {
        &mut self.samples
    }

    /// This plane as the inter kernels read it.
    pub(crate) fn view(&self) -> RefPlane<'_, S> {
        RefPlane {
            samples: &self.samples,
            width: self.width,
            height: self.height,
            stride: self.stride,
        }
    }

    /// This plane as the inter kernels write it.
    pub(crate) fn target(&mut self) -> TargetPlane<'_, S> {
        TargetPlane {
            samples: &mut self.samples,
            stride: self.stride,
        }
    }

    /// Copies the samples of a plane of the same size. A plane of another size
    /// leaves this one unchanged.
    pub(crate) fn copy_samples_from(&mut self, other: &Self) {
        if self.samples.len() == other.samples.len() {
            self.samples.copy_from_slice(&other.samples);
        }
    }
}

// A manual impl: a derive prints every sample.
impl<S> fmt::Debug for Plane<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Plane")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("stride", &self.stride)
            .finish()
    }
}

/// The output area of a picture in luma samples: the conformance window
/// (7.4.3.2.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DisplayWindow {
    /// Left edge.
    pub(crate) x: usize,
    /// Top edge.
    pub(crate) y: usize,
    /// Width.
    pub(crate) width: usize,
    /// Height.
    pub(crate) height: usize,
}

/// A 4:2:0 picture: luma and two chroma planes of half the size, rounded up.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Picture<S> {
    luma: Plane<S>,
    cb: Plane<S>,
    cr: Plane<S>,
    window: DisplayWindow,
}

/// The host bytes of 10-bit planes. A picture of `u8` samples leaves them empty.
#[derive(Clone, Debug, Default)]
pub(crate) struct HostScratch {
    luma: Vec<u8>,
    cb: Vec<u8>,
    cr: Vec<u8>,
}

impl<S: Sample> Picture<S> {
    /// Allocates a picture that outputs the whole coded area.
    ///
    /// # Errors
    ///
    /// [`HevcError::Allocation`] for a size that overflows or does not fit.
    pub(crate) fn new(width: usize, height: usize) -> Result<Self, HevcError> {
        let (chroma_width, chroma_height) = (width.div_ceil(2), height.div_ceil(2));
        Ok(Self {
            luma: Plane::new(width, height)?,
            cb: Plane::new(chroma_width, chroma_height)?,
            cr: Plane::new(chroma_width, chroma_height)?,
            window: DisplayWindow {
                x: 0,
                y: 0,
                width,
                height,
            },
        })
    }

    /// Sets the output area. The left and top edges fall on even samples, because
    /// the conformance window counts in units of `SubWidthC` and `SubHeightC`.
    ///
    /// # Errors
    ///
    /// [`HevcError::DisplayWindow`] for an empty area, an odd left or top edge,
    /// or an area outside the picture.
    pub(crate) fn set_window(&mut self, window: DisplayWindow) -> Result<(), HevcError> {
        let fits = |start: usize, size: usize, limit: usize| {
            size > 0
                && start.is_multiple_of(2)
                && start.checked_add(size).is_some_and(|end| end <= limit)
        };
        if !fits(window.x, window.width, self.luma.width())
            || !fits(window.y, window.height, self.luma.height())
        {
            return Err(HevcError::DisplayWindow {
                x: window.x,
                y: window.y,
                width: window.width,
                height: window.height,
            });
        }
        self.window = window;
        Ok(())
    }

    /// The luma plane.
    pub(crate) const fn luma(&self) -> &Plane<S> {
        &self.luma
    }

    /// The Cb plane.
    #[cfg(test)]
    pub(crate) const fn cb(&self) -> &Plane<S> {
        &self.cb
    }

    /// The Cr plane.
    #[cfg(test)]
    pub(crate) const fn cr(&self) -> &Plane<S> {
        &self.cr
    }

    /// The plane of `cIdx` `c_idx`. An index above 2 selects Cr.
    pub(crate) const fn plane(&self, c_idx: usize) -> &Plane<S> {
        match c_idx {
            0 => &self.luma,
            1 => &self.cb,
            _ => &self.cr,
        }
    }

    /// The plane of `cIdx` `c_idx` for writing. An index above 2 selects Cr.
    pub(crate) const fn plane_mut(&mut self, c_idx: usize) -> &mut Plane<S> {
        match c_idx {
            0 => &mut self.luma,
            1 => &mut self.cb,
            _ => &mut self.cr,
        }
    }

    /// The output area.
    pub(crate) const fn window(&self) -> DisplayWindow {
        self.window
    }

    /// The luma plane for writing.
    pub(crate) const fn luma_mut(&mut self) -> &mut Plane<S> {
        &mut self.luma
    }

    /// The Cb plane for writing.
    pub(crate) const fn cb_mut(&mut self) -> &mut Plane<S> {
        &mut self.cb
    }

    /// The Cr plane for writing.
    pub(crate) const fn cr_mut(&mut self) -> &mut Plane<S> {
        &mut self.cr
    }

    /// The output area as I420 planes for `u8` samples, or I010 for `u16`.
    ///
    /// `u8` planes borrow this picture with no copy. `u16` planes copy once into
    /// `scratch`, because a safe cast from `u16` to bytes does not exist.
    ///
    /// # Errors
    ///
    /// [`HevcError::Allocation`] when `scratch` does not fit in memory.
    pub(crate) fn host_planes<'a>(
        &'a self,
        scratch: &'a mut HostScratch,
    ) -> Result<HostPlanes<'a>, HevcError> {
        let window = self.window;
        let (chroma_x, chroma_y) = (window.x / 2, window.y / 2);
        let y = plane_bytes(&self.luma, window.x, window.y, &mut scratch.luma)?;
        let u = plane_bytes(&self.cb, chroma_x, chroma_y, &mut scratch.cb)?;
        let v = plane_bytes(&self.cr, chroma_x, chroma_y, &mut scratch.cr)?;
        Ok(HostPlanes {
            y,
            y_stride: self.luma.stride * S::BYTES,
            u,
            u_stride: self.cb.stride * S::BYTES,
            v,
            v_stride: self.cr.stride * S::BYTES,
            width: window.width,
            height: window.height,
        })
    }
}

/// The bytes of `plane` from sample `( x, y )` to its end.
fn plane_bytes<'a, S: Sample>(
    plane: &'a Plane<S>,
    x: usize,
    y: usize,
    scratch: &'a mut Vec<u8>,
) -> Result<&'a [u8], HevcError> {
    let bytes = S::host_bytes(&plane.samples, scratch).map_err(|_| HevcError::Allocation {
        what: "host plane bytes",
        count: plane.samples.len() * S::BYTES,
    })?;
    let offset = (y * plane.stride + x) * S::BYTES;
    Ok(bytes.get(offset..).unwrap_or_default())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A value that differs per plane, row, and column.
    fn value(plane: usize, x: usize, y: usize) -> u16 {
        (plane * 100 + y * 10 + x) as u16
    }

    fn fill<S: Sample>(picture: &mut Picture<S>, make: fn(u16) -> S) {
        fill_plane(picture.luma_mut(), 0, make);
        fill_plane(picture.cb_mut(), 1, make);
        fill_plane(picture.cr_mut(), 2, make);
    }

    fn fill_plane<S: Sample>(plane: &mut Plane<S>, index: usize, make: fn(u16) -> S) {
        for y in 0..plane.height() {
            let row = plane.row_mut(y).unwrap();
            for (x, sample) in row.iter_mut().enumerate() {
                *sample = make(value(index, x, y));
            }
        }
    }

    #[test]
    fn odd_sizes_round_chroma_up() {
        let picture = Picture::<u8>::new(321, 181).unwrap();
        assert_eq!(picture.luma().width(), 321);
        assert_eq!(picture.luma().height(), 181);
        assert_eq!(picture.cb().width(), 161);
        assert_eq!(picture.cb().height(), 91);
        assert_eq!(picture.cr().width(), 161);
        assert_eq!(picture.cr().height(), 91);
    }

    #[test]
    fn strides_align_and_rows_keep_their_width() {
        let picture = Picture::<u16>::new(17, 3).unwrap();
        assert_eq!(picture.luma().stride(), 32);
        assert_eq!(picture.cb().stride(), 16);
        assert_eq!(picture.luma().row(2).map(<[u16]>::len), Some(17));
        assert_eq!(picture.luma().row(3), None);
    }

    #[test]
    fn eight_bit_host_planes_read_the_picture() {
        let mut picture = Picture::<u8>::new(3, 3).unwrap();
        fill(&mut picture, |sample| sample as u8);
        let mut scratch = HostScratch::default();
        let planes = picture.host_planes(&mut scratch).unwrap();
        assert_eq!((planes.width, planes.height), (3, 3));
        assert_eq!(
            (planes.y_stride, planes.u_stride, planes.v_stride),
            (16, 16, 16)
        );
        for y in 0..3 {
            for x in 0..3 {
                assert_eq!(planes.y[y * 16 + x], value(0, x, y) as u8);
            }
        }
        for y in 0..2 {
            for x in 0..2 {
                assert_eq!(planes.u[y * 16 + x], value(1, x, y) as u8);
                assert_eq!(planes.v[y * 16 + x], value(2, x, y) as u8);
            }
        }
    }

    #[test]
    fn ten_bit_host_planes_hold_little_endian_samples() {
        let mut picture = Picture::<u16>::new(5, 3).unwrap();
        fill(&mut picture, |sample| sample + 700);
        let mut scratch = HostScratch::default();
        let planes = picture.host_planes(&mut scratch).unwrap();
        assert_eq!((planes.y_stride, planes.u_stride), (32, 32));
        let read = |bytes: &[u8], stride: usize, x: usize, y: usize| {
            let at = y * stride + 2 * x;
            u16::from_le_bytes([bytes[at], bytes[at + 1]])
        };
        assert_eq!(read(planes.y, 32, 4, 2), value(0, 4, 2) + 700);
        assert_eq!(read(planes.u, 32, 2, 1), value(1, 2, 1) + 700);
        assert_eq!(read(planes.v, 32, 1, 1), value(2, 1, 1) + 700);
    }

    #[test]
    fn a_window_moves_every_plane() {
        let mut picture = Picture::<u8>::new(8, 8).unwrap();
        fill(&mut picture, |sample| sample as u8);
        picture
            .set_window(DisplayWindow {
                x: 2,
                y: 4,
                width: 6,
                height: 2,
            })
            .unwrap();
        let mut scratch = HostScratch::default();
        let planes = picture.host_planes(&mut scratch).unwrap();
        assert_eq!((planes.width, planes.height), (6, 2));
        assert_eq!(planes.y[0], value(0, 2, 4) as u8);
        assert_eq!(planes.u[0], value(1, 1, 2) as u8);
        assert_eq!(planes.v[16 + 1], value(2, 2, 3) as u8);
    }

    #[test]
    fn bad_windows_are_refused() {
        let mut picture = Picture::<u8>::new(8, 8).unwrap();
        for (x, y, width, height) in [(1, 0, 2, 2), (0, 3, 2, 2), (0, 0, 9, 2), (0, 0, 0, 2)] {
            let window = DisplayWindow {
                x,
                y,
                width,
                height,
            };
            assert_eq!(
                picture.set_window(window),
                Err(HevcError::DisplayWindow {
                    x,
                    y,
                    width,
                    height
                })
            );
        }
    }

    #[test]
    fn an_overflowing_size_is_refused_before_allocation() {
        assert!(matches!(
            Plane::<u16>::new(usize::MAX, 2),
            Err(HevcError::Allocation { .. })
        ));
        assert!(matches!(
            Plane::<u8>::new(1 << 40, 1 << 40),
            Err(HevcError::Allocation { .. })
        ));
    }
}
