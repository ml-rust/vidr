//! Host plane copies on the CPU backend, and the checks the WebGPU tests share.

use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;
use vidr::decode::{PixelFormat, PlaneForm};
use vidr::encode::EncodeFrame;
use vidr::frame::{HostFrame, HostPlanes, download_planes, upload_planes};
use vidr::{Dimensions, FrameError};

use super::support::FORMATS;

/// Even sizes, each odd shape, and frames a single pixel wide or high.
pub(crate) const SIZES: [(usize, usize); 9] = [
    (6, 4),
    (64, 36),
    (322, 180),
    (321, 181),
    (321, 180),
    (320, 181),
    (1, 1),
    (3, 1),
    (1, 3),
];

/// Bytes a pitched source holds between its rows. A read of one shows in a plane.
const GAP_BYTE: u8 = 0xA5;

/// One tight host plane of pseudo-random samples, every bit of the value in use.
fn tight_plane(columns: usize, rows: usize, format: PixelFormat, seed: u32) -> Vec<u8> {
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    let mut next = move || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        state >> 12
    };
    let count = columns * rows;
    match format {
        PixelFormat::Nv12 => (0..count).map(|_| next() as u8).collect(),
        _ => (0..count)
            .flat_map(|_| ((next() & 0x3FF) as u16).to_le_bytes())
            .collect(),
    }
}

/// `tight` with `gap` bytes of [`GAP_BYTE`] after each row but the last.
fn pitched(tight: &[u8], row_bytes: usize, gap: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(tight.len() + gap * tight.len() / row_bytes);
    let rows = tight.len() / row_bytes;
    for (index, row) in tight.chunks_exact(row_bytes).enumerate() {
        out.extend_from_slice(row);
        if index + 1 < rows {
            out.extend(std::iter::repeat_n(GAP_BYTE, gap));
        }
    }
    out
}

/// A tight host frame of `width x height`.
pub(crate) struct Source {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) format: PixelFormat,
    pub(crate) y: Vec<u8>,
    pub(crate) u: Vec<u8>,
    pub(crate) v: Vec<u8>,
}

impl Source {
    pub(crate) fn new((width, height): (usize, usize), format: PixelFormat, seed: u32) -> Self {
        let (columns, rows) = (width.div_ceil(2), height.div_ceil(2));
        Self {
            width,
            height,
            format,
            y: tight_plane(width, height, format, seed),
            u: tight_plane(columns, rows, format, seed + 1),
            v: tight_plane(columns, rows, format, seed + 2),
        }
    }

    fn sample(&self) -> usize {
        self.format.bytes_per_sample()
    }

    fn luma_row(&self) -> usize {
        self.width * self.sample()
    }

    fn chroma_row(&self) -> usize {
        self.width.div_ceil(2) * self.sample()
    }

    /// The planes with their tight strides.
    fn planes(&self) -> HostPlanes<'_> {
        HostPlanes {
            y: &self.y,
            y_stride: self.luma_row(),
            u: &self.u,
            u_stride: self.chroma_row(),
            v: &self.v,
            v_stride: self.chroma_row(),
            width: self.width,
            height: self.height,
        }
    }

    /// The bytes of the NV12 or P010 luma plane: `value << 6` for P010.
    pub(crate) fn expected_y(&self) -> Vec<u8> {
        self.stored(&self.y)
    }

    /// The bytes of the interleaved chroma plane.
    pub(crate) fn expected_uv(&self) -> Vec<u8> {
        let (u, v) = (self.stored(&self.u), self.stored(&self.v));
        let sample = self.sample();
        u.chunks_exact(sample)
            .zip(v.chunks_exact(sample))
            .flat_map(|(cb, cr)| cb.iter().chain(cr).copied())
            .collect()
    }

    fn stored(&self, plane: &[u8]) -> Vec<u8> {
        match self.format {
            PixelFormat::Nv12 => plane.to_vec(),
            _ => plane
                .as_chunks::<2>()
                .0
                .iter()
                .flat_map(|sample| (u16::from_le_bytes([sample[0], sample[1]]) << 6).to_le_bytes())
                .collect(),
        }
    }

    fn display(&self) -> Dimensions {
        Dimensions::new(self.width as u32, self.height as u32)
            .unwrap_or_else(|error| panic!("display size: {error}"))
    }
}

/// The frame an encoder takes from uploaded planes, in the backend's form.
fn encode_frame<'a, R: Runtime<DType = DType>>(
    y: &'a Tensor<R>,
    uv: &'a Tensor<R>,
    source: &Source,
) -> EncodeFrame<'a, R> {
    if R::packed_color_planes() {
        let frame = EncodeFrame::from_packed_planes(y, uv, source.format, source.display());
        assert_eq!(frame.plane_form(), PlaneForm::Words);
        frame
    } else {
        let frame = EncodeFrame::new(y, uv);
        assert_eq!(frame.plane_form(), PlaneForm::Samples);
        frame
    }
}

fn assert_same(out: &HostFrame, source: &Source, label: &str) {
    assert_eq!(out.width(), source.width, "{label}: width");
    assert_eq!(out.height(), source.height, "{label}: height");
    assert_eq!(out.format(), source.format, "{label}: format");
    assert!(out.y() == source.y.as_slice(), "{label}: y");
    assert!(out.u() == source.u.as_slice(), "{label}: u");
    assert!(out.v() == source.v.as_slice(), "{label}: v");
}

/// Upload then download gives the source bytes, at every size and format, from
/// tight and pitched sources. The stored planes hold NV12 or P010 as documented.
pub(crate) fn check_round_trip<R: Runtime<DType = DType>>(device: &R::Device) {
    for size in SIZES {
        for format in FORMATS {
            let source = Source::new(size, format, 11);
            let (luma_row, chroma_row) = (source.luma_row(), source.chroma_row());
            let label = format!("{size:?} {format} on {}", R::name());
            let mut out = HostFrame::new();

            let (y, uv) = upload_planes::<R>(device, source.planes(), format)
                .unwrap_or_else(|error| panic!("{label}: tight upload: {error}"));
            check_stored(&y, &uv, &source, &label);
            download_planes(&encode_frame(&y, &uv, &source), &mut out)
                .unwrap_or_else(|error| panic!("{label}: download: {error}"));
            assert_same(&out, &source, &label);

            // The frame's own planes pass straight back in.
            let (y, uv) = upload_planes::<R>(device, out.planes(), format)
                .unwrap_or_else(|error| panic!("{label}: reupload: {error}"));
            check_stored(&y, &uv, &source, &label);

            let gap = 7;
            let (pitched_y, pitched_u, pitched_v) = (
                pitched(&source.y, luma_row, gap),
                pitched(&source.u, chroma_row, gap + 2),
                pitched(&source.v, chroma_row, gap + 4),
            );
            let planes = HostPlanes {
                y: &pitched_y,
                y_stride: luma_row + gap,
                u: &pitched_u,
                u_stride: chroma_row + gap + 2,
                v: &pitched_v,
                v_stride: chroma_row + gap + 4,
                width: source.width,
                height: source.height,
            };
            let (y, uv) = upload_planes::<R>(device, planes, format)
                .unwrap_or_else(|error| panic!("{label}: pitched upload: {error}"));
            let mut out = HostFrame::new();
            download_planes(&encode_frame(&y, &uv, &source), &mut out)
                .unwrap_or_else(|error| panic!("{label}: pitched download: {error}"));
            assert_same(&out, &source, &format!("{label}, pitched"));
        }
    }
}

/// The tensors hold the documented NV12 or P010 bytes in the backend's form.
fn check_stored<R: Runtime<DType = DType>>(
    y: &Tensor<R>,
    uv: &Tensor<R>,
    source: &Source,
    label: &str,
) {
    let (luma_row, chroma_bytes) = (source.luma_row(), 2 * source.chroma_row());
    let read = |plane: &Tensor<R>, row: usize| {
        let bytes = plane
            .to_bytes()
            .unwrap_or_else(|error| panic!("{label}: read: {error}"));
        if plane.dtype() != DType::U32 {
            return bytes;
        }
        let pitch = plane.shape()[1] * 4;
        assert_eq!(pitch, row.div_ceil(4) * 4, "{label}: words per row");
        let mut data = Vec::with_capacity(row * plane.shape()[0]);
        for stored in bytes.chunks_exact(pitch) {
            assert!(
                stored[row..].iter().all(|&byte| byte == 0),
                "{label}: a byte past the row is not 0"
            );
            data.extend_from_slice(&stored[..row]);
        }
        data
    };
    assert!(read(y, luma_row) == source.expected_y(), "{label}: y plane");
    assert!(
        read(uv, chroma_bytes) == source.expected_uv(),
        "{label}: uv plane"
    );
}

/// A view into a wider, taller surface downloads without the bytes around it.
/// `first_row` above 0 gives the view a nonzero offset. The surface is
/// `rows + first_row` rows of `columns + 3` elements, and every other element is
/// a sentinel.
fn surface_view<R: Runtime<DType = DType>>(
    tight: &[u8],
    rows: usize,
    columns: usize,
    dtype: DType,
    first_row: usize,
    device: &R::Device,
) -> Tensor<R> {
    let element = dtype.size_in_bytes();
    let row_bytes = columns * element;
    let pitch = (columns + 3) * element;
    let mut surface = vec![GAP_BYTE; (rows + first_row) * pitch];
    for (index, row) in tight.chunks_exact(row_bytes).enumerate() {
        let start = (first_row + index) * pitch;
        surface[start..start + row_bytes].copy_from_slice(row);
    }
    let shape = [rows + first_row, columns + 3];
    Tensor::<R>::from_bytes(&surface, &shape, dtype, device)
        .and_then(|plane| plane.narrow(0, first_row, rows))
        .and_then(|plane| plane.narrow(1, 0, columns))
        .unwrap_or_else(|error| panic!("surface view: {error}"))
}

/// Pitched and offset plane views in the backend's form download byte-exact.
pub(crate) fn check_pitched_download<R: Runtime<DType = DType>>(device: &R::Device) {
    for size in [(6, 4), (321, 181), (64, 36)] {
        for format in FORMATS {
            for first_row in [0, 2] {
                let source = Source::new(size, format, 23);
                let label = format!("{size:?} {format} from row {first_row} on {}", R::name());
                let view = |tight: &[u8], rows: usize, row_bytes: usize, columns: usize| {
                    if R::packed_color_planes() {
                        let packed = words(tight, row_bytes);
                        let words_per_row = row_bytes.div_ceil(4);
                        surface_view::<R>(
                            &packed,
                            rows,
                            words_per_row,
                            DType::U32,
                            first_row,
                            device,
                        )
                    } else {
                        let dtype = format.sample_dtype();
                        surface_view::<R>(tight, rows, columns, dtype, first_row, device)
                    }
                };
                let y = view(
                    &source.expected_y(),
                    source.height,
                    source.luma_row(),
                    source.width,
                );
                let uv = view(
                    &source.expected_uv(),
                    source.height.div_ceil(2),
                    2 * source.chroma_row(),
                    source.width.div_ceil(2) * 2,
                );
                let mut out = HostFrame::new();
                download_planes(&encode_frame(&y, &uv, &source), &mut out)
                    .unwrap_or_else(|error| panic!("{label}: {error}"));
                assert_same(&out, &source, &label);
            }
        }
    }
}

/// Tight rows padded with 0 to whole words.
fn words(tight: &[u8], row_bytes: usize) -> Vec<u8> {
    let pitch = row_bytes.div_ceil(4) * 4;
    let mut out = Vec::with_capacity(tight.len() / row_bytes * pitch);
    for row in tight.chunks_exact(row_bytes) {
        out.extend_from_slice(row);
        out.resize(out.len() + pitch - row_bytes, 0);
    }
    out
}

/// A second frame of the same size lands in the same buffers with new bytes, so
/// it allocates nothing on the host.
pub(crate) fn check_buffer_reuse<R: Runtime<DType = DType>>(device: &R::Device) {
    for format in FORMATS {
        let (first, second) = (
            Source::new((321, 181), format, 31),
            Source::new((321, 181), format, 41),
        );
        let label = format!("{format} on {}", R::name());
        let mut out = HostFrame::new();
        let (y, uv) = upload_planes::<R>(device, first.planes(), format)
            .unwrap_or_else(|error| panic!("{label}: {error}"));
        download_planes(&encode_frame(&y, &uv, &first), &mut out)
            .unwrap_or_else(|error| panic!("{label}: {error}"));
        assert_same(&out, &first, &label);
        let buffers = (out.y().as_ptr(), out.u().as_ptr(), out.v().as_ptr());

        let (y, uv) = upload_planes::<R>(device, second.planes(), format)
            .unwrap_or_else(|error| panic!("{label}: {error}"));
        download_planes(&encode_frame(&y, &uv, &second), &mut out)
            .unwrap_or_else(|error| panic!("{label}: {error}"));
        assert_same(&out, &second, &label);
        assert_eq!(
            buffers,
            (out.y().as_ptr(), out.u().as_ptr(), out.v().as_ptr()),
            "{label}: a buffer moved"
        );
    }
}

/// A smaller frame after a larger one keeps the larger buffers and reports its
/// own size.
pub(crate) fn check_size_change<R: Runtime<DType = DType>>(device: &R::Device) {
    let mut out = HostFrame::new();
    for size in [(64, 36), (7, 5), (6, 4), (64, 36)] {
        let source = Source::new(size, PixelFormat::Nv12, 53);
        let (y, uv) = upload_planes::<R>(device, source.planes(), PixelFormat::Nv12)
            .unwrap_or_else(|error| panic!("{size:?}: {error}"));
        download_planes(&encode_frame(&y, &uv, &source), &mut out)
            .unwrap_or_else(|error| panic!("{size:?}: {error}"));
        assert_same(&out, &source, &format!("{size:?}"));
    }
}

fn device() -> CpuDevice {
    CpuDevice::new()
}

#[test]
fn round_trips_are_byte_exact_on_cpu() {
    check_round_trip::<CpuRuntime>(&device());
}

#[test]
fn pitched_and_offset_views_download_byte_exact_on_cpu() {
    check_pitched_download::<CpuRuntime>(&device());
}

#[test]
fn a_second_frame_reuses_the_buffers_on_cpu() {
    check_buffer_reuse::<CpuRuntime>(&device());
}

#[test]
fn a_size_change_downloads_on_cpu() {
    check_size_change::<CpuRuntime>(&device());
}

fn refused(planes: HostPlanes<'_>, format: PixelFormat) -> FrameError {
    upload_planes::<CpuRuntime>(&device(), planes, format).expect_err("the upload passes")
}

#[test]
fn upload_refuses_bad_planes_by_name() {
    let source = Source::new((6, 4), PixelFormat::Nv12, 3);
    let planes = source.planes();

    let zero = HostPlanes { width: 0, ..planes };
    assert_eq!(
        refused(zero, PixelFormat::Nv12),
        FrameError::HostFrameSize {
            width: 0,
            height: 4
        }
    );

    let narrow = HostPlanes {
        y_stride: 5,
        ..planes
    };
    let error = refused(narrow, PixelFormat::Nv12);
    assert_eq!(
        error,
        FrameError::HostStride {
            plane: "y",
            stride: 5,
            row_bytes: 6
        }
    );
    assert!(error.to_string().contains("stride of 5"), "{error}");

    let short = HostPlanes {
        u: &source.u[..5],
        ..planes
    };
    assert_eq!(
        refused(short, PixelFormat::Nv12),
        FrameError::HostPlaneShort {
            plane: "u",
            needed: 6,
            found: 5
        }
    );

    let wide = HostPlanes {
        v_stride: 100,
        ..planes
    };
    assert_eq!(
        refused(wide, PixelFormat::Nv12),
        FrameError::HostPlaneShort {
            plane: "v",
            needed: 103,
            found: 6
        }
    );
}

/// `value << 6` cuts a value above 1023, so the upload refuses it and names it.
#[test]
fn upload_refuses_a_value_above_ten_bits() {
    let mut source = Source::new((6, 4), PixelFormat::P010, 3);
    let at = source.chroma_row() + 2;
    source.v[at..at + 2].copy_from_slice(&1024_u16.to_le_bytes());
    assert_eq!(
        refused(source.planes(), PixelFormat::P010),
        FrameError::HostSampleRange {
            plane: "v",
            row: 1,
            column: 1,
            value: 1024
        }
    );
}

/// A sample-form frame states its format through its dtype. Another dtype and
/// another rank are named refusals.
#[test]
fn download_refuses_planes_it_cannot_read() {
    let dev = device();
    let mut out = HostFrame::new();
    let blank = |shape: &[usize], dtype| {
        Tensor::<CpuRuntime>::zeros(shape, dtype, &dev)
            .unwrap_or_else(|error| panic!("{shape:?}: {error}"))
    };

    let (y, uv) = (blank(&[4, 6], DType::F32), blank(&[2, 6], DType::F32));
    let error =
        download_planes(&EncodeFrame::new(&y, &uv), &mut out).expect_err("F32 planes download");
    assert!(
        matches!(error, FrameError::PlaneMismatch { plane: "luma", .. }),
        "{error}"
    );

    let (y, uv) = (blank(&[4, 6, 1], DType::U8), blank(&[2, 6], DType::U8));
    let error = download_planes(&EncodeFrame::new(&y, &uv), &mut out)
        .expect_err("a rank-3 plane downloads");
    assert!(
        matches!(error, FrameError::PlaneMismatch { plane: "luma", .. }),
        "{error}"
    );

    let (y, uv) = (blank(&[4, 6], DType::U8), blank(&[3, 6], DType::U8));
    let error = download_planes(&EncodeFrame::new(&y, &uv), &mut out)
        .expect_err("a chroma plane of the wrong shape downloads");
    assert!(
        matches!(
            error,
            FrameError::PlaneMismatch {
                plane: "chroma",
                ..
            }
        ),
        "{error}"
    );

    let (y, uv) = (blank(&[4, 6], DType::U8), blank(&[2, 6], DType::U8));
    let display = Dimensions::new(6, 4).unwrap_or_else(|error| panic!("{error}"));
    let frame = EncodeFrame::from_packed_planes(&y, &uv, PixelFormat::Nv12, display);
    let error = download_planes(&frame, &mut out).expect_err("sample planes pass as words");
    assert!(
        matches!(error, FrameError::PlaneMismatch { plane: "luma", .. }),
        "{error}"
    );
}
