//! Hand-built sequence headers, frame headers, and OBUs for parser tests. The bits follow the
//! syntax tables of the AV1 specification.

use super::seq_header::SequenceHeader;
use crate::bitstream::test_bits::BitWriter;

/// A sequence header payload: 8-bit 4:2:0, 4 bits of order hint, and no optional tool.
/// `width` and `height` fit 12 bits, and `idc` is `operating_point_idc[ 0 ]`.
pub(crate) fn sequence_payload(width: u32, height: u32, idc: u64) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.u(3, 0).u(1, 0).u(1, 0);
    // No timing, no display delay, one operating point at level 2.0.
    w.u(1, 0).u(1, 0).u(5, 0).u(12, idc).u(5, 0);
    w.u(4, 11).u(4, 11);
    w.u(12, u64::from(width - 1)).u(12, u64::from(height - 1));
    // No frame ids, 64x64 superblocks, no intra tools.
    w.u(1, 0).u(1, 0).u(1, 0).u(1, 0);
    // No compound or warp tools, order hints on, no joint compound, no reference MVs.
    w.u(1, 0).u(1, 0).u(1, 0).u(1, 0).u(1, 1).u(1, 0).u(1, 0);
    // Screen content tools off, which reads no integer MV syntax.
    w.u(1, 0).u(1, 0);
    w.u(3, 3);
    // No superres, CDEF, or restoration.
    w.u(1, 0).u(1, 0).u(1, 0);
    // 8-bit, not mono, no description, limited range, chroma position 0, one delta.
    w.u(1, 0).u(1, 0).u(1, 0).u(1, 0).u(2, 0).u(1, 0);
    // No film grain.
    w.u(1, 0);
    w.u(1, 1).align();
    w.bytes()
}

/// What [`sequence_payload`] parses to, for tests that build no OBU.
pub(crate) fn plain_sequence(width: u32, height: u32) -> SequenceHeader {
    super::seq_header::parse_sequence_header(&sequence_payload(width, height, 0))
        .unwrap_or_else(|error| panic!("the test sequence does not parse: {error}"))
}

/// Writes `quantization_params()` to `reduced_tx_set` of an intra frame at `base_q_idx` 100. The
/// frame holds no segmentation, and loop filter levels 1, 2, 3, and 4.
fn write_intra_tail(w: &mut BitWriter) {
    w.u(8, 100).u(1, 0).u(1, 0).u(1, 0).u(1, 0);
    w.u(1, 0);
    w.u(1, 0);
    w.u(6, 1).u(6, 2).u(6, 3).u(6, 4).u(3, 0);
    w.u(1, 1).u(1, 0);
    w.u(1, 1);
    w.u(1, 0);
}

/// A shown key frame at `order_hint`, under [`sequence_payload`]. `two_tiles` turns on the one
/// `increment_tile_cols_log2` of a frame 128 wide, with `tile_size_bytes_minus_1` 0.
pub(crate) fn key_header(order_hint: u64, two_tiles: bool) -> BitWriter {
    let mut w = BitWriter::new();
    w.u(1, 0).u(2, 0).u(1, 1);
    w.u(1, 0);
    w.u(1, 0);
    w.u(4, order_hint);
    w.u(1, 0);
    w.u(1, 0);
    write_tile_info(&mut w, two_tiles);
    write_intra_tail(&mut w);
    w
}

fn write_tile_info(w: &mut BitWriter, two_tiles: bool) {
    w.u(1, 1);
    if two_tiles {
        w.u(1, 1).u(1, 0).u(2, 0);
    }
}

/// The options of [`inter_header`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct InterOptions {
    pub(crate) order_hint: u64,
    pub(crate) primary_ref_frame: u64,
    pub(crate) refresh_frame_flags: u64,
    pub(crate) ref_frame_idx: [u64; 7],
    pub(crate) show_frame: bool,
    pub(crate) showable_frame: bool,
}

impl Default for InterOptions {
    fn default() -> Self {
        Self {
            order_hint: 1,
            primary_ref_frame: 7,
            refresh_frame_flags: 1,
            ref_frame_idx: [0; 7],
            show_frame: true,
            showable_frame: false,
        }
    }
}

/// An inter frame under [`sequence_payload`], with explicit references and one tile. Loop filter
/// deltas stay as they load.
pub(crate) fn inter_header(options: InterOptions) -> BitWriter {
    let mut w = BitWriter::new();
    w.u(1, 0).u(2, 1).flag(options.show_frame);
    if !options.show_frame {
        w.flag(options.showable_frame);
    }
    w.u(1, 0);
    w.u(1, 0);
    w.u(1, 0);
    w.u(4, options.order_hint);
    w.u(3, options.primary_ref_frame);
    w.u(8, options.refresh_frame_flags);
    w.u(1, 0);
    for name in options.ref_frame_idx {
        w.u(3, name);
    }
    write_inter_tail(&mut w);
    w
}

/// Writes an inter frame from `render_and_frame_size_different` on.
pub(crate) fn write_inter_tail(w: &mut BitWriter) {
    w.u(1, 0);
    w.u(1, 0);
    w.u(1, 1);
    w.u(1, 0);
    w.u(1, 0);
    w.u(1, 1);
    w.u(8, 100).u(1, 0).u(1, 0).u(1, 0).u(1, 0);
    w.u(1, 0);
    w.u(1, 0);
    w.u(6, 1).u(6, 2).u(6, 3).u(6, 4).u(3, 0);
    w.u(1, 1).u(1, 0);
    w.u(1, 1);
    w.u(1, 0);
    w.u(1, 0);
    w.u(7, 0);
}

/// A `show_existing_frame` header of `name`.
pub(crate) fn show_existing_header(name: u64) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.u(1, 1).u(3, name);
    w.u(1, 1).align();
    w.bytes()
}

/// The payload of a frame header OBU: the header, then its trailing bits.
pub(crate) fn header_payload(header: &BitWriter) -> Vec<u8> {
    let mut w = header.clone();
    w.u(1, 1).align();
    w.bytes()
}

/// An OBU with a size field. `layer` is the temporal and spatial id of an extension header.
pub(crate) fn obu_bytes(obu_type: u8, layer: Option<(u8, u8)>, payload: &[u8]) -> Vec<u8> {
    let flag = u8::from(layer.is_some());
    let mut out = vec![(obu_type << 3) | (flag << 2) | 0x02];
    if let Some((temporal, spatial)) = layer {
        out.push((temporal << 5) | (spatial << 3));
    }
    super::obu::write_leb128(&mut out, payload.len() as u64);
    out.extend_from_slice(payload);
    out
}

/// The payload of a frame OBU: the header, its alignment, then `tiles`.
pub(crate) fn frame_payload(header: &BitWriter, tiles: &[u8]) -> Vec<u8> {
    let mut w = header.clone();
    w.align();
    let mut out = w.bytes();
    out.extend_from_slice(tiles);
    out
}
