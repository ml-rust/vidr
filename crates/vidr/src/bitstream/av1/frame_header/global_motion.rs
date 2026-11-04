//! `global_motion_params()` and its subexp codes (AV1 specification 5.9.24 to 5.9.29).

use super::tile_info::read_ns;
use super::types::{
    AFFINE, FrameHeader, IDENTITY, IDENTITY_WARP, LAST_FRAME, REFS_PER_FRAME, ROTZOOM,
    TOTAL_REFS_PER_FRAME, TRANSLATION, WARPEDMODEL_PREC_BITS,
};
use crate::bitstream::bits::BitReader;
use crate::error::BitstreamError;

/// `GM_ABS_ALPHA_BITS`.
const GM_ABS_ALPHA_BITS: u32 = 12;
/// `GM_ALPHA_PREC_BITS`.
const GM_ALPHA_PREC_BITS: u32 = 15;
/// `GM_ABS_TRANS_ONLY_BITS`.
const GM_ABS_TRANS_ONLY_BITS: u32 = 9;
/// `GM_TRANS_ONLY_PREC_BITS`.
const GM_TRANS_ONLY_PREC_BITS: u32 = 3;
/// `GM_ABS_TRANS_BITS`.
const GM_ABS_TRANS_BITS: u32 = 12;
/// `GM_TRANS_PREC_BITS`.
const GM_TRANS_PREC_BITS: u32 = 6;

const fn inverse_recenter(r: i32, v: i32) -> i32 {
    if v > 2 * r {
        v
    } else if v & 1 == 1 {
        r - ((v + 1) >> 1)
    } else {
        r + (v >> 1)
    }
}

fn decode_subexp(reader: &mut BitReader<'_>, num_syms: i32) -> Result<i32, BitstreamError> {
    let mut i = 0_u32;
    let mut mk = 0_i32;
    let k = 3_u32;
    loop {
        let b2 = if i > 0 { k + i - 1 } else { k };
        let a = 1_i32 << b2;
        if num_syms <= mk + 3 * a {
            let bits = read_ns(reader, (num_syms - mk) as u32, "subexp_final_bits")?;
            return Ok(bits as i32 + mk);
        }
        if reader.bit("subexp_more_bits")? {
            i += 1;
            mk += a;
        } else {
            let bits = reader.bits(b2, "subexp_bits")? as i32;
            return Ok(bits + mk);
        }
    }
}

fn decode_unsigned_subexp_with_ref(
    reader: &mut BitReader<'_>,
    mx: i32,
    r: i32,
) -> Result<i32, BitstreamError> {
    let v = decode_subexp(reader, mx)?;
    Ok(if (r << 1) <= mx {
        inverse_recenter(r, v)
    } else {
        mx - 1 - inverse_recenter(mx - 1 - r, v)
    })
}

fn decode_signed_subexp_with_ref(
    reader: &mut BitReader<'_>,
    low: i32,
    high: i32,
    r: i32,
) -> Result<i32, BitstreamError> {
    Ok(decode_unsigned_subexp_with_ref(reader, high - low, r - low)? + low)
}

fn read_global_param(
    reader: &mut BitReader<'_>,
    kind: u8,
    idx: usize,
    previous: i32,
    high_precision: bool,
) -> Result<i32, BitstreamError> {
    let (abs_bits, prec_bits) = if idx >= 2 {
        (GM_ABS_ALPHA_BITS, GM_ALPHA_PREC_BITS)
    } else if kind == TRANSLATION {
        let reduce = u32::from(!high_precision);
        (
            GM_ABS_TRANS_ONLY_BITS - reduce,
            GM_TRANS_ONLY_PREC_BITS - reduce,
        )
    } else {
        (GM_ABS_TRANS_BITS, GM_TRANS_PREC_BITS)
    };
    let prec_diff = WARPEDMODEL_PREC_BITS - prec_bits;
    let diagonal = idx % 3 == 2;
    let round = if diagonal {
        1_i32 << WARPEDMODEL_PREC_BITS
    } else {
        0
    };
    let sub = if diagonal { 1_i32 << prec_bits } else { 0 };
    let mx = 1_i32 << abs_bits;
    let r = (previous >> prec_diff) - sub;
    let value = decode_signed_subexp_with_ref(reader, -mx, mx + 1, r)?;
    Ok((value << prec_diff) + round)
}

/// Reads `global_motion_params()`. `previous` is `PrevGmParams`.
pub(super) fn global_motion_params(
    reader: &mut BitReader<'_>,
    previous: &[[i32; 6]; TOTAL_REFS_PER_FRAME],
    header: &mut FrameHeader,
) -> Result<(), BitstreamError> {
    header.gm_type = [IDENTITY; TOTAL_REFS_PER_FRAME];
    header.gm_params = [IDENTITY_WARP; TOTAL_REFS_PER_FRAME];
    if header.frame_is_intra {
        return Ok(());
    }
    let saved = previous.iter().enumerate();
    for (reference, &last) in saved.skip(LAST_FRAME).take(REFS_PER_FRAME) {
        let kind = if reader.bit("is_global")? {
            if reader.bit("is_rot_zoom")? {
                ROTZOOM
            } else if reader.bit("is_translation")? {
                TRANSLATION
            } else {
                AFFINE
            }
        } else {
            IDENTITY
        };
        header.gm_type[reference] = kind;
        let high_precision = header.allow_high_precision_mv;
        let params = &mut header.gm_params[reference];
        if kind >= ROTZOOM {
            params[2] = read_global_param(reader, kind, 2, last[2], high_precision)?;
            params[3] = read_global_param(reader, kind, 3, last[3], high_precision)?;
            if kind == AFFINE {
                params[4] = read_global_param(reader, kind, 4, last[4], high_precision)?;
                params[5] = read_global_param(reader, kind, 5, last[5], high_precision)?;
            } else {
                params[4] = -params[3];
                params[5] = params[2];
            }
        }
        if kind >= TRANSLATION {
            params[0] = read_global_param(reader, kind, 0, last[0], high_precision)?;
            params[1] = read_global_param(reader, kind, 1, last[1], high_precision)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::test_bits::BitWriter;

    #[test]
    fn inverse_recenter_holds_its_three_branches() {
        assert_eq!(inverse_recenter(3, 7), 7);
        assert_eq!(inverse_recenter(3, 1), 2);
        assert_eq!(inverse_recenter(3, 2), 4);
        assert_eq!(inverse_recenter(3, 0), 3);
    }

    /// The first band holds 8 values in 3 bits, with `subexp_more_bits` 0.
    #[test]
    fn subexp_reads_the_first_and_the_next_band() {
        let mut w = BitWriter::new();
        w.flag(false).u(3, 5);
        // More bits, then the next band of 8 values: 8 + 2.
        w.flag(true).flag(false).u(3, 2);
        let bytes = w.bytes();
        let mut reader = BitReader::new(&bytes, "test");
        assert_eq!(decode_subexp(&mut reader, 8193), Ok(5));
        assert_eq!(decode_subexp(&mut reader, 8193), Ok(10));
    }

    /// A rotzoom reference with a delta of 0 from the identity keeps the identity. Its entries 4
    /// and 5 come from 3 and 2.
    #[test]
    fn a_zero_delta_keeps_the_previous_parameters() {
        let mut w = BitWriter::new();
        // Reference LAST: is_global, is_rot_zoom.
        w.flag(true).flag(true);
        // Four parameters, each `subexp_more_bits` 0 and `subexp_bits` 0. The last two are the
        // translation at `GM_ABS_TRANS_BITS`.
        for _ in 0..4 {
            w.flag(false).u(3, 0);
        }
        // Six references with no global motion.
        w.u(6, 0);
        let bytes = w.bytes();
        let mut reader = BitReader::new(&bytes, "test");
        let mut header = FrameHeader::default();
        let previous = [IDENTITY_WARP; TOTAL_REFS_PER_FRAME];
        global_motion_params(&mut reader, &previous, &mut header).expect("parses");
        assert_eq!(header.gm_type[1], ROTZOOM);
        assert_eq!(header.gm_params[1], IDENTITY_WARP);
        assert_eq!(header.gm_type[2], IDENTITY);
    }

    /// `PrevGmParams` steers the result: the same code gives another value from another reference
    /// value. Parameter 2 holds 1 << 16 at rest, at 15 bits of precision.
    #[test]
    fn the_previous_parameters_shift_the_result() {
        let mut previous = [IDENTITY_WARP; TOTAL_REFS_PER_FRAME];
        previous[1][2] = (1 << 16) + 2 * 40;
        let mut w = BitWriter::new();
        w.flag(true).flag(true);
        for _ in 0..4 {
            w.flag(false).u(3, 0);
        }
        w.u(6, 0);
        let bytes = w.bytes();
        let mut reader = BitReader::new(&bytes, "test");
        let mut header = FrameHeader::default();
        global_motion_params(&mut reader, &previous, &mut header).expect("parses");
        assert_eq!(header.gm_params[1][2], (1 << 16) + 80);
        assert_eq!(header.gm_params[1][5], (1 << 16) + 80);
    }

    #[test]
    fn an_intra_frame_reads_no_bit_and_holds_the_identity() {
        let mut reader = BitReader::new(&[], "test");
        let mut header = FrameHeader {
            frame_is_intra: true,
            ..FrameHeader::default()
        };
        let previous = [[7; 6]; TOTAL_REFS_PER_FRAME];
        global_motion_params(&mut reader, &previous, &mut header).expect("parses");
        assert_eq!(header.gm_params[3], IDENTITY_WARP);
    }
}
