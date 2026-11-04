//! `loop_filter_params()`, `cdef_params()`, `lr_params()`, and `read_tx_mode()` (AV1
//! specification 5.9.11, 5.9.19 to 5.9.21).

use super::quant::read_su;
use super::types::{
    DEFAULT_REF_DELTAS, FrameHeader, ONLY_4X4, REMAP_LR_TYPE, RESTORE_NONE, TX_MODE_LARGEST,
    TX_MODE_SELECT,
};
use crate::bitstream::av1::seq_header::SequenceHeader;
use crate::bitstream::bits::BitReader;
use crate::error::BitstreamError;

/// Reads `loop_filter_params()`. `header` already holds the deltas of `load_previous()`.
pub(super) fn loop_filter_params(
    reader: &mut BitReader<'_>,
    sequence: &SequenceHeader,
    header: &mut FrameHeader,
) -> Result<(), BitstreamError> {
    let skip = header.coded_lossless || header.allow_intrabc;
    let filter = &mut header.loop_filter;
    if skip {
        filter.ref_deltas = DEFAULT_REF_DELTAS;
        filter.mode_deltas = [0; 2];
        return Ok(());
    }
    filter.level[0] = reader.u8(6, "loop_filter_level")?;
    filter.level[1] = reader.u8(6, "loop_filter_level")?;
    if !sequence.color.mono_chrome && (filter.level[0] != 0 || filter.level[1] != 0) {
        filter.level[2] = reader.u8(6, "loop_filter_level")?;
        filter.level[3] = reader.u8(6, "loop_filter_level")?;
    }
    filter.sharpness = reader.u8(3, "loop_filter_sharpness")?;
    filter.delta_enabled = reader.bit("loop_filter_delta_enabled")?;
    if !filter.delta_enabled {
        return Ok(());
    }
    filter.delta_update = reader.bit("loop_filter_delta_update")?;
    if !filter.delta_update {
        return Ok(());
    }
    for index in 0..filter.ref_deltas.len() {
        filter.update_ref_delta[index] = reader.bit("update_ref_delta")?;
        if filter.update_ref_delta[index] {
            filter.ref_deltas[index] = read_su(reader, 7, "loop_filter_ref_deltas")? as i8;
        }
    }
    for index in 0..filter.mode_deltas.len() {
        filter.update_mode_delta[index] = reader.bit("update_mode_delta")?;
        if filter.update_mode_delta[index] {
            filter.mode_deltas[index] = read_su(reader, 7, "loop_filter_mode_deltas")? as i8;
        }
    }
    Ok(())
}

/// Reads `cdef_params()`. Secondary strengths keep their bitstream values, because Vulkan Video
/// takes those.
pub(super) fn cdef_params(
    reader: &mut BitReader<'_>,
    sequence: &SequenceHeader,
    header: &mut FrameHeader,
) -> Result<(), BitstreamError> {
    if header.coded_lossless || header.allow_intrabc || !sequence.enable_cdef {
        return Ok(());
    }
    let cdef = &mut header.cdef;
    cdef.damping_minus_3 = reader.u8(2, "cdef_damping_minus_3")?;
    cdef.bits = reader.u8(2, "cdef_bits")?;
    for index in 0..(1_usize << cdef.bits) {
        cdef.y_pri_strength[index] = reader.u8(4, "cdef_y_pri_strength")?;
        cdef.y_sec_strength[index] = reader.u8(2, "cdef_y_sec_strength")?;
        if !sequence.color.mono_chrome {
            cdef.uv_pri_strength[index] = reader.u8(4, "cdef_uv_pri_strength")?;
            cdef.uv_sec_strength[index] = reader.u8(2, "cdef_uv_sec_strength")?;
        }
    }
    Ok(())
}

/// Reads `lr_params()`.
pub(super) fn lr_params(
    reader: &mut BitReader<'_>,
    sequence: &SequenceHeader,
    header: &mut FrameHeader,
) -> Result<(), BitstreamError> {
    if header.all_lossless || header.allow_intrabc || !sequence.enable_restoration {
        return Ok(());
    }
    let lr = &mut header.loop_restoration;
    for plane in 0..sequence.color.num_planes() {
        lr.lr_type[plane] = reader.u8(2, "lr_type")?;
        let restoration = REMAP_LR_TYPE[usize::from(lr.lr_type[plane] & 3)];
        lr.frame_restoration_type[plane] = restoration;
        if restoration != RESTORE_NONE {
            lr.uses_lr = true;
            lr.uses_chroma_lr |= plane > 0;
        }
    }
    if !lr.uses_lr {
        return Ok(());
    }
    lr.lr_unit_shift = u8::from(reader.bit("lr_unit_shift")?);
    if sequence.use_128x128_superblock {
        lr.lr_unit_shift += 1;
    } else if lr.lr_unit_shift == 1 {
        lr.lr_unit_shift += u8::from(reader.bit("lr_unit_shift")?);
    }
    let subsampled = sequence.color.subsampling_x && sequence.color.subsampling_y;
    if subsampled && lr.uses_chroma_lr {
        lr.lr_uv_shift = u8::from(reader.bit("lr_uv_shift")?);
    }
    Ok(())
}

/// Reads `read_tx_mode()`.
pub(super) fn read_tx_mode(
    reader: &mut BitReader<'_>,
    header: &mut FrameHeader,
) -> Result<(), BitstreamError> {
    header.tx_mode = if header.coded_lossless {
        ONLY_4X4
    } else if reader.bit("tx_mode_select")? {
        TX_MODE_SELECT
    } else {
        TX_MODE_LARGEST
    };
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::av1::frame_header::{
        PRIMARY_REF_NONE, RESTORE_SGRPROJ, RESTORE_SWITCHABLE, RESTORE_WIENER,
    };
    use crate::bitstream::test_bits::BitWriter;

    fn parsed_filter(previous: [i8; 8], bits: &BitWriter) -> FrameHeader {
        let mut header = FrameHeader {
            primary_ref_frame: 0,
            ..FrameHeader::default()
        };
        header.loop_filter.ref_deltas = previous;
        header.loop_filter.mode_deltas = [3, -3];
        let bytes = bits.bytes();
        let mut reader = BitReader::new(&bytes, "test");
        loop_filter_params(&mut reader, &SequenceHeader::default(), &mut header).expect("parses");
        header
    }

    /// With `loop_filter_delta_update` 0, the deltas stay what the primary reference saved.
    #[test]
    fn the_previous_deltas_stay_with_no_update() {
        let mut w = BitWriter::new();
        w.u(6, 10)
            .u(6, 11)
            .u(6, 4)
            .u(6, 5)
            .u(3, 2)
            .flag(true)
            .flag(false);
        let header = parsed_filter([7, 6, 5, 4, 3, 2, 1, 0], &w);
        assert_eq!(header.loop_filter.level, [10, 11, 4, 5]);
        assert_eq!(header.loop_filter.sharpness, 2);
        assert_eq!(header.loop_filter.ref_deltas, [7, 6, 5, 4, 3, 2, 1, 0]);
        assert_eq!(header.loop_filter.mode_deltas, [3, -3]);
    }

    /// An update changes only the deltas it names.
    #[test]
    fn an_update_changes_only_the_named_deltas() {
        let mut w = BitWriter::new();
        // Both luma levels 0, so no chroma levels follow.
        w.u(6, 0).u(6, 0).u(3, 0).flag(true).flag(true);
        for index in 0..8 {
            w.flag(index == 2);
            if index == 2 {
                w.u(7, 0b111_1110);
            }
        }
        w.flag(false).flag(true).u(7, 5);
        let header = parsed_filter([7, 6, 5, 4, 3, 2, 1, 0], &w);
        assert_eq!(header.loop_filter.ref_deltas, [7, 6, -2, 4, 3, 2, 1, 0]);
        assert_eq!(header.loop_filter.mode_deltas, [3, 5]);
        assert_eq!(header.loop_filter.level, [0, 0, 0, 0]);
        assert!(header.loop_filter.update_ref_delta[2]);
        assert_eq!(header.loop_filter.update_mode_delta, [false, true]);
    }

    #[test]
    fn a_lossless_frame_resets_the_deltas_and_reads_no_bit() {
        let mut header = FrameHeader {
            coded_lossless: true,
            primary_ref_frame: PRIMARY_REF_NONE,
            ..FrameHeader::default()
        };
        header.loop_filter.ref_deltas = [9; 8];
        let mut reader = BitReader::new(&[], "test");
        loop_filter_params(&mut reader, &SequenceHeader::default(), &mut header).expect("parses");
        assert_eq!(header.loop_filter.ref_deltas, DEFAULT_REF_DELTAS);
    }

    /// The bitstream value 3 stays 3. Section 5.9.19 turns it into 4 for the decode only.
    #[test]
    fn cdef_keeps_the_bitstream_secondary_strength() {
        let sequence = SequenceHeader {
            enable_cdef: true,
            ..SequenceHeader::default()
        };
        let mut w = BitWriter::new();
        w.u(2, 2).u(2, 1);
        for strengths in [(5, 3, 7, 3), (1, 0, 2, 1)] {
            w.u(4, strengths.0)
                .u(2, strengths.1)
                .u(4, strengths.2)
                .u(2, strengths.3);
        }
        let bytes = w.bytes();
        let mut reader = BitReader::new(&bytes, "test");
        let mut header = FrameHeader::default();
        cdef_params(&mut reader, &sequence, &mut header).expect("parses");
        assert_eq!(header.cdef.damping_minus_3, 2);
        assert_eq!(header.cdef.y_sec_strength[0], 3);
        assert_eq!(header.cdef.uv_sec_strength[0], 3);
        assert_eq!(header.cdef.y_pri_strength[1], 1);
    }

    /// `lr_type` 1, 2, and 3 map to switchable, Wiener, and self-guided. An `lr_unit_shift` of 2
    /// gives 256 samples, and a `lr_uv_shift` of 1 gives chroma 128.
    #[test]
    fn restoration_maps_each_type_and_its_size() {
        let sequence = SequenceHeader {
            enable_restoration: true,
            color: crate::bitstream::av1::seq_header::ColorConfig {
                subsampling_x: true,
                subsampling_y: true,
                ..Default::default()
            },
            ..SequenceHeader::default()
        };
        let mut w = BitWriter::new();
        w.u(2, 1).u(2, 2).u(2, 3).flag(true).flag(true).flag(true);
        let bytes = w.bytes();
        let mut reader = BitReader::new(&bytes, "test");
        let mut header = FrameHeader::default();
        lr_params(&mut reader, &sequence, &mut header).expect("parses");
        let lr = header.loop_restoration;
        assert_eq!(
            lr.frame_restoration_type,
            [RESTORE_SWITCHABLE, RESTORE_WIENER, RESTORE_SGRPROJ]
        );
        assert!(lr.uses_lr && lr.uses_chroma_lr);
        assert_eq!(lr.lr_unit_shift, 2);
        assert_eq!(lr.log2_size_minus_5(0), 3);
        assert_eq!(lr.log2_size_minus_5(1), 2);
        assert_eq!(lr.log2_size_minus_5(2), 2);
    }
}
