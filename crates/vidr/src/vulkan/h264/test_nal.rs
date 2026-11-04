//! Hand-built H.264 headers for tests, on the shared bit writer.

use crate::bitstream::test_bits::{BitWriter, escape};

/// Writes header syntax behind a NAL header byte.
pub(super) struct NalWriter {
    bits: BitWriter,
}

impl NalWriter {
    /// Starts a NAL unit with its header byte.
    pub(super) fn new(header: u8) -> Self {
        let mut bits = BitWriter::new();
        bits.u(8, u64::from(header));
        Self { bits }
    }

    /// Writes the low `count` bits of `value`.
    pub(super) fn bits(&mut self, value: u32, count: u32) -> &mut Self {
        self.bits.u(count, u64::from(value));
        self
    }

    /// Writes one flag.
    pub(super) fn flag(&mut self, value: bool) -> &mut Self {
        self.bits.flag(value);
        self
    }

    /// Writes `ue(v)`.
    pub(super) fn ue(&mut self, value: u32) -> &mut Self {
        self.bits.ue(value);
        self
    }

    /// Writes `se(v)`.
    pub(super) fn se(&mut self, value: i32) -> &mut Self {
        self.bits.se(i64::from(value));
        self
    }

    /// Ends the unit with trailing bits, and adds emulation prevention bytes.
    pub(super) fn finish(&mut self) -> Vec<u8> {
        self.bits.trailing();
        escape(&self.bits.bytes())
    }
}

/// The SPS of a libx264 320x180 High stream, with VUI and a bottom crop of 12 rows.
pub(super) const REAL_SPS: [u8; 28] = [
    0x67, 0x64, 0x00, 0x0c, 0xac, 0xd9, 0x41, 0x41, 0x9f, 0x9f, 0x01, 0x6a, 0x02, 0x02, 0x02, 0x80,
    0x00, 0x00, 0x03, 0x00, 0x80, 0x00, 0x00, 0x19, 0x07, 0x8a, 0x14, 0xcb,
];
/// The PPS of the same stream.
pub(super) const REAL_PPS: [u8; 6] = [0x68, 0xeb, 0xe3, 0xcb, 0x22, 0xc0];

/// What a test SPS varies.
#[derive(Clone, Copy)]
pub(super) struct TestSps {
    pub(super) profile_idc: u8,
    pub(super) constraint_flags: u8,
    pub(super) id: u32,
    pub(super) poc_type: u32,
    pub(super) refs: u32,
    pub(super) frame_mbs_only: bool,
    pub(super) gaps_allowed: bool,
    pub(super) bit_depth_minus8: u32,
    pub(super) chroma_format: u32,
}

impl Default for TestSps {
    fn default() -> Self {
        Self {
            profile_idc: 100,
            constraint_flags: 0,
            id: 0,
            poc_type: 0,
            refs: 4,
            frame_mbs_only: true,
            gaps_allowed: false,
            bit_depth_minus8: 0,
            chroma_format: 1,
        }
    }
}

/// A 320x192 SPS at level 3.0, with `log2_max_frame_num` 4 and `log2_max_poc_lsb` 6.
pub(super) fn sps_nal(sps: &TestSps) -> Vec<u8> {
    let mut nal = NalWriter::new(0x67);
    nal.bits(u32::from(sps.profile_idc), 8)
        .bits(u32::from(sps.constraint_flags), 8)
        .bits(30, 8)
        .ue(sps.id);
    if matches!(sps.profile_idc, 100 | 110 | 122 | 244) {
        nal.ue(sps.chroma_format);
        if sps.chroma_format == 3 {
            nal.flag(false);
        }
        nal.ue(sps.bit_depth_minus8)
            .ue(sps.bit_depth_minus8)
            .flag(false)
            .flag(false);
    }
    nal.ue(0).ue(sps.poc_type);
    match sps.poc_type {
        0 => {
            nal.ue(2);
        }
        1 => {
            nal.flag(false).se(-1).se(1).ue(2).se(2).se(4);
        }
        _ => {}
    }
    nal.ue(sps.refs).flag(sps.gaps_allowed).ue(19).ue(11);
    nal.flag(sps.frame_mbs_only);
    if !sps.frame_mbs_only {
        nal.flag(true);
    }
    nal.flag(true).flag(false).flag(false);
    nal.finish()
}

/// A PPS for SPS `sps_id`. Slice groups, when present, take map type 2.
pub(super) fn pps_nal(id: u32, sps_id: u32, slice_groups_minus1: u32) -> Vec<u8> {
    let mut nal = NalWriter::new(0x68);
    nal.ue(id)
        .ue(sps_id)
        .flag(false)
        .flag(false)
        .ue(slice_groups_minus1);
    if slice_groups_minus1 > 0 {
        nal.ue(2);
        for _ in 0..slice_groups_minus1 {
            nal.ue(0).ue(0);
        }
    }
    nal.ue(0).ue(0).flag(false).bits(0, 2).se(0).se(0).se(0);
    nal.flag(true).flag(false).flag(false);
    nal.finish()
}
