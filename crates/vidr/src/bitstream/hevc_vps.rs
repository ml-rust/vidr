//! HEVC `video_parameter_set_rbsp()` (ITU-T H.265 7.3.2.1) for `nuh_layer_id` 0.
//!
//! `hrd_parameters()` and `layer_id_included_flag` are read and dropped,
//! because no decoding process reads them.

use super::bits::BitReader;
use super::hevc_params;
use super::hevc_ptl::{self, ProfileTierLevel};
use super::hevc_vui;
use super::nal::HevcNalType;
use crate::error::BitstreamError;

const STRUCTURE: &str = "HEVC VPS";
/// The largest `vps_max_dec_pic_buffering_minus1`. `MaxDpbSize` is at most 16
/// (A.4.2).
const MAX_DEC_PIC_BUFFERING_MINUS1: u32 = 15;

/// One `video_parameter_set_rbsp()`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct VideoParameterSet {
    pub(crate) vps_video_parameter_set_id: u8,
    pub(crate) vps_base_layer_internal_flag: bool,
    pub(crate) vps_base_layer_available_flag: bool,
    pub(crate) vps_max_layers_minus1: u8,
    pub(crate) vps_max_sub_layers_minus1: u8,
    pub(crate) vps_temporal_id_nesting_flag: bool,
    pub(crate) profile_tier_level: ProfileTierLevel,
    pub(crate) vps_sub_layer_ordering_info_present_flag: bool,
    /// Indexed by sub-layer, with the 7.4.3.1 inference for absent entries.
    pub(crate) vps_max_dec_pic_buffering_minus1: [u32; 7],
    /// Indexed by sub-layer, with the 7.4.3.1 inference for absent entries.
    pub(crate) vps_max_num_reorder_pics: [u32; 7],
    /// Indexed by sub-layer, with the 7.4.3.1 inference for absent entries.
    pub(crate) vps_max_latency_increase_plus1: [u32; 7],
    pub(crate) vps_max_layer_id: u8,
    pub(crate) vps_num_layer_sets_minus1: u32,
    pub(crate) vps_timing_info_present_flag: bool,
    pub(crate) vps_num_units_in_tick: u32,
    pub(crate) vps_time_scale: u32,
    pub(crate) vps_poc_proportional_to_timing_flag: bool,
    pub(crate) vps_num_ticks_poc_diff_one_minus1: u32,
    pub(crate) vps_num_hrd_parameters: u32,
    pub(crate) vps_extension_flag: bool,
}

impl VideoParameterSet {
    /// Reads a VPS NAL unit, 2-byte header included.
    pub(crate) fn parse(nal: &[u8]) -> Result<Self, BitstreamError> {
        let rbsp = hevc_params::payload(nal, HevcNalType::VPS, STRUCTURE)?;
        let mut reader = BitReader::new(&rbsp, STRUCTURE);
        let vps_video_parameter_set_id = reader.u8(4, "vps_video_parameter_set_id")?;
        let vps_base_layer_internal_flag = reader.bit("vps_base_layer_internal_flag")?;
        let vps_base_layer_available_flag = reader.bit("vps_base_layer_available_flag")?;
        let vps_max_layers_minus1 = reader.u8(6, "vps_max_layers_minus1")?;
        if vps_max_layers_minus1 >= 63 {
            return Err(reader.invalid(
                "vps_max_layers_minus1",
                vps_max_layers_minus1.into(),
                "0 to 62",
            ));
        }
        let vps_max_sub_layers_minus1 = reader.u8(3, "vps_max_sub_layers_minus1")?;
        if vps_max_sub_layers_minus1 > 6 {
            return Err(reader.invalid(
                "vps_max_sub_layers_minus1",
                vps_max_sub_layers_minus1.into(),
                "0 to 6",
            ));
        }
        let vps_temporal_id_nesting_flag = reader.bit("vps_temporal_id_nesting_flag")?;
        if vps_max_sub_layers_minus1 == 0 && !vps_temporal_id_nesting_flag {
            return Err(reader.invalid(
                "vps_temporal_id_nesting_flag",
                0,
                "1 when vps_max_sub_layers_minus1 is 0",
            ));
        }
        // A decoder ignores the value of `vps_reserved_0xffff_16bits` (7.4.3.1).
        reader.skip(16, "vps_reserved_0xffff_16bits")?;
        let profile_tier_level = hevc_ptl::parse_full(&mut reader, vps_max_sub_layers_minus1)?;

        let top = usize::from(vps_max_sub_layers_minus1);
        let vps_sub_layer_ordering_info_present_flag =
            reader.bit("vps_sub_layer_ordering_info_present_flag")?;
        let first = if vps_sub_layer_ordering_info_present_flag {
            0
        } else {
            top
        };
        let mut vps_max_dec_pic_buffering_minus1 = [0u32; 7];
        let mut vps_max_num_reorder_pics = [0u32; 7];
        let mut vps_max_latency_increase_plus1 = [0u32; 7];
        for layer in first..=top {
            vps_max_dec_pic_buffering_minus1[layer] =
                reader.ue("vps_max_dec_pic_buffering_minus1")?;
            vps_max_num_reorder_pics[layer] = reader.ue("vps_max_num_reorder_pics")?;
            vps_max_latency_increase_plus1[layer] = reader.ue("vps_max_latency_increase_plus1")?;
        }
        for layer in 0..first {
            vps_max_dec_pic_buffering_minus1[layer] = vps_max_dec_pic_buffering_minus1[top];
            vps_max_num_reorder_pics[layer] = vps_max_num_reorder_pics[top];
            vps_max_latency_increase_plus1[layer] = vps_max_latency_increase_plus1[top];
        }
        // These limits apply only with an internal base layer (7.4.3.1).
        if vps_base_layer_internal_flag {
            check_ordering(
                &reader,
                &vps_max_dec_pic_buffering_minus1[..=top],
                &vps_max_num_reorder_pics[..=top],
            )?;
        }

        let vps_max_layer_id = reader.u8(6, "vps_max_layer_id")?;
        if vps_max_layer_id >= 63 {
            return Err(reader.invalid("vps_max_layer_id", vps_max_layer_id.into(), "0 to 62"));
        }
        let vps_num_layer_sets_minus1 =
            reader.ue_max("vps_num_layer_sets_minus1", 1023, "0 to 1023")?;
        reader.skip(
            u64::from(vps_num_layer_sets_minus1) * (u64::from(vps_max_layer_id) + 1),
            "layer_id_included_flag",
        )?;

        let mut vps = Self {
            vps_video_parameter_set_id,
            vps_base_layer_internal_flag,
            vps_base_layer_available_flag,
            vps_max_layers_minus1,
            vps_max_sub_layers_minus1,
            vps_temporal_id_nesting_flag,
            profile_tier_level,
            vps_sub_layer_ordering_info_present_flag,
            vps_max_dec_pic_buffering_minus1,
            vps_max_num_reorder_pics,
            vps_max_latency_increase_plus1,
            vps_max_layer_id,
            vps_num_layer_sets_minus1,
            vps_timing_info_present_flag: reader.bit("vps_timing_info_present_flag")?,
            vps_num_units_in_tick: 0,
            vps_time_scale: 0,
            vps_poc_proportional_to_timing_flag: false,
            vps_num_ticks_poc_diff_one_minus1: 0,
            vps_num_hrd_parameters: 0,
            vps_extension_flag: false,
        };
        if vps.vps_timing_info_present_flag {
            vps.read_timing(&mut reader)?;
        }
        // A decoder ignores the extension data that follows this flag.
        vps.vps_extension_flag = reader.bit("vps_extension_flag")?;
        Ok(vps)
    }

    fn read_timing(&mut self, reader: &mut BitReader<'_>) -> Result<(), BitstreamError> {
        self.vps_num_units_in_tick = reader.bits(32, "vps_num_units_in_tick")? as u32;
        if self.vps_num_units_in_tick == 0 {
            return Err(reader.invalid("vps_num_units_in_tick", 0, "above 0"));
        }
        self.vps_time_scale = reader.bits(32, "vps_time_scale")? as u32;
        if self.vps_time_scale == 0 {
            return Err(reader.invalid("vps_time_scale", 0, "above 0"));
        }
        self.vps_poc_proportional_to_timing_flag =
            reader.bit("vps_poc_proportional_to_timing_flag")?;
        if self.vps_poc_proportional_to_timing_flag {
            self.vps_num_ticks_poc_diff_one_minus1 =
                reader.ue("vps_num_ticks_poc_diff_one_minus1")?;
        }
        self.vps_num_hrd_parameters = reader.ue_max(
            "vps_num_hrd_parameters",
            self.vps_num_layer_sets_minus1 + 1,
            "0 to vps_num_layer_sets_minus1 + 1",
        )?;
        let lowest_set = u32::from(!self.vps_base_layer_internal_flag);
        let mut common = None;
        for index in 0..self.vps_num_hrd_parameters {
            let layer_set = reader.ue_max(
                "hrd_layer_set_idx",
                self.vps_num_layer_sets_minus1,
                "at most vps_num_layer_sets_minus1",
            )?;
            if layer_set < lowest_set {
                return Err(reader.invalid(
                    "hrd_layer_set_idx",
                    layer_set.into(),
                    "at least 1 without an internal base layer",
                ));
            }
            // `cprms_present_flag[ 0 ]` is inferred as 1.
            let cprms_present = index == 0 || reader.bit("cprms_present_flag")?;
            let inherited = if cprms_present { None } else { common };
            common = Some(hevc_vui::skip_hrd_parameters(
                reader,
                inherited,
                self.vps_max_sub_layers_minus1,
            )?);
        }
        Ok(())
    }
}

/// Checks `vps_max_dec_pic_buffering_minus1` and `vps_max_num_reorder_pics`
/// (7.4.3.1): each in range, and neither falling from one sub-layer to the next.
fn check_ordering(
    reader: &BitReader<'_>,
    dec_pic_buffering: &[u32],
    num_reorder: &[u32],
) -> Result<(), BitstreamError> {
    let mut previous = (0, 0);
    for (&buffering, &reorder) in dec_pic_buffering.iter().zip(num_reorder) {
        if buffering > MAX_DEC_PIC_BUFFERING_MINUS1 || buffering < previous.0 {
            return Err(reader.invalid(
                "vps_max_dec_pic_buffering_minus1",
                buffering.into(),
                "0 to 15, and no lower than the sub-layer before it",
            ));
        }
        if reorder > buffering || reorder < previous.1 {
            return Err(reader.invalid(
                "vps_max_num_reorder_pics",
                reorder.into(),
                "0 to vps_max_dec_pic_buffering_minus1, and no lower than the sub-layer before it",
            ));
        }
        previous = (buffering, reorder);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::hevc_params::tests::X265_VPS;
    use super::super::test_bits::{BitWriter, hevc_nal};
    use super::*;

    /// The values a test varies. The defaults make a valid single-layer VPS.
    #[derive(Clone, Copy)]
    struct Spec {
        internal: bool,
        max_layers_minus1: u64,
        sub_layers_minus1: u64,
        nesting: bool,
        ordering: bool,
        dec_pic_buffering: [u32; 2],
        num_reorder: [u32; 2],
        max_layer_id: u64,
        layer_sets_minus1: u32,
        timing: bool,
        units_in_tick: u64,
        time_scale: u64,
        num_hrd: u32,
        layer_set_idx: u32,
    }

    const SPEC: Spec = Spec {
        internal: true,
        max_layers_minus1: 0,
        sub_layers_minus1: 0,
        nesting: true,
        ordering: true,
        dec_pic_buffering: [4, 5],
        num_reorder: [1, 2],
        max_layer_id: 0,
        layer_sets_minus1: 0,
        timing: false,
        units_in_tick: 1001,
        time_scale: 30000,
        num_hrd: 0,
        layer_set_idx: 0,
    };

    fn build(spec: Spec) -> Vec<u8> {
        let mut w = BitWriter::new();
        w.u(4, 3)
            .flag(spec.internal)
            .flag(true)
            .u(6, spec.max_layers_minus1)
            .u(3, spec.sub_layers_minus1)
            .flag(spec.nesting)
            .u(16, 0xffff);
        // General profile, tier, and level: 96 bits, level 120 last.
        w.u(8, 1)
            .u(32, 0x6000_0000)
            .u(48, 0x9000_0000_0000)
            .u(8, 120);
        let layers = spec.sub_layers_minus1.min(6) as usize;
        for _ in 0..layers {
            w.flag(false).flag(false);
        }
        if layers > 0 {
            w.u(2 * (8 - layers as u32), 0);
        }
        w.flag(spec.ordering);
        let first = if spec.ordering { 0 } else { layers };
        for layer in first..=layers {
            let at = layer.min(1);
            w.ue(spec.dec_pic_buffering[at])
                .ue(spec.num_reorder[at])
                .ue(layer as u32);
        }
        w.u(6, spec.max_layer_id).ue(spec.layer_sets_minus1);
        for _ in 0..u64::from(spec.layer_sets_minus1) * (spec.max_layer_id + 1) {
            w.flag(false);
        }
        w.flag(spec.timing);
        if spec.timing {
            w.u(32, spec.units_in_tick)
                .u(32, spec.time_scale)
                .flag(true)
                .ue(9)
                .ue(spec.num_hrd);
            for index in 0..spec.num_hrd {
                w.ue(spec.layer_set_idx);
                if index > 0 {
                    w.flag(false);
                }
                if index == 0 {
                    // NAL only, no sub-picture parameters: 8 + 15 bits.
                    w.flag(true).flag(false).flag(false).u(23, 0);
                }
                // Per sub-layer: fixed_general, elemental, cpb_cnt, one NAL entry.
                for _ in 0..=layers {
                    w.flag(true).ue(0).ue(0).ue(0).ue(0).flag(false);
                }
            }
        }
        w.flag(false).trailing();
        hevc_nal(32, &w.bytes())
    }

    fn parse(spec: Spec) -> Result<VideoParameterSet, BitstreamError> {
        VideoParameterSet::parse(&build(spec))
    }

    #[test]
    fn reads_a_real_vps() {
        let vps = VideoParameterSet::parse(&X265_VPS).unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(vps.vps_max_sub_layers_minus1, 0);
        assert!(vps.vps_temporal_id_nesting_flag);
        assert_eq!(vps.profile_tier_level.general.profile_idc, 1);
        assert!(!vps.vps_extension_flag);
    }

    #[test]
    fn reads_every_field_of_a_hand_built_vps() {
        let spec = Spec {
            sub_layers_minus1: 2,
            nesting: false,
            max_layer_id: 1,
            layer_sets_minus1: 1,
            timing: true,
            num_hrd: 2,
            layer_set_idx: 1,
            ..SPEC
        };
        let vps = parse(spec).unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(vps.vps_video_parameter_set_id, 3);
        assert_eq!(vps.vps_max_sub_layers_minus1, 2);
        assert_eq!(vps.vps_max_dec_pic_buffering_minus1[..3], [4, 5, 5]);
        assert_eq!(vps.vps_max_num_reorder_pics[..3], [1, 2, 2]);
        assert_eq!(vps.vps_max_latency_increase_plus1[..3], [0, 1, 2]);
        assert_eq!(vps.vps_num_units_in_tick, 1001);
        assert_eq!(vps.vps_time_scale, 30000);
        assert!(vps.vps_poc_proportional_to_timing_flag);
        assert_eq!(vps.vps_num_ticks_poc_diff_one_minus1, 9);
        assert_eq!(vps.vps_num_hrd_parameters, 2);
        assert!(!vps.vps_extension_flag);
    }

    #[test]
    fn absent_ordering_info_copies_the_top_sub_layer() {
        let spec = Spec {
            sub_layers_minus1: 2,
            nesting: false,
            ordering: false,
            ..SPEC
        };
        let vps = parse(spec).unwrap_or_else(|error| panic!("{error}"));
        assert!(!vps.vps_sub_layer_ordering_info_present_flag);
        assert_eq!(vps.vps_max_dec_pic_buffering_minus1[..3], [5, 5, 5]);
        assert_eq!(vps.vps_max_latency_increase_plus1[..3], [2, 2, 2]);
    }

    #[test]
    fn out_of_range_values_are_refused() {
        let cases: [(&str, u64, Spec); 12] = [
            (
                "vps_max_layers_minus1",
                63,
                Spec {
                    max_layers_minus1: 63,
                    ..SPEC
                },
            ),
            (
                "vps_max_sub_layers_minus1",
                7,
                Spec {
                    sub_layers_minus1: 7,
                    ..SPEC
                },
            ),
            (
                "vps_temporal_id_nesting_flag",
                0,
                Spec {
                    nesting: false,
                    ..SPEC
                },
            ),
            (
                "vps_max_dec_pic_buffering_minus1",
                16,
                Spec {
                    dec_pic_buffering: [16, 16],
                    ..SPEC
                },
            ),
            (
                "vps_max_dec_pic_buffering_minus1",
                3,
                Spec {
                    sub_layers_minus1: 1,
                    nesting: false,
                    dec_pic_buffering: [4, 3],
                    num_reorder: [0, 0],
                    ..SPEC
                },
            ),
            (
                "vps_max_num_reorder_pics",
                5,
                Spec {
                    num_reorder: [5, 5],
                    ..SPEC
                },
            ),
            (
                "vps_max_num_reorder_pics",
                1,
                Spec {
                    sub_layers_minus1: 1,
                    nesting: false,
                    num_reorder: [2, 1],
                    ..SPEC
                },
            ),
            (
                "vps_max_layer_id",
                63,
                Spec {
                    max_layer_id: 63,
                    ..SPEC
                },
            ),
            (
                "vps_num_layer_sets_minus1",
                1024,
                Spec {
                    layer_sets_minus1: 1024,
                    ..SPEC
                },
            ),
            (
                "vps_num_units_in_tick",
                0,
                Spec {
                    timing: true,
                    units_in_tick: 0,
                    ..SPEC
                },
            ),
            (
                "vps_time_scale",
                0,
                Spec {
                    timing: true,
                    time_scale: 0,
                    ..SPEC
                },
            ),
            (
                "vps_num_hrd_parameters",
                2,
                Spec {
                    timing: true,
                    num_hrd: 2,
                    ..SPEC
                },
            ),
        ];
        for (name, expected, spec) in cases {
            assert!(
                matches!(
                    parse(spec),
                    Err(BitstreamError::InvalidField { field, value, .. })
                        if field == name && value == expected
                ),
                "{name}: {:?}",
                parse(spec)
            );
        }
        // The index above the layer set count, and below the lowest set.
        let high = Spec {
            timing: true,
            num_hrd: 1,
            layer_set_idx: 1,
            ..SPEC
        };
        assert!(matches!(
            parse(high),
            Err(BitstreamError::InvalidField {
                field: "hrd_layer_set_idx",
                value: 1,
                ..
            })
        ));
        let low = Spec {
            internal: false,
            timing: true,
            num_hrd: 1,
            ..SPEC
        };
        assert!(matches!(
            parse(low),
            Err(BitstreamError::InvalidField {
                field: "hrd_layer_set_idx",
                value: 0,
                ..
            })
        ));
    }

    #[test]
    fn ordering_limits_are_off_without_an_internal_base_layer() {
        let spec = Spec {
            internal: false,
            dec_pic_buffering: [16, 16],
            ..SPEC
        };
        assert!(parse(spec).is_ok());
    }

    #[test]
    fn truncation_at_every_byte_never_panics() {
        let spec = Spec {
            sub_layers_minus1: 2,
            nesting: false,
            max_layer_id: 1,
            layer_sets_minus1: 1,
            timing: true,
            num_hrd: 2,
            layer_set_idx: 1,
            ..SPEC
        };
        let nal = build(spec);
        for end in 0..nal.len() {
            let result = VideoParameterSet::parse(&nal[..end]);
            // Only the last byte holds nothing but the trailing bits.
            if end + 1 < nal.len() {
                assert!(result.is_err(), "cut at byte {end}");
            }
        }
    }

    #[test]
    fn wrong_nal_type_is_refused() {
        let mut nal = build(SPEC);
        nal[0] = 33 << 1;
        assert!(matches!(
            VideoParameterSet::parse(&nal),
            Err(BitstreamError::InvalidField {
                field: "nal_unit_type",
                value: 33,
                ..
            })
        ));
    }
}
