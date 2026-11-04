//! The stream check for this decoder, and a flat parameter set for the parser.

use super::error::HevcError;
use super::layout::{LayoutSizes, PicLayout};
use crate::bitstream::hevc_pps::PicParameterSet;
use crate::bitstream::hevc_slice::{SLICE_TYPE_I, SLICE_TYPE_P, SliceSegmentHeader};
use crate::bitstream::hevc_sps_full::SeqParameterSet;

/// `MaxLumaPs` of levels 6 to 6.2 (Table A.8).
const MAX_LUMA_PICTURE_SIZE: u64 = 35_651_584;
/// `Sqrt( MaxLumaPs * 8 )`, rounded down (A.4.1).
const MAX_PICTURE_SIDE: u32 = 16_888;

/// The slice types (Table 7-7).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SliceKind {
    /// B slice.
    B,
    /// P slice.
    P,
    /// I slice.
    I,
}

/// PCM sizes of the SPS (7.4.3.2.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PcmParams {
    /// `Log2MinIpcmCbSizeY`.
    pub(crate) log2_min_size: u32,
    /// `Log2MaxIpcmCbSizeY`.
    pub(crate) log2_max_size: u32,
    /// `PcmBitDepthY`.
    pub(crate) bit_depth_luma: u32,
    /// `PcmBitDepthC`.
    pub(crate) bit_depth_chroma: u32,
}

/// Everything the parser reads from the SPS, PPS, and slice header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CodingParams {
    /// `pic_width_in_luma_samples`.
    pub(crate) width: u32,
    /// `pic_height_in_luma_samples`.
    pub(crate) height: u32,
    /// `CtbLog2SizeY`.
    pub(crate) ctb_log2: u32,
    /// `MinCbLog2SizeY`.
    pub(crate) min_cb_log2: u32,
    /// `MinTbLog2SizeY`.
    pub(crate) min_tb_log2: u32,
    /// `MaxTbLog2SizeY`.
    pub(crate) max_tb_log2: u32,
    /// `max_transform_hierarchy_depth_inter`.
    pub(crate) max_depth_inter: u32,
    /// `max_transform_hierarchy_depth_intra`.
    pub(crate) max_depth_intra: u32,
    /// `amp_enabled_flag`.
    pub(crate) amp_enabled: bool,
    /// `None` without `pcm_enabled_flag`.
    pub(crate) pcm: Option<PcmParams>,
    /// `BitDepthY`, equal to `BitDepthC`.
    pub(crate) bit_depth: u32,
    /// `sign_data_hiding_enabled_flag`.
    pub(crate) sign_data_hiding: bool,
    /// `transform_skip_enabled_flag`.
    pub(crate) transform_skip: bool,
    /// `cu_qp_delta_enabled_flag`.
    pub(crate) cu_qp_delta: bool,
    /// `Log2MinCuQpDeltaSize` (7-36).
    pub(crate) log2_min_qg_size: u32,
    /// `transquant_bypass_enabled_flag`.
    pub(crate) transquant_bypass: bool,
    /// `pps_cb_qp_offset + slice_cb_qp_offset`.
    pub(crate) cb_qp_offset: i32,
    /// `pps_cr_qp_offset + slice_cr_qp_offset`.
    pub(crate) cr_qp_offset: i32,
    /// `slice_type`.
    pub(crate) kind: SliceKind,
    /// `initType` (9-7).
    pub(crate) init_type: usize,
    /// `SliceQpY` (7-54).
    pub(crate) slice_qp: i32,
    /// `SliceAddrRs`.
    pub(crate) slice_addr_rs: u32,
    /// `MaxNumMergeCand`.
    pub(crate) max_merge_cand: u32,
    /// `num_ref_idx_l0_active_minus1` and `num_ref_idx_l1_active_minus1`.
    pub(crate) num_ref_idx_minus1: [u32; 2],
    /// `mvd_l1_zero_flag`.
    pub(crate) mvd_l1_zero: bool,
}

impl CodingParams {
    /// `QpBdOffsetY`, equal to `QpBdOffsetC`.
    pub(crate) const fn qp_bd_offset(&self) -> i32 {
        6 * (self.bit_depth as i32 - 8)
    }

    /// Reads the fields the parser needs. Call [`check_stream`] on the SPS and
    /// PPS first.
    ///
    /// # Errors
    ///
    /// [`HevcError::InvalidSyntax`] for a `SliceQpY` outside its range (7-54).
    pub(crate) fn from_headers(
        sps: &SeqParameterSet,
        pps: &PicParameterSet,
        slice: &SliceSegmentHeader,
    ) -> Result<Self, HevcError> {
        let kind = match slice.slice_type {
            SLICE_TYPE_I => SliceKind::I,
            SLICE_TYPE_P => SliceKind::P,
            _ => SliceKind::B,
        };
        let slice_qp = 26 + pps.init_qp_minus26 + slice.slice_qp_delta;
        let bit_depth = sps.bit_depth_y();
        let lowest_qp = -6 * (bit_depth as i32 - 8);
        if !(lowest_qp..=51).contains(&slice_qp) {
            return Err(HevcError::InvalidSyntax {
                element: "SliceQpY",
                value: i64::from(slice_qp),
                allowed: "-QpBdOffsetY to 51",
            });
        }
        let pcm = sps.pcm_enabled_flag.then(|| {
            let log2_min_size = sps.log2_min_pcm_luma_coding_block_size_minus3 + 3;
            PcmParams {
                log2_min_size,
                log2_max_size: log2_min_size + sps.log2_diff_max_min_pcm_luma_coding_block_size,
                bit_depth_luma: u32::from(sps.pcm_sample_bit_depth_luma_minus1) + 1,
                bit_depth_chroma: u32::from(sps.pcm_sample_bit_depth_chroma_minus1) + 1,
            }
        });
        let init_type = super::cabac::context::init_type(
            kind == SliceKind::I,
            kind == SliceKind::P,
            slice.cabac_init_flag,
        );
        Ok(Self {
            width: sps.pic_width_in_luma_samples,
            height: sps.pic_height_in_luma_samples,
            ctb_log2: sps.ctb_log2_size_y(),
            min_cb_log2: sps.min_cb_log2_size_y(),
            min_tb_log2: sps.log2_min_luma_transform_block_size_minus2 + 2,
            max_tb_log2: sps.max_tb_log2_size_y(),
            max_depth_inter: sps.max_transform_hierarchy_depth_inter,
            max_depth_intra: sps.max_transform_hierarchy_depth_intra,
            amp_enabled: sps.amp_enabled_flag,
            pcm,
            bit_depth,
            sign_data_hiding: pps.sign_data_hiding_enabled_flag,
            transform_skip: pps.transform_skip_enabled_flag,
            cu_qp_delta: pps.cu_qp_delta_enabled_flag,
            log2_min_qg_size: sps.ctb_log2_size_y() - pps.diff_cu_qp_delta_depth,
            transquant_bypass: pps.transquant_bypass_enabled_flag,
            cb_qp_offset: pps.pps_cb_qp_offset + slice.slice_cb_qp_offset,
            cr_qp_offset: pps.pps_cr_qp_offset + slice.slice_cr_qp_offset,
            kind,
            init_type,
            slice_qp,
            slice_addr_rs: slice.slice_addr_rs,
            max_merge_cand: slice.max_num_merge_cand(),
            num_ref_idx_minus1: [
                slice.num_ref_idx_l0_active_minus1,
                slice.num_ref_idx_l1_active_minus1,
            ],
            mvd_l1_zero: slice.mvd_l1_zero_flag,
        })
    }
}

/// Refuses a stream outside Main (A.3.2) and Main 10 (A.3.3) at 4:2:0. Also refuses
/// unequal luma and chroma bit depth, which Main 10 allows.
///
/// # Errors
///
/// - [`HevcError::Unsupported`] names the tool a stream uses.
/// - [`HevcError::PictureTooLarge`] for a picture above level 6.2.
/// - [`HevcError::Header`] for a PPS that does not fit its SPS, or for a range
///   extension tool. The parameter set parsers refuse screen content tools.
pub(crate) fn check_stream(sps: &SeqParameterSet, pps: &PicParameterSet) -> Result<(), HevcError> {
    if sps.chroma_format_idc != 1 {
        return Err(HevcError::Unsupported {
            field: "chroma_format_idc",
            value: sps.chroma_format_idc.into(),
            supported: "1, which is 4:2:0",
        });
    }
    if sps.bit_depth_luma_minus8 != sps.bit_depth_chroma_minus8 {
        return Err(HevcError::Unsupported {
            field: "bit_depth_chroma_minus8",
            value: sps.bit_depth_chroma_minus8.into(),
            supported: "a value equal to bit_depth_luma_minus8",
        });
    }
    if sps.bit_depth_luma_minus8 > 2 {
        return Err(HevcError::Unsupported {
            field: "bit_depth_luma_minus8",
            value: sps.bit_depth_luma_minus8.into(),
            supported: "0 to 2",
        });
    }
    let extensions = [
        (
            "sps_multilayer_extension_flag",
            sps.sps_multilayer_extension_flag,
        ),
        ("sps_3d_extension_flag", sps.sps_3d_extension_flag),
        (
            "pps_multilayer_extension_flag",
            pps.pps_multilayer_extension_flag,
        ),
        ("pps_3d_extension_flag", pps.pps_3d_extension_flag),
    ];
    if let Some((field, _)) = extensions.into_iter().find(|(_, set)| *set) {
        return Err(HevcError::Unsupported {
            field,
            value: 1,
            supported: "0, because this decoder takes nuh_layer_id 0 only",
        });
    }
    sps.sps_range_extension.refuse_tools()?;
    pps.pps_range_extension.refuse_tools()?;
    pps.check_with_sps(sps)?;
    if pps.tiles_enabled_flag && pps.entropy_coding_sync_enabled_flag {
        return Err(HevcError::Unsupported {
            field: "entropy_coding_sync_enabled_flag",
            value: 1,
            supported: "0 when tiles_enabled_flag is 1",
        });
    }
    let width = sps.pic_width_in_luma_samples;
    let height = sps.pic_height_in_luma_samples;
    if u64::from(width) * u64::from(height) > MAX_LUMA_PICTURE_SIZE
        || width > MAX_PICTURE_SIDE
        || height > MAX_PICTURE_SIDE
    {
        return Err(HevcError::PictureTooLarge { width, height });
    }
    Ok(())
}

/// Builds the address tables of the active SPS and PPS. Call [`check_stream`]
/// first.
///
/// # Errors
///
/// [`HevcError::Allocation`] when a table does not fit in memory.
pub(crate) fn picture_layout(
    sps: &SeqParameterSet,
    pps: &PicParameterSet,
) -> Result<PicLayout, HevcError> {
    PicLayout::new(&LayoutSizes {
        width: sps.pic_width_in_luma_samples,
        height: sps.pic_height_in_luma_samples,
        ctb_log2: sps.ctb_log2_size_y(),
        min_tb_log2: sps.log2_min_luma_transform_block_size_minus2 + 2,
        column_widths: &pps.column_widths(sps),
        row_heights: &pps.row_heights(sps),
    })
}

#[cfg(test)]
impl CodingParams {
    /// A 16x16 I slice with a 16x16 CTB and every optional tool off.
    pub(crate) const fn for_tests() -> Self {
        Self {
            width: 16,
            height: 16,
            ctb_log2: 4,
            min_cb_log2: 3,
            min_tb_log2: 2,
            max_tb_log2: 4,
            max_depth_inter: 0,
            max_depth_intra: 0,
            amp_enabled: false,
            pcm: None,
            bit_depth: 8,
            sign_data_hiding: false,
            transform_skip: false,
            cu_qp_delta: false,
            log2_min_qg_size: 4,
            transquant_bypass: false,
            cb_qp_offset: 0,
            cr_qp_offset: 0,
            kind: SliceKind::I,
            init_type: 0,
            slice_qp: 26,
            slice_addr_rs: 0,
            max_merge_cand: 5,
            num_ref_idx_minus1: [0, 0],
            mvd_l1_zero: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::hevc_pps::tests as pps_tests;
    use crate::bitstream::hevc_sps_full::tests as sps_tests;

    fn sps(spec: &sps_tests::Spec) -> SeqParameterSet {
        sps_tests::valid(spec)
    }

    fn pps() -> PicParameterSet {
        pps_tests::valid(&pps_tests::Spec::default())
    }

    #[test]
    fn main_and_main_10_pass() {
        assert_eq!(
            check_stream(&sps(&sps_tests::Spec::default()), &pps()),
            Ok(())
        );
        let main_10 = sps_tests::Spec {
            bit_depth_luma_minus8: 2,
            bit_depth_chroma_minus8: 2,
            ..sps_tests::Spec::default()
        };
        assert_eq!(check_stream(&sps(&main_10), &pps()), Ok(()));
    }

    #[test]
    fn other_chroma_formats_are_refused_by_name() {
        for chroma_format_idc in [0, 2, 3] {
            let spec = sps_tests::Spec {
                chroma_format_idc,
                ..sps_tests::Spec::default()
            };
            assert_eq!(
                check_stream(&sps(&spec), &pps()),
                Err(HevcError::Unsupported {
                    field: "chroma_format_idc",
                    value: chroma_format_idc.into(),
                    supported: "1, which is 4:2:0",
                })
            );
        }
    }

    #[test]
    fn unequal_depth_is_refused_by_name() {
        let spec = sps_tests::Spec {
            bit_depth_luma_minus8: 2,
            bit_depth_chroma_minus8: 0,
            ..sps_tests::Spec::default()
        };
        assert!(matches!(
            check_stream(&sps(&spec), &pps()),
            Err(HevcError::Unsupported {
                field: "bit_depth_chroma_minus8",
                value: 0,
                ..
            })
        ));
    }

    #[test]
    fn depth_above_10_is_refused_by_name() {
        let spec = sps_tests::Spec {
            bit_depth_luma_minus8: 4,
            bit_depth_chroma_minus8: 4,
            ..sps_tests::Spec::default()
        };
        assert!(matches!(
            check_stream(&sps(&spec), &pps()),
            Err(HevcError::Unsupported {
                field: "bit_depth_luma_minus8",
                value: 4,
                ..
            })
        ));
    }

    #[test]
    fn multilayer_is_refused_by_name() {
        let spec = sps_tests::Spec {
            multilayer: true,
            ..sps_tests::Spec::default()
        };
        assert!(matches!(
            check_stream(&sps(&spec), &pps()),
            Err(HevcError::Unsupported {
                field: "sps_multilayer_extension_flag",
                ..
            })
        ));
    }

    #[test]
    fn range_extension_tools_are_refused() {
        let mut flags = [false; 9];
        flags[3] = true;
        let spec = sps_tests::Spec {
            range_extension: Some(flags),
            ..sps_tests::Spec::default()
        };
        assert!(matches!(
            check_stream(&sps(&spec), &pps()),
            Err(HevcError::Header(_))
        ));
    }

    #[test]
    fn tiles_with_wavefront_are_refused() {
        let pps = pps_tests::valid(&pps_tests::Spec {
            wavefront: true,
            tiles: Some((1, 0, None)),
            ..pps_tests::Spec::default()
        });
        assert!(matches!(
            check_stream(&sps(&sps_tests::Spec::default()), &pps),
            Err(HevcError::Unsupported {
                field: "entropy_coding_sync_enabled_flag",
                ..
            })
        ));
    }

    #[test]
    fn a_picture_above_level_6_2_is_refused() {
        let wide = sps_tests::Spec {
            width: 16_896,
            height: 8,
            ..sps_tests::Spec::default()
        };
        assert_eq!(
            check_stream(&sps(&wide), &pps()),
            Err(HevcError::PictureTooLarge {
                width: 16_896,
                height: 8
            })
        );
        let large = sps_tests::Spec {
            width: 8_192,
            height: 4_360,
            ..sps_tests::Spec::default()
        };
        assert!(matches!(
            check_stream(&sps(&large), &pps()),
            Err(HevcError::PictureTooLarge { .. })
        ));
        let largest = sps_tests::Spec {
            width: 8_192,
            height: 4_352,
            ..sps_tests::Spec::default()
        };
        assert_eq!(check_stream(&sps(&largest), &pps()), Ok(()));
    }

    #[test]
    fn from_headers_flattens_the_fields() {
        let sps = sps(&sps_tests::Spec::default());
        let pps = pps_tests::valid(&pps_tests::Spec {
            qp_delta_depth: Some(2),
            cb_offset: -3,
            cr_offset: 4,
            init_qp: 2,
            ..pps_tests::Spec::default()
        });
        let slice = SliceSegmentHeader {
            slice_type: SLICE_TYPE_P,
            slice_qp_delta: -1,
            slice_cb_qp_offset: 1,
            slice_cr_qp_offset: -2,
            cabac_init_flag: true,
            five_minus_max_num_merge_cand: 3,
            num_ref_idx_l0_active_minus1: 2,
            ..SliceSegmentHeader::default()
        };
        let params = CodingParams::from_headers(&sps, &pps, &slice).unwrap();
        assert_eq!(params.width, 416);
        assert_eq!(params.ctb_log2, 6);
        assert_eq!(params.min_cb_log2, 3);
        assert_eq!(params.min_tb_log2, 2);
        assert_eq!(params.max_tb_log2, 5);
        assert_eq!(params.bit_depth, 8);
        assert_eq!(params.qp_bd_offset(), 0);
        assert_eq!(params.log2_min_qg_size, 4);
        assert!(params.cu_qp_delta);
        assert_eq!(params.cb_qp_offset, -2);
        assert_eq!(params.cr_qp_offset, 2);
        assert_eq!(params.kind, SliceKind::P);
        assert_eq!(params.init_type, 2);
        assert_eq!(params.slice_qp, 27);
        assert_eq!(params.max_merge_cand, 2);
        assert_eq!(params.num_ref_idx_minus1, [2, 0]);
    }

    #[test]
    fn a_slice_qp_out_of_range_is_refused() {
        let sps = sps(&sps_tests::Spec::default());
        let slice = SliceSegmentHeader {
            slice_type: SLICE_TYPE_I,
            slice_qp_delta: 40,
            ..SliceSegmentHeader::default()
        };
        assert!(matches!(
            CodingParams::from_headers(&sps, &pps(), &slice),
            Err(HevcError::InvalidSyntax {
                element: "SliceQpY",
                value: 66,
                ..
            })
        ));
    }

    #[test]
    fn picture_layout_reads_the_tiles() {
        let sps = sps(&sps_tests::Spec::default());
        let pps = pps_tests::valid(&pps_tests::Spec {
            tiles: Some((1, 0, None)),
            ..pps_tests::Spec::default()
        });
        let layout = picture_layout(&sps, &pps).unwrap();
        // 416x240 with a 64x64 CTB gives 7x4 CTBs, and columns of 3 and 4.
        assert_eq!(layout.ctb_count(), 28);
        assert_eq!(layout.tile_id(2), Some(0));
        assert_eq!(layout.tile_id(3), Some(1));
    }
}
