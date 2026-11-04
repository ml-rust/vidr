//! Context variables (9.3.2.2): their indices and initial states.
//!
//! The table holds one run of contexts per syntax element, in Table 9-4 order.
//! An initType with no entry in Table 9-4 for an element holds 154 there, a
//! value decoding never reads. The elements that need range or screen content
//! extensions have no contexts here. A stream with those tools is refused.

/// First context of `sao_merge_left_flag` and `sao_merge_up_flag`.
pub(crate) const SAO_MERGE: usize = 0;
/// First context of `sao_type_idx_luma` and `sao_type_idx_chroma`.
pub(crate) const SAO_TYPE_IDX: usize = 1;
/// First context of `split_cu_flag`.
pub(crate) const SPLIT_CU: usize = 2;
/// First context of `cu_transquant_bypass_flag`.
pub(crate) const CU_TRANSQUANT_BYPASS: usize = 5;
/// First context of `cu_skip_flag`.
pub(crate) const CU_SKIP: usize = 6;
/// First context of `pred_mode_flag`.
pub(crate) const PRED_MODE: usize = 9;
/// First context of `part_mode`.
pub(crate) const PART_MODE: usize = 10;
/// First context of `prev_intra_luma_pred_flag`.
pub(crate) const PREV_INTRA_LUMA_PRED: usize = 14;
/// First context of `intra_chroma_pred_mode`.
pub(crate) const INTRA_CHROMA_PRED_MODE: usize = 15;
/// First context of `rqt_root_cbf`.
pub(crate) const RQT_ROOT_CBF: usize = 16;
/// First context of `merge_flag`.
pub(crate) const MERGE_FLAG: usize = 17;
/// First context of `merge_idx`.
pub(crate) const MERGE_IDX: usize = 18;
/// First context of `inter_pred_idc`.
pub(crate) const INTER_PRED_IDC: usize = 19;
/// First context of `ref_idx_l0` and `ref_idx_l1`.
pub(crate) const REF_IDX: usize = 24;
/// First context of `mvp_l0_flag` and `mvp_l1_flag`.
pub(crate) const MVP_FLAG: usize = 26;
/// First context of `split_transform_flag`.
pub(crate) const SPLIT_TRANSFORM: usize = 27;
/// First context of `cbf_luma`.
pub(crate) const CBF_LUMA: usize = 30;
/// First context of `cbf_cb` and `cbf_cr`.
pub(crate) const CBF_CHROMA: usize = 32;
/// First context of `abs_mvd_greater0_flag`.
pub(crate) const ABS_MVD_GREATER0: usize = 36;
/// First context of `abs_mvd_greater1_flag`.
pub(crate) const ABS_MVD_GREATER1: usize = 37;
/// First context of `cu_qp_delta_abs`.
pub(crate) const CU_QP_DELTA_ABS: usize = 38;
/// First context of `transform_skip_flag`, luma then chroma.
pub(crate) const TRANSFORM_SKIP: usize = 40;
/// First context of `last_sig_coeff_x_prefix`.
pub(crate) const LAST_X_PREFIX: usize = 42;
/// First context of `last_sig_coeff_y_prefix`.
pub(crate) const LAST_Y_PREFIX: usize = 60;
/// First context of `coded_sub_block_flag`.
pub(crate) const CODED_SUB_BLOCK: usize = 78;
/// First context of `sig_coeff_flag`.
pub(crate) const SIG_COEFF: usize = 82;
/// First context of `coeff_abs_level_greater1_flag`.
pub(crate) const GREATER1: usize = 124;
/// First context of `coeff_abs_level_greater2_flag`.
pub(crate) const GREATER2: usize = 148;
/// Contexts in the table.
pub(crate) const NUM_CONTEXTS: usize = 154;
/// A power of two above [`NUM_CONTEXTS`]. An index masked by it is always in
/// bounds, so a context read has no panic path.
const TABLE_SIZE: usize = 256;

const INIT_TYPE_0: [u8; NUM_CONTEXTS] = [
    // SAO_MERGE
    153, // SAO_TYPE_IDX
    200, // SPLIT_CU
    139, 141, 157, // CU_TRANSQUANT_BYPASS
    154, // CU_SKIP
    154, 154, 154, // PRED_MODE
    154, // PART_MODE
    184, 154, 154, 154, // PREV_INTRA_LUMA_PRED
    184, // INTRA_CHROMA_PRED_MODE
    63,  // RQT_ROOT_CBF
    154, // MERGE_FLAG
    154, // MERGE_IDX
    154, // INTER_PRED_IDC
    154, 154, 154, 154, 154, // REF_IDX
    154, 154, // MVP_FLAG
    154, // SPLIT_TRANSFORM
    153, 138, 138, // CBF_LUMA
    111, 141, // CBF_CHROMA
    94, 138, 182, 154, // ABS_MVD_GREATER0
    154, // ABS_MVD_GREATER1
    154, // CU_QP_DELTA_ABS
    154, 154, // TRANSFORM_SKIP
    139, 139, // LAST_X_PREFIX
    110, 110, 124, 125, 140, 153, 125, 127, 140, 109, 111, 143, 127, 111, 79, 108, 123, 63,
    // LAST_Y_PREFIX
    110, 110, 124, 125, 140, 153, 125, 127, 140, 109, 111, 143, 127, 111, 79, 108, 123, 63,
    // CODED_SUB_BLOCK
    91, 171, 134, 141, // SIG_COEFF
    111, 111, 125, 110, 110, 94, 124, 108, 124, 107, 125, 141, 179, 153, 125, 107, 125, 141, 179,
    153, 125, 107, 125, 141, 179, 153, 125, 140, 139, 182, 182, 152, 136, 152, 136, 153, 136, 139,
    111, 136, 139, 111, // GREATER1
    140, 92, 137, 138, 140, 152, 138, 139, 153, 74, 149, 92, 139, 107, 122, 152, 140, 179, 166,
    182, 140, 227, 122, 197, // GREATER2
    138, 153, 136, 167, 152, 152,
];

const INIT_TYPE_1: [u8; NUM_CONTEXTS] = [
    // SAO_MERGE
    153, // SAO_TYPE_IDX
    185, // SPLIT_CU
    107, 139, 126, // CU_TRANSQUANT_BYPASS
    154, // CU_SKIP
    197, 185, 201, // PRED_MODE
    149, // PART_MODE
    154, 139, 154, 154, // PREV_INTRA_LUMA_PRED
    154, // INTRA_CHROMA_PRED_MODE
    152, // RQT_ROOT_CBF
    79,  // MERGE_FLAG
    110, // MERGE_IDX
    122, // INTER_PRED_IDC
    95, 79, 63, 31, 31, // REF_IDX
    153, 153, // MVP_FLAG
    168, // SPLIT_TRANSFORM
    124, 138, 94, // CBF_LUMA
    153, 111, // CBF_CHROMA
    149, 107, 167, 154, // ABS_MVD_GREATER0
    140, // ABS_MVD_GREATER1
    198, // CU_QP_DELTA_ABS
    154, 154, // TRANSFORM_SKIP
    139, 139, // LAST_X_PREFIX
    125, 110, 94, 110, 95, 79, 125, 111, 110, 78, 110, 111, 111, 95, 94, 108, 123, 108,
    // LAST_Y_PREFIX
    125, 110, 94, 110, 95, 79, 125, 111, 110, 78, 110, 111, 111, 95, 94, 108, 123, 108,
    // CODED_SUB_BLOCK
    121, 140, 61, 154, // SIG_COEFF
    155, 154, 139, 153, 139, 123, 123, 63, 153, 166, 183, 140, 136, 153, 154, 166, 183, 140, 136,
    153, 154, 166, 183, 140, 136, 153, 154, 170, 153, 123, 123, 107, 121, 107, 121, 167, 151, 183,
    140, 151, 183, 140, // GREATER1
    154, 196, 196, 167, 154, 152, 167, 182, 182, 134, 149, 136, 153, 121, 136, 137, 169, 194, 166,
    167, 154, 167, 137, 182, // GREATER2
    107, 167, 91, 122, 107, 167,
];

const INIT_TYPE_2: [u8; NUM_CONTEXTS] = [
    // SAO_MERGE
    153, // SAO_TYPE_IDX
    160, // SPLIT_CU
    107, 139, 126, // CU_TRANSQUANT_BYPASS
    154, // CU_SKIP
    197, 185, 201, // PRED_MODE
    134, // PART_MODE
    154, 139, 154, 154, // PREV_INTRA_LUMA_PRED
    183, // INTRA_CHROMA_PRED_MODE
    152, // RQT_ROOT_CBF
    79,  // MERGE_FLAG
    154, // MERGE_IDX
    137, // INTER_PRED_IDC
    95, 79, 63, 31, 31, // REF_IDX
    153, 153, // MVP_FLAG
    168, // SPLIT_TRANSFORM
    224, 167, 122, // CBF_LUMA
    153, 111, // CBF_CHROMA
    149, 92, 167, 154, // ABS_MVD_GREATER0
    169, // ABS_MVD_GREATER1
    198, // CU_QP_DELTA_ABS
    154, 154, // TRANSFORM_SKIP
    139, 139, // LAST_X_PREFIX
    125, 110, 124, 110, 95, 94, 125, 111, 111, 79, 125, 126, 111, 111, 79, 108, 123, 93,
    // LAST_Y_PREFIX
    125, 110, 124, 110, 95, 94, 125, 111, 111, 79, 125, 126, 111, 111, 79, 108, 123, 93,
    // CODED_SUB_BLOCK
    121, 140, 61, 154, // SIG_COEFF
    170, 154, 139, 153, 139, 123, 123, 63, 124, 166, 183, 140, 136, 153, 154, 166, 183, 140, 136,
    153, 154, 166, 183, 140, 136, 153, 154, 170, 153, 138, 138, 122, 121, 122, 121, 167, 151, 183,
    140, 151, 183, 140, // GREATER1
    154, 196, 167, 167, 154, 152, 167, 182, 182, 134, 149, 136, 153, 121, 136, 122, 169, 208, 166,
    167, 154, 152, 167, 182, // GREATER2
    107, 167, 91, 107, 107, 167,
];

/// The context variables of one slice segment, packed as
/// `pStateIdx << 1 | valMps`.
///
/// A copy is the storage process of 9.3.2.4, and assigning a copy back is the
/// synchronization process of 9.3.2.5.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ContextTable {
    states: [u8; TABLE_SIZE],
}

impl ContextTable {
    /// Initializes every context for `init_type` (9-7) at `slice_qp`.
    /// `init_type` above 2 selects initType 2.
    pub(crate) fn new(init_type: usize, slice_qp: i32) -> Self {
        let values = match init_type {
            0 => &INIT_TYPE_0,
            1 => &INIT_TYPE_1,
            _ => &INIT_TYPE_2,
        };
        let mut states = [0u8; TABLE_SIZE];
        for (state, &value) in states.iter_mut().zip(values) {
            *state = initial_state(value, slice_qp);
        }
        Self { states }
    }

    /// The packed state of context `index`.
    pub(crate) fn state_mut(&mut self, index: usize) -> &mut u8 {
        &mut self.states[index & (TABLE_SIZE - 1)]
    }
}

/// The initType of a slice (9-7).
pub(crate) const fn init_type(is_i: bool, is_p: bool, cabac_init_flag: bool) -> usize {
    if is_i {
        0
    } else if is_p == cabac_init_flag {
        2
    } else {
        1
    }
}

/// The packed state an `initValue` gives at `slice_qp` (9-4 to 9-6).
fn initial_state(init_value: u8, slice_qp: i32) -> u8 {
    let slope_idx = i32::from(init_value >> 4);
    let offset_idx = i32::from(init_value & 15);
    let m = slope_idx * 5 - 45;
    let n = (offset_idx << 3) - 16;
    let pre_ctx_state = (((m * slice_qp.clamp(0, 51)) >> 4) + n).clamp(1, 126);
    let val_mps = pre_ctx_state > 63;
    let p_state_idx = if val_mps {
        pre_ctx_state - 64
    } else {
        63 - pre_ctx_state
    };
    ((p_state_idx as u8) << 1) | u8::from(val_mps)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_type_follows_9_7() {
        assert_eq!(init_type(true, false, false), 0);
        assert_eq!(init_type(true, false, true), 0);
        assert_eq!(init_type(false, true, false), 1);
        assert_eq!(init_type(false, true, true), 2);
        assert_eq!(init_type(false, false, false), 2);
        assert_eq!(init_type(false, false, true), 1);
    }

    /// Hand-computed from 9-4 to 9-6.
    #[test]
    fn initial_state_follows_the_equations() {
        // 154: m = 0, n = 64, preCtxState 64, valMps 1, pStateIdx 0.
        assert_eq!(initial_state(154, 26), 1);
        // 139: slopeIdx 8, offsetIdx 11. m = -5, n = 72. At QP 26:
        // (-130 >> 4) + 72 = -9 + 72 = 63. valMps 0, pStateIdx 0.
        assert_eq!(initial_state(139, 26), 0);
        // 200: slopeIdx 12, offsetIdx 8. m = 15, n = 48. At QP 40:
        // (600 >> 4) + 48 = 85. valMps 1, pStateIdx 21.
        assert_eq!(initial_state(200, 40), (21 << 1) | 1);
        // 63: slopeIdx 3, offsetIdx 15. m = -30, n = 104. At QP 51:
        // (-1530 >> 4) + 104 = -96 + 104 = 8. valMps 0, pStateIdx 55.
        assert_eq!(initial_state(63, 51), 55 << 1);
    }

    #[test]
    fn slice_qp_clips_to_0_and_51() {
        assert_eq!(initial_state(200, 90), initial_state(200, 51));
        assert_eq!(initial_state(200, -12), initial_state(200, 0));
    }

    #[test]
    fn pre_ctx_state_clips_to_1_and_126() {
        // 240: m = 30, n = -16. At QP 0 the value clips up to 1: pStateIdx 62.
        assert_eq!(initial_state(240, 0), 62 << 1);
        // 255: m = 30, n = 104. At QP 51: 95 + 104 clips to 126: pStateIdx 62.
        assert_eq!(initial_state(255, 51), (62 << 1) | 1);
    }

    /// Spot checks of the table layout against Table 9-4.
    #[test]
    fn layout_follows_table_9_4() {
        let qp = 30;
        let get =
            |init_type: usize, index: usize| *ContextTable::new(init_type, qp).state_mut(index);
        // Table 9-7: 139, 141, 157 then 107, 139, 126.
        assert_eq!(get(0, SPLIT_CU + 2), initial_state(157, qp));
        assert_eq!(get(1, SPLIT_CU), initial_state(107, qp));
        assert_eq!(get(2, SPLIT_CU + 2), initial_state(126, qp));
        // Table 9-11: 184 at initType 0, 154, 139, 154, 154 after it.
        assert_eq!(get(0, PART_MODE), initial_state(184, qp));
        assert_eq!(get(1, PART_MODE + 1), initial_state(139, qp));
        assert_eq!(get(2, PART_MODE + 3), initial_state(154, qp));
        // Table 9-23: 140 and 198 at initType 1, 169 and 198 at initType 2.
        assert_eq!(get(1, ABS_MVD_GREATER0), initial_state(140, qp));
        assert_eq!(get(1, ABS_MVD_GREATER1), initial_state(198, qp));
        assert_eq!(get(2, ABS_MVD_GREATER0), initial_state(169, qp));
        assert_eq!(get(2, ABS_MVD_GREATER1), initial_state(198, qp));
        // Table 9-29: ctxIdx 41 ends initType 0, ctxIdx 42 opens initType 1.
        assert_eq!(get(0, SIG_COEFF + 41), initial_state(111, qp));
        assert_eq!(get(1, SIG_COEFF), initial_state(155, qp));
        assert_eq!(get(2, SIG_COEFF + 41), initial_state(140, qp));
        // Table 9-30: ctxIdx 71 ends initType 2.
        assert_eq!(get(2, GREATER1 + 23), initial_state(182, qp));
        // Table 9-31: ctxIdx 17 ends initType 2.
        assert_eq!(get(2, GREATER2 + 5), initial_state(167, qp));
    }

    #[test]
    fn a_masked_index_stays_in_bounds() {
        let mut table = ContextTable::new(0, 26);
        assert_eq!(
            *table.state_mut(TABLE_SIZE + 2),
            *ContextTable::new(0, 26).state_mut(2)
        );
    }

    #[test]
    fn the_runs_fill_the_table() {
        assert_eq!(GREATER2 + 6, NUM_CONTEXTS);
        const { assert!(NUM_CONTEXTS <= TABLE_SIZE) };
    }
}
