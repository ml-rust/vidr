//! `ctxInc` of the residual coding elements (9.3.4.2.3 to 9.3.4.2.7).

/// `ctxIdxMap` (Table 9-50). The table stops at entry 14, because position 15 of
/// a 4x4 block is never parsed. Entry 15 only keeps the array a power of two.
const CTX_IDX_MAP: [u8; 16] = [0, 1, 4, 5, 2, 3, 4, 5, 6, 6, 8, 8, 7, 7, 8, 8];

/// `sigCtx` of (9-45) to (9-48), by `prevCsbf` and `( yP << 2 ) + xP`.
const SIG_CTX_PATTERN: [[u8; 16]; 4] = [
    sig_pattern(0),
    sig_pattern(1),
    sig_pattern(2),
    sig_pattern(3),
];

const fn sig_pattern(prev_csbf: u8) -> [u8; 16] {
    let mut pattern = [0u8; 16];
    let mut index = 0;
    while index < 16 {
        let x_p = index & 3;
        let y_p = index >> 2;
        pattern[index] = match prev_csbf {
            0 => {
                if x_p + y_p == 0 {
                    2
                } else if x_p + y_p < 3 {
                    1
                } else {
                    0
                }
            }
            1 => {
                if y_p == 0 {
                    2
                } else if y_p == 1 {
                    1
                } else {
                    0
                }
            }
            2 => {
                if x_p == 0 {
                    2
                } else if x_p == 1 {
                    1
                } else {
                    0
                }
            }
            _ => 2,
        };
        index += 1;
    }
    pattern
}

/// `ctxOffset` and `ctxShift` of `last_sig_coeff_x_prefix` and
/// `last_sig_coeff_y_prefix` (9.3.4.2.3).
pub(super) const fn last_prefix_ctx(log2_size: u32, c_idx: usize) -> (u32, u32) {
    if c_idx == 0 {
        (
            3 * (log2_size - 2) + ((log2_size - 1) >> 2),
            (log2_size + 1) >> 2,
        )
    } else {
        (15, log2_size - 2)
    }
}

/// `ctxInc` of `coded_sub_block_flag` (9-38, 9-39), from the flags of the right
/// and lower neighbour sub-blocks.
pub(super) const fn coded_sub_block_ctx_inc(right: bool, below: bool, c_idx: usize) -> usize {
    let any = (right || below) as usize;
    if c_idx == 0 { any } else { 2 + any }
}

/// `ctxInc` of `sig_coeff_flag` inside one sub-block (9.3.4.2.5), without the
/// `transform_skip_context_enabled_flag` case, which is refused.
#[derive(Clone, Copy, Debug)]
pub(super) struct SigCtx {
    log2_size: u32,
    prev_csbf: usize,
    /// True for the sub-block at `( 0, 0 )`.
    first_sub_block: bool,
    /// 27 for chroma (9-55).
    component_offset: usize,
    /// The additions of (9-49) to (9-53).
    pattern_offset: usize,
}

impl SigCtx {
    /// The context of sub-block `( x_s, y_s )`, with `prevCsbf` from (9-43, 9-44).
    pub(super) const fn new(
        log2_size: u32,
        c_idx: usize,
        scan_idx: usize,
        sub_block: (u32, u32),
        prev_csbf: usize,
    ) -> Self {
        let first_sub_block = sub_block.0 + sub_block.1 == 0;
        let pattern_offset = if c_idx == 0 {
            let position = if first_sub_block { 0 } else { 3 };
            let size = if log2_size == 3 {
                if scan_idx == 0 { 9 } else { 15 }
            } else {
                21
            };
            position + size
        } else if log2_size == 3 {
            9
        } else {
            12
        };
        Self {
            log2_size,
            prev_csbf,
            first_sub_block,
            component_offset: if c_idx == 0 { 0 } else { 27 },
            pattern_offset,
        }
    }

    /// `ctxInc` at inner location `( x_p, y_p )`, each 0 to 3.
    #[inline]
    pub(super) const fn inc(&self, x_p: usize, y_p: usize) -> usize {
        let at = ((y_p & 3) << 2) + (x_p & 3);
        let sig_ctx = if self.log2_size == 2 {
            // In a 4x4 block, xC and yC equal xP and yP (9-41).
            CTX_IDX_MAP[at] as usize
        } else if self.first_sub_block && at == 0 {
            // (9-42).
            0
        } else {
            SIG_CTX_PATTERN[self.prev_csbf & 3][at] as usize + self.pattern_offset
        };
        sig_ctx + self.component_offset
    }
}

/// `ctxSet` at the first `coeff_abs_level_greater1_flag` of a sub-block (9-56 to
/// 9-58). `last_greater1_ctx` is the final `greater1Ctx` of the sub-block before,
/// updated with its last flag. The first sub-block of a transform block passes 1.
pub(super) const fn greater1_ctx_set(
    sub_block_index: usize,
    c_idx: usize,
    last_greater1_ctx: u32,
) -> usize {
    let base = if sub_block_index == 0 || c_idx > 0 {
        0
    } else {
        2
    };
    if last_greater1_ctx == 0 {
        base + 1
    } else {
        base
    }
}

/// `ctxInc` of `coeff_abs_level_greater1_flag` (9-59, 9-60).
pub(super) const fn greater1_ctx_inc(ctx_set: usize, greater1_ctx: u32, c_idx: usize) -> usize {
    let capped = if greater1_ctx > 3 { 3 } else { greater1_ctx };
    ctx_set * 4 + capped as usize + if c_idx > 0 { 16 } else { 0 }
}

/// `ctxInc` of `coeff_abs_level_greater2_flag` (9-61, 9-62).
pub(super) const fn greater2_ctx_inc(ctx_set: usize, c_idx: usize) -> usize {
    if c_idx > 0 { ctx_set + 4 } else { ctx_set }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Worked values of 9.3.4.2.3 at each size.
    #[test]
    fn last_prefix_contexts_follow_9_34() {
        assert_eq!(last_prefix_ctx(2, 0), (0, 0));
        assert_eq!(last_prefix_ctx(3, 0), (3, 1));
        assert_eq!(last_prefix_ctx(4, 0), (6, 1));
        assert_eq!(last_prefix_ctx(5, 0), (10, 1));
        assert_eq!(last_prefix_ctx(2, 1), (15, 0));
        assert_eq!(last_prefix_ctx(3, 2), (15, 1));
        assert_eq!(last_prefix_ctx(4, 1), (15, 2));
    }

    #[test]
    fn coded_sub_block_contexts_follow_9_38_and_9_39() {
        assert_eq!(coded_sub_block_ctx_inc(false, false, 0), 0);
        assert_eq!(coded_sub_block_ctx_inc(true, false, 0), 1);
        assert_eq!(coded_sub_block_ctx_inc(true, true, 0), 1);
        assert_eq!(coded_sub_block_ctx_inc(false, false, 1), 2);
        assert_eq!(coded_sub_block_ctx_inc(false, true, 2), 3);
    }

    /// Table 9-50 in a luma 4x4 block, and offset by 27 in chroma.
    #[test]
    fn a_4x4_block_reads_table_9_50() {
        let luma = SigCtx::new(2, 0, 0, (0, 0), 0);
        assert_eq!(luma.inc(0, 0), 0);
        assert_eq!(luma.inc(1, 0), 1);
        assert_eq!(luma.inc(2, 0), 4);
        assert_eq!(luma.inc(3, 0), 5);
        assert_eq!(luma.inc(0, 1), 2);
        assert_eq!(luma.inc(1, 1), 3);
        assert_eq!(luma.inc(2, 2), 8);
        assert_eq!(luma.inc(2, 3), 8);
        assert_eq!(luma.inc(0, 3), 7);
        let chroma = SigCtx::new(2, 1, 0, (0, 0), 0);
        assert_eq!(chroma.inc(2, 3), 35);
    }

    #[test]
    fn the_dc_position_has_its_own_context() {
        assert_eq!(SigCtx::new(3, 0, 0, (0, 0), 3).inc(0, 0), 0);
        assert_eq!(SigCtx::new(5, 0, 0, (0, 0), 1).inc(0, 0), 0);
        assert_eq!(SigCtx::new(4, 2, 0, (0, 0), 2).inc(0, 0), 27);
    }

    /// Hand-computed from (9-45) to (9-53).
    #[test]
    fn larger_blocks_add_the_size_and_position_offsets() {
        // Luma 8x8, diagonal, first sub-block, prevCsbf 0. xP + yP = 1: 1 + 9.
        assert_eq!(SigCtx::new(3, 0, 0, (0, 0), 0).inc(1, 0), 10);
        // Luma 8x8, horizontal scan: 1 + 15.
        assert_eq!(SigCtx::new(3, 0, 1, (0, 0), 0).inc(1, 0), 16);
        // Luma 8x8, diagonal, a later sub-block, prevCsbf 0, xP + yP = 0: 2 + 3 + 9.
        assert_eq!(SigCtx::new(3, 0, 0, (1, 0), 0).inc(0, 0), 14);
        // Luma 16x16, a later sub-block, prevCsbf 1, yP = 1: 1 + 3 + 21.
        assert_eq!(SigCtx::new(4, 0, 0, (2, 1), 1).inc(3, 1), 25);
        // Luma 32x32, first sub-block, prevCsbf 2, xP = 2: 0 + 21.
        assert_eq!(SigCtx::new(5, 0, 0, (0, 0), 2).inc(2, 1), 21);
        // Luma 32x32, a later sub-block, prevCsbf 3: 2 + 3 + 21, the last luma value.
        assert_eq!(SigCtx::new(5, 0, 0, (1, 1), 3).inc(3, 3), 26);
        // Chroma 8x8, prevCsbf 0, xP + yP = 3: 0 + 9 + 27.
        assert_eq!(SigCtx::new(3, 1, 0, (1, 0), 0).inc(1, 2), 36);
        // Chroma 16x16, prevCsbf 3: 2 + 12 + 27, the last context.
        assert_eq!(SigCtx::new(4, 2, 0, (1, 0), 3).inc(0, 3), 41);
    }

    /// Every value in the table, against (9-45) to (9-48) written out.
    #[test]
    fn the_pattern_table_equals_the_equations() {
        for y_p in 0..4usize {
            for x_p in 0..4usize {
                let at = (y_p << 2) + x_p;
                let by_sum = match x_p + y_p {
                    0 => 2,
                    1 | 2 => 1,
                    _ => 0,
                };
                let by_row = [2, 1, 0, 0][y_p];
                let by_column = [2, 1, 0, 0][x_p];
                assert_eq!(SIG_CTX_PATTERN[0][at], by_sum);
                assert_eq!(SIG_CTX_PATTERN[1][at], by_row);
                assert_eq!(SIG_CTX_PATTERN[2][at], by_column);
                assert_eq!(SIG_CTX_PATTERN[3][at], 2);
            }
        }
    }

    #[test]
    fn greater1_sets_follow_9_56_to_9_58() {
        assert_eq!(greater1_ctx_set(0, 0, 1), 0);
        assert_eq!(greater1_ctx_set(0, 0, 0), 1);
        assert_eq!(greater1_ctx_set(3, 0, 2), 2);
        assert_eq!(greater1_ctx_set(3, 0, 0), 3);
        assert_eq!(greater1_ctx_set(3, 1, 2), 0);
        assert_eq!(greater1_ctx_set(3, 2, 0), 1);
    }

    #[test]
    fn greater_flag_contexts_follow_9_59_to_9_62() {
        assert_eq!(greater1_ctx_inc(0, 1, 0), 1);
        assert_eq!(greater1_ctx_inc(3, 0, 0), 12);
        assert_eq!(greater1_ctx_inc(3, 5, 0), 15);
        assert_eq!(greater1_ctx_inc(1, 2, 1), 22);
        assert_eq!(greater1_ctx_inc(1, 9, 2), 23);
        assert_eq!(greater2_ctx_inc(3, 0), 3);
        assert_eq!(greater2_ctx_inc(1, 1), 5);
    }
}
