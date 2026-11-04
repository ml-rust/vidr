//! A `residual_coding()` writer for tests, written from 7.3.8.11 and 9.3.4.2.
//!
//! Each `ctxInc` here follows the specification's equations step by step. The
//! parser takes the same values from tables, so a table error breaks a round trip.

use crate::decode::cpu::hevc::cabac::context as ctx;
use crate::decode::cpu::hevc::scan::scan_order;

use super::test_writer::SyntaxWriter;

/// One transform block for [`write_residual`].
pub(super) struct BlockSpec<'a> {
    pub(super) log2_size: u32,
    pub(super) c_idx: usize,
    pub(super) scan_idx: usize,
    pub(super) sign_data_hiding: bool,
    pub(super) transquant_bypass: bool,
    /// `transform_skip_flag`, when the syntax holds it.
    pub(super) transform_skip: Option<bool>,
    /// `TransCoeffLevel`, row by row. At least one level is not zero.
    pub(super) levels: &'a [i32],
}

impl BlockSpec<'_> {
    fn level(&self, x: usize, y: usize) -> i32 {
        self.levels[(y << self.log2_size) + x]
    }
}

/// (9-34).
fn last_ctx_inc(bin: u32, log2_size: u32, c_idx: usize) -> usize {
    let (offset, shift) = if c_idx == 0 {
        (
            3 * (log2_size - 2) + ((log2_size - 1) >> 2),
            (log2_size + 1) >> 2,
        )
    } else {
        (15, log2_size - 2)
    };
    ((bin >> shift) + offset) as usize
}

/// `last_sig_coeff_x_prefix` and `last_sig_coeff_x_suffix` of a location, with the
/// suffix width: the inverse of (7-74) and (7-75).
fn last_codes(value: u32) -> (u32, u32, u32) {
    if value < 4 {
        return (value, 0, 0);
    }
    let mut prefix = 4;
    loop {
        let bits = (prefix >> 1) - 1;
        let base = (1 << bits) * (2 + (prefix & 1));
        if value < base + (1 << bits) {
            return (prefix, value - base, bits);
        }
        prefix += 1;
    }
}

fn write_last_prefix(
    writer: &mut SyntaxWriter,
    first_context: usize,
    prefix: u32,
    spec: &BlockSpec<'_>,
) {
    let c_max = (spec.log2_size << 1) - 1;
    for bin in 0..prefix {
        let inc = last_ctx_inc(bin, spec.log2_size, spec.c_idx);
        writer.ctx(first_context + inc, true);
    }
    if prefix < c_max {
        let inc = last_ctx_inc(prefix, spec.log2_size, spec.c_idx);
        writer.ctx(first_context + inc, false);
    }
}

/// `ctxInc` of `sig_coeff_flag`: (9-41) to (9-55) in the order the clause states.
fn sig_ctx_inc(spec: &BlockSpec<'_>, x_c: usize, y_c: usize, coded: &[[bool; 8]; 8]) -> usize {
    const CTX_IDX_MAP: [usize; 15] = [0, 1, 4, 5, 2, 3, 4, 5, 6, 6, 8, 8, 7, 7, 8];
    let sig_ctx = if spec.log2_size == 2 {
        CTX_IDX_MAP[(y_c << 2) + x_c]
    } else if x_c + y_c == 0 {
        0
    } else {
        let (x_s, y_s) = (x_c >> 2, y_c >> 2);
        let last = (1usize << (spec.log2_size - 2)) - 1;
        let mut prev_csbf = 0;
        if x_s < last {
            prev_csbf += usize::from(coded[x_s + 1][y_s]);
        }
        if y_s < last {
            prev_csbf += usize::from(coded[x_s][y_s + 1]) << 1;
        }
        let (x_p, y_p) = (x_c & 3, y_c & 3);
        let mut value = match prev_csbf {
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
        if spec.c_idx == 0 {
            if x_s + y_s > 0 {
                value += 3;
            }
            value += if spec.log2_size == 3 {
                if spec.scan_idx == 0 { 9 } else { 15 }
            } else {
                21
            };
        } else {
            value += if spec.log2_size == 3 { 9 } else { 12 };
        }
        value
    };
    if spec.c_idx == 0 {
        sig_ctx
    } else {
        27 + sig_ctx
    }
}

/// `coeff_abs_level_remaining` for one `cRiceParam` (9.3.3.11).
fn write_remaining(writer: &mut SyntaxWriter, value: u32, rice: u32) {
    let c_max = 4 << rice;
    if value < c_max {
        let prefix = value >> rice;
        for _ in 0..prefix {
            writer.bypass(true);
        }
        writer.bypass(false);
        writer.bits(value - (prefix << rice), rice);
    } else {
        for _ in 0..4 {
            writer.bypass(true);
        }
        writer.exp_golomb(value - c_max, rice + 1);
    }
}

/// Writes a block whose one level is 1 at DC, with no transform skip flag.
pub(super) fn write_dc(writer: &mut SyntaxWriter, log2_size: u32, c_idx: usize) {
    let mut levels = vec![0; 1 << (2 * log2_size)];
    levels[0] = 1;
    write_residual(
        writer,
        &BlockSpec {
            log2_size,
            c_idx,
            scan_idx: 0,
            sign_data_hiding: false,
            transquant_bypass: false,
            transform_skip: None,
            levels: &levels,
        },
    );
}

/// True when sign data hiding applies to these significant positions.
pub(super) const fn sign_hidden(last_position: usize, first_position: usize) -> bool {
    last_position - first_position > 3
}

/// Writes one `residual_coding()`.
pub(super) fn write_residual(writer: &mut SyntaxWriter, spec: &BlockSpec<'_>) {
    let sub_scan = scan_order(spec.log2_size - 2, spec.scan_idx);
    let position_scan = scan_order(2, spec.scan_idx);
    let location = |sub: usize, position: usize| {
        let [x_s, y_s] = sub_scan[sub];
        let [x_p, y_p] = position_scan[position];
        (
            (usize::from(x_s) << 2) + usize::from(x_p),
            (usize::from(y_s) << 2) + usize::from(y_p),
        )
    };
    let mut last = None;
    'search: for sub in (0..sub_scan.len()).rev() {
        for position in (0..16).rev() {
            let (x, y) = location(sub, position);
            if spec.level(x, y) != 0 {
                last = Some((sub, position, x, y));
                break 'search;
            }
        }
    }
    let Some((last_sub, last_position, x_last, y_last)) = last else {
        panic!("a coded block holds a level that is not zero");
    };
    if let Some(flag) = spec.transform_skip {
        writer.ctx(ctx::TRANSFORM_SKIP + usize::from(spec.c_idx > 0), flag);
    }
    // (7-78) swaps the coded values at scanIdx 2.
    let (x_coded, y_coded) = if spec.scan_idx == 2 {
        (y_last, x_last)
    } else {
        (x_last, y_last)
    };
    let (x_prefix, x_suffix, x_bits) = last_codes(x_coded as u32);
    let (y_prefix, y_suffix, y_bits) = last_codes(y_coded as u32);
    write_last_prefix(writer, ctx::LAST_X_PREFIX, x_prefix, spec);
    write_last_prefix(writer, ctx::LAST_Y_PREFIX, y_prefix, spec);
    writer.bits(x_suffix, x_bits);
    writer.bits(y_suffix, y_bits);

    let mut coded = [[false; 8]; 8];
    let last_grid = (1usize << (spec.log2_size - 2)) - 1;
    // lastGreater1Ctx of 9.3.4.2.6, updated with the last flag of each sub-block.
    let mut last_greater1_ctx = 1u32;
    for index in (0..=last_sub).rev() {
        let [x_s, y_s] = sub_scan[index];
        let (x_s, y_s) = (usize::from(x_s), usize::from(y_s));
        let any_level = (0..16).any(|position| {
            let (x, y) = location(index, position);
            spec.level(x, y) != 0
        });
        let mut sub_block_coded = true;
        let mut infer_dc = false;
        if index < last_sub && index > 0 {
            let mut csbf_ctx = 0;
            if x_s < last_grid {
                csbf_ctx += usize::from(coded[x_s + 1][y_s]);
            }
            if y_s < last_grid {
                csbf_ctx += usize::from(coded[x_s][y_s + 1]);
            }
            let inc = csbf_ctx.min(1) + if spec.c_idx == 0 { 0 } else { 2 };
            writer.ctx(ctx::CODED_SUB_BLOCK + inc, any_level);
            sub_block_coded = any_level;
            infer_dc = true;
        }
        coded[x_s][y_s] = sub_block_coded;
        let mut significant = Vec::new();
        let first_parsed = if index == last_sub {
            significant.push(last_position);
            last_position
        } else {
            16
        };
        for position in (0..first_parsed).rev() {
            let (x, y) = location(index, position);
            let level = spec.level(x, y);
            if sub_block_coded && (position > 0 || !infer_dc) {
                let inc = sig_ctx_inc(spec, x, y, &coded);
                writer.ctx(ctx::SIG_COEFF + inc, level != 0);
                if level != 0 {
                    infer_dc = false;
                    significant.push(position);
                }
            } else if sub_block_coded {
                assert_ne!(level, 0, "an inferred DC level is not zero");
                significant.push(position);
            }
        }
        if significant.is_empty() {
            continue;
        }
        let abs_at = |position: usize| {
            let (x, y) = location(index, position);
            spec.level(x, y).unsigned_abs()
        };
        // (9-56) to (9-58).
        let mut ctx_set = if index == 0 || spec.c_idx > 0 { 0 } else { 2 };
        if last_greater1_ctx == 0 {
            ctx_set += 1;
        }
        let mut greater1_ctx = 1u32;
        let mut first_greater1 = None;
        for (order, &position) in significant.iter().take(8).enumerate() {
            let flag = abs_at(position) > 1;
            let inc =
                ctx_set * 4 + greater1_ctx.min(3) as usize + if spec.c_idx > 0 { 16 } else { 0 };
            writer.ctx(ctx::GREATER1 + inc, flag);
            if flag && first_greater1.is_none() {
                first_greater1 = Some(order);
            }
            if greater1_ctx > 0 {
                if flag {
                    greater1_ctx = 0;
                } else {
                    greater1_ctx += 1;
                }
            }
        }
        last_greater1_ctx = greater1_ctx;
        if let Some(order) = first_greater1 {
            let inc = ctx_set + if spec.c_idx > 0 { 4 } else { 0 };
            writer.ctx(ctx::GREATER2 + inc, abs_at(significant[order]) > 2);
        }
        let last_in_sub_block = significant[0];
        let first_in_sub_block = significant[significant.len() - 1];
        let hidden = spec.sign_data_hiding
            && !spec.transquant_bypass
            && sign_hidden(last_in_sub_block, first_in_sub_block);
        for &position in &significant {
            let sign_is_hidden = hidden && position == first_in_sub_block;
            if !sign_is_hidden {
                let (x, y) = location(index, position);
                writer.bypass(spec.level(x, y) < 0);
            }
        }
        let mut rice = 0u32;
        for (order, &position) in significant.iter().enumerate() {
            let abs = abs_at(position);
            let mut base_level = 1;
            if order < 8 && abs > 1 {
                base_level += 1;
            }
            if first_greater1 == Some(order) && abs > 2 {
                base_level += 1;
            }
            let threshold = if order >= 8 {
                1
            } else if first_greater1 == Some(order) {
                3
            } else {
                2
            };
            if base_level == threshold {
                write_remaining(writer, abs - base_level, rice);
                if abs > 3 * (1 << rice) {
                    rice = (rice + 1).min(4);
                }
            } else {
                assert_eq!(abs, base_level);
            }
        }
    }
}
