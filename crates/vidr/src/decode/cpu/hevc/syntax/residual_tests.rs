//! `residual_coding()` through encoded coefficients.

use crate::decode::cpu::hevc::cabac::context as ctx;
use crate::decode::cpu::hevc::error::HevcError;
use crate::decode::cpu::hevc::params::CodingParams;
use crate::decode::cpu::hevc::scan::{SCAN_DIAGONAL, SCAN_HORIZONTAL, SCAN_VERTICAL, scan_order};

use super::records::TbRecord;
use super::residual::TbArgs;
use super::test_residual_writer::{BlockSpec, sign_hidden, write_residual};
use super::test_writer::{Harness, SyntaxWriter, check_end};

struct Xorshift(u64);

impl Xorshift {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// The options a block test varies.
#[derive(Clone, Copy)]
struct Options {
    log2_size: u32,
    c_idx: usize,
    scan_idx: usize,
    sign_data_hiding: bool,
    transquant_bypass: bool,
    transform_skip: Option<bool>,
}

const fn options(log2_size: u32, c_idx: usize, scan_idx: usize) -> Options {
    Options {
        log2_size,
        c_idx,
        scan_idx,
        sign_data_hiding: false,
        transquant_bypass: false,
        transform_skip: None,
    }
}

/// Writes one block, parses it, and returns its dense levels and its record.
fn round_trip(options: Options, levels: &[i32]) -> (Vec<i32>, TbRecord) {
    let params = CodingParams {
        sign_data_hiding: options.sign_data_hiding,
        transform_skip: options.transform_skip.is_some(),
        ..CodingParams::for_tests()
    };
    let mut writer = SyntaxWriter::new(&params);
    write_residual(
        &mut writer,
        &BlockSpec {
            log2_size: options.log2_size,
            c_idx: options.c_idx,
            scan_idx: options.scan_idx,
            sign_data_hiding: options.sign_data_hiding,
            transquant_bypass: options.transquant_bypass,
            transform_skip: options.transform_skip,
            levels,
        },
    );
    let (bytes, contexts) = writer.finish();
    let mut harness = Harness::new(params);
    let mut parser = harness.parser(&bytes);
    parser
        .residual_coding(&TbArgs {
            x0: 0,
            y0: 0,
            log2_size: options.log2_size,
            c_idx: options.c_idx,
            scan_idx: options.scan_idx,
            transquant_bypass: options.transquant_bypass,
        })
        .unwrap();
    check_end(&mut parser, &contexts);
    let mut dense = vec![0; levels.len()];
    for coeff in &parser.records.coeffs {
        let at = (usize::from(coeff.y) << options.log2_size) + usize::from(coeff.x);
        assert_eq!(dense[at], 0, "a location appears twice");
        dense[at] = i32::from(coeff.level);
    }
    (dense, parser.records.tbs[0])
}

fn assert_round_trip(options: Options, levels: &[i32]) {
    let (dense, record) = round_trip(options, levels);
    assert_eq!(dense, levels);
    assert_eq!(record.log2_size, options.log2_size);
    assert_eq!(record.c_idx, options.c_idx);
    let coded = levels.iter().filter(|&&level| level != 0).count();
    assert_eq!(record.coeff_count, coded);
}

fn single_level(log2_size: u32, x: usize, y: usize, level: i32) -> Vec<i32> {
    let mut levels = vec![0; 1 << (2 * log2_size)];
    levels[(y << log2_size) + x] = level;
    levels
}

#[test]
fn a_dc_level_round_trips_at_every_size() {
    for log2_size in 2..=5 {
        assert_round_trip(options(log2_size, 0, 0), &single_level(log2_size, 0, 0, 1));
        assert_round_trip(options(log2_size, 0, 0), &single_level(log2_size, 0, 0, -1));
    }
    for log2_size in 2..=4 {
        assert_round_trip(options(log2_size, 1, 0), &single_level(log2_size, 0, 0, 3));
        assert_round_trip(options(log2_size, 2, 0), &single_level(log2_size, 0, 0, -3));
    }
}

/// Every location of every size, so every prefix and suffix of 7-74 to 7-77 runs.
#[test]
fn every_last_position_round_trips() {
    for log2_size in 2..=5 {
        let size = 1usize << log2_size;
        for y in 0..size {
            for x in 0..size {
                assert_round_trip(options(log2_size, 0, 0), &single_level(log2_size, x, y, 2));
            }
        }
    }
}

/// (7-78) swaps the coded values at the vertical scan. The horizontal and vertical
/// scans apply at 4x4 and 8x8 (7.4.9.11).
#[test]
fn directional_scans_round_trip_every_position() {
    for log2_size in 2..=3 {
        let size = 1usize << log2_size;
        for scan_idx in [SCAN_HORIZONTAL, SCAN_VERTICAL] {
            for y in 0..size {
                for x in 0..size {
                    assert_round_trip(
                        options(log2_size, 0, scan_idx),
                        &single_level(log2_size, x, y, -2),
                    );
                }
            }
        }
    }
}

/// The ninth significant coefficient of a sub-block has no greater1 flag. Its level
/// parses as `coeff_abs_level_remaining` from a base of 1.
#[test]
fn more_than_eight_coefficients_use_the_remaining_code() {
    let levels = vec![1, 2, 3, 1, 4, 1, 1, 9, 2, 1, -1, -2, 3, -4, 5, 6];
    assert_round_trip(options(2, 0, 0), &levels);
    assert_round_trip(options(2, 1, 0), &levels);
    assert_round_trip(options(2, 0, SCAN_HORIZONTAL), &levels);
    assert_round_trip(options(2, 0, SCAN_VERTICAL), &levels);
}

/// Large levels walk `cRiceParam` up to 4, and the Exp-Golomb suffix runs up to
/// the coefficient limits.
#[test]
fn escape_codes_reach_the_coefficient_limits() {
    let levels = vec![
        32_767, -32_768, 15, -16, 3, 30_000, 5, -7, 100, 1000, -2000, 4, 1, 0, 8, 12,
    ];
    assert_round_trip(options(2, 0, 0), &levels);
    let mut large = vec![0; 64];
    large[0] = -32_768;
    large[63] = 32_767;
    large[36] = 12_345;
    assert_round_trip(options(3, 0, 0), &large);
}

/// `cRiceParam` advances from each level that takes the code (9-24).
#[test]
fn the_rice_parameter_grows_across_a_sub_block() {
    // Levels in parsing order grow, so each one raises the parameter.
    let scan = scan_order(2, SCAN_DIAGONAL);
    let mut levels = vec![0; 16];
    let values = [4, 5, 9, 17, 40, 80, 200, 400, 800, 1600, 3200, 6400];
    for (&[x, y], &value) in scan.iter().rev().zip(&values) {
        levels[usize::from(y) * 4 + usize::from(x)] = value;
    }
    assert_round_trip(options(2, 0, 0), &levels);
}

/// Sub-blocks of a large block hold zero, one, or several levels. So
/// `coded_sub_block_flag`, `prevCsbf`, and the DC inference all vary.
#[test]
fn sparse_sub_blocks_round_trip() {
    let mut levels = vec![0; 256];
    // Sub-block ( 3, 3 ) holds the last level alone.
    levels[255] = 1;
    // Sub-block ( 2, 3 ) has only its DC level, which is inferred.
    levels[(12 << 4) + 8] = -3;
    // Sub-block ( 3, 1 ) has one level off DC, so the DC flag parses as 0.
    levels[(5 << 4) + 13] = 2;
    // Sub-block ( 0, 1 ) has several levels.
    levels[(4 << 4) + 1] = 1;
    levels[(6 << 4) + 2] = -1;
    levels[(7 << 4) + 3] = 7;
    // Sub-block ( 0, 0 ) holds levels with its flag inferred.
    levels[0] = 20;
    assert_round_trip(options(4, 0, 0), &levels);
    assert_round_trip(options(4, 1, 0), &levels);
}

#[test]
fn sub_block_zero_can_hold_no_level() {
    let mut levels = vec![0; 64];
    levels[(4 << 3) + 4] = 5;
    assert_round_trip(options(3, 0, 0), &levels);
}

/// When sign data hiding applies, the parity of the sum gives the sign of the
/// lowest-scan level (7.3.8.11).
#[test]
fn hidden_signs_follow_the_parity() {
    let hiding = Options {
        sign_data_hiding: true,
        ..options(2, 0, 0)
    };
    let scan = scan_order(2, SCAN_DIAGONAL);
    let at = |position: usize| usize::from(scan[position][1]) * 4 + usize::from(scan[position][0]);
    // Positions 5 and 0: distance 5 is above 3. Sum 4 is even: the level at
    // position 0 is positive.
    let mut levels = vec![0; 16];
    levels[at(5)] = -3;
    levels[at(0)] = 1;
    assert_round_trip(hiding, &levels);
    // Sum 3 is odd: the level at position 0 is negative.
    levels[at(5)] = 2;
    levels[at(0)] = -1;
    assert_round_trip(hiding, &levels);
    // Distance 3 hides nothing, and both signs are coded.
    let mut levels = vec![0; 16];
    levels[at(3)] = -2;
    levels[at(0)] = -2;
    assert_round_trip(hiding, &levels);
    // Distance 4 hides. The sum 5 is odd, so the hidden sign is negative.
    let mut levels = vec![0; 16];
    levels[at(4)] = -3;
    levels[at(0)] = -2;
    assert_round_trip(hiding, &levels);
}

#[test]
fn transquant_bypass_keeps_every_sign() {
    let bypass = Options {
        sign_data_hiding: true,
        transquant_bypass: true,
        ..options(2, 0, 0)
    };
    let scan = scan_order(2, SCAN_DIAGONAL);
    let at = |position: usize| usize::from(scan[position][1]) * 4 + usize::from(scan[position][0]);
    let mut levels = vec![0; 16];
    levels[at(5)] = 3;
    levels[at(0)] = -1;
    // Hiding on this block gives sum 4, which is even, and a positive second level.
    assert_round_trip(bypass, &levels);
}

#[test]
fn the_transform_skip_flag_parses_at_4x4_only() {
    for (flag, c_idx) in [(true, 0), (false, 0), (true, 1), (false, 2)] {
        let block = Options {
            transform_skip: Some(flag),
            ..options(2, c_idx, 0)
        };
        let levels = single_level(2, 1, 2, 3);
        let (dense, record) = round_trip(block, &levels);
        assert_eq!(dense, levels);
        assert_eq!(record.transform_skip, flag);
    }
    // At 8x8 with the tool on, the flag is absent.
    let params = CodingParams {
        transform_skip: true,
        ..CodingParams::for_tests()
    };
    let mut writer = SyntaxWriter::new(&params);
    write_residual(
        &mut writer,
        &BlockSpec {
            log2_size: 3,
            c_idx: 0,
            scan_idx: 0,
            sign_data_hiding: false,
            transquant_bypass: false,
            transform_skip: None,
            levels: &single_level(3, 0, 0, 1),
        },
    );
    let (bytes, contexts) = writer.finish();
    let mut harness = Harness::new(params);
    let mut parser = harness.parser(&bytes);
    parser
        .residual_coding(&TbArgs {
            x0: 0,
            y0: 0,
            log2_size: 3,
            c_idx: 0,
            scan_idx: 0,
            transquant_bypass: false,
        })
        .unwrap();
    check_end(&mut parser, &contexts);
    assert!(!parser.records.tbs[0].transform_skip);
}

/// `cu_transquant_bypass_flag` removes the flag (7.3.8.11).
#[test]
fn transquant_bypass_has_no_transform_skip_flag() {
    let params = CodingParams {
        transform_skip: true,
        ..CodingParams::for_tests()
    };
    let mut writer = SyntaxWriter::new(&params);
    write_residual(
        &mut writer,
        &BlockSpec {
            log2_size: 2,
            c_idx: 0,
            scan_idx: 0,
            sign_data_hiding: false,
            transquant_bypass: true,
            transform_skip: None,
            levels: &single_level(2, 0, 0, 1),
        },
    );
    let (bytes, contexts) = writer.finish();
    let mut harness = Harness::new(params);
    let mut parser = harness.parser(&bytes);
    parser
        .residual_coding(&TbArgs {
            x0: 0,
            y0: 0,
            log2_size: 2,
            c_idx: 0,
            scan_idx: 0,
            transquant_bypass: true,
        })
        .unwrap();
    check_end(&mut parser, &contexts);
    assert!(!parser.records.tbs[0].transform_skip);
}

/// Sets the sign of the lowest-scan level in each sub-block with a hidden sign. The
/// block then matches what the parity gives (7.3.8.11).
fn fix_hidden_signs(options: Options, levels: &mut [i32]) {
    let sub_scan = scan_order(options.log2_size - 2, options.scan_idx);
    let position_scan = scan_order(2, options.scan_idx);
    for &[x_s, y_s] in sub_scan {
        let significant: Vec<(usize, usize)> = position_scan
            .iter()
            .enumerate()
            .map(|(position, &[x_p, y_p])| {
                let x = (usize::from(x_s) << 2) + usize::from(x_p);
                let y = (usize::from(y_s) << 2) + usize::from(y_p);
                (position, (y << options.log2_size) + x)
            })
            .filter(|&(_, at)| levels[at] != 0)
            .collect();
        let (Some(&(first_position, first_at)), Some(&(last_position, _))) =
            (significant.first(), significant.last())
        else {
            continue;
        };
        if !sign_hidden(last_position, first_position) {
            continue;
        }
        let sum: u64 = significant
            .iter()
            .map(|&(_, at)| u64::from(levels[at].unsigned_abs()))
            .sum();
        let magnitude = levels[first_at].abs();
        levels[first_at] = if sum & 1 == 1 { -magnitude } else { magnitude };
    }
}

/// A random block at one density, with some large levels.
fn random_levels(rng: &mut Xorshift, log2_size: u32) -> Vec<i32> {
    let count = 1usize << (2 * log2_size);
    let density = 1 + rng.next() % 16;
    let mut levels: Vec<i32> = (0..count)
        .map(|_| {
            let value = rng.next();
            if value % 16 >= density {
                return 0;
            }
            let magnitude = match (value >> 8) % 16 {
                0..=9 => 1 + ((value >> 16) % 2) as i32,
                10..=13 => 1 + ((value >> 16) % 20) as i32,
                14 => 1 + ((value >> 16) % 1000) as i32,
                _ => 1 + ((value >> 16) % 32_767) as i32,
            };
            if (value >> 40) & 1 == 1 {
                -magnitude
            } else {
                magnitude
            }
        })
        .collect();
    if levels.iter().all(|&level| level == 0) {
        levels[(rng.next() as usize) % count] = 1;
    }
    levels
}

#[test]
fn random_blocks_round_trip() {
    let mut rng = Xorshift(0x2545_f491_4f6c_dd1d);
    for log2_size in 2..=5u32 {
        let scans: &[usize] = if log2_size <= 3 {
            &[SCAN_DIAGONAL, SCAN_HORIZONTAL, SCAN_VERTICAL]
        } else {
            &[SCAN_DIAGONAL]
        };
        for &scan_idx in scans {
            for c_idx in 0..3 {
                if c_idx > 0 && log2_size == 5 {
                    continue;
                }
                for trial in 0..30 {
                    let block = Options {
                        sign_data_hiding: trial & 1 == 1,
                        transquant_bypass: trial & 7 == 3,
                        ..options(log2_size, c_idx, scan_idx)
                    };
                    let mut levels = random_levels(&mut rng, log2_size);
                    if block.sign_data_hiding && !block.transquant_bypass {
                        fix_hidden_signs(block, &mut levels);
                    }
                    assert_round_trip(block, &levels);
                }
            }
        }
    }
}

/// Bins for a DC level that reaches `coeff_abs_level_remaining` with a base of 3:
/// last location 0, `greater1` 1, `greater2` 1, and a positive sign.
fn dc_level_up_to_remaining(writer: &mut SyntaxWriter) {
    writer.ctx(ctx::LAST_X_PREFIX, false);
    writer.ctx(ctx::LAST_Y_PREFIX, false);
    writer.ctx(ctx::GREATER1 + 1, true);
    writer.ctx(ctx::GREATER2, true);
    writer.bypass(false);
}

fn parse_error(writer: SyntaxWriter) -> HevcError {
    let params = CodingParams::for_tests();
    let bytes = writer.into_bytes();
    let mut harness = Harness::new(params);
    let mut parser = harness.parser(&bytes);
    let result = parser.residual_coding(&TbArgs {
        x0: 0,
        y0: 0,
        log2_size: 2,
        c_idx: 0,
        scan_idx: 0,
        transquant_bypass: false,
    });
    match result {
        Ok(()) => panic!("this block is invalid, and the parser accepts it"),
        Err(error) => error,
    }
}

/// `TransCoeffLevel` stops at `CoeffMaxY` (7.4.9.11).
#[test]
fn a_level_above_the_limit_is_refused() {
    let params = CodingParams::for_tests();
    let mut writer = SyntaxWriter::new(&params);
    dc_level_up_to_remaining(&mut writer);
    // The level is 3 + remaining: 32768, one above the limit. rice is 0, so the
    // code is 1111 and then EG1 of 32765 - 4.
    for _ in 0..4 {
        writer.bypass(true);
    }
    writer.exp_golomb(32_765 - 4, 1);
    assert_eq!(
        parse_error(writer),
        HevcError::InvalidSyntax {
            element: "TransCoeffLevel",
            value: 32_768,
            allowed: "-32768 to 32767",
        }
    );
}

/// An Exp-Golomb prefix past any legal value stops with a named error.
#[test]
fn an_endless_exp_golomb_prefix_is_refused() {
    let params = CodingParams::for_tests();
    let mut writer = SyntaxWriter::new(&params);
    dc_level_up_to_remaining(&mut writer);
    for _ in 0..40 {
        writer.bypass(true);
    }
    assert!(matches!(
        parse_error(writer),
        HevcError::InvalidSyntax {
            element: "coeff_abs_level_remaining",
            ..
        }
    ));
}

/// The contexts of an all-zero start: `LAST_X_PREFIX` is on `ctxInc` 0 at 4x4.
#[test]
fn a_dc_level_at_4x4_spends_the_expected_contexts() {
    let params = CodingParams::for_tests();
    let mut writer = SyntaxWriter::new(&params);
    writer.ctx(ctx::LAST_X_PREFIX, false);
    writer.ctx(ctx::LAST_Y_PREFIX, false);
    writer.ctx(ctx::GREATER1 + 1, false);
    writer.bypass(true);
    let (bytes, contexts) = writer.finish();
    let mut harness = Harness::new(params);
    let mut parser = harness.parser(&bytes);
    parser
        .residual_coding(&TbArgs {
            x0: 4,
            y0: 8,
            log2_size: 2,
            c_idx: 0,
            scan_idx: 0,
            transquant_bypass: false,
        })
        .unwrap();
    check_end(&mut parser, &contexts);
    assert_eq!(parser.records.coeffs.len(), 1);
    assert_eq!(parser.records.coeffs[0].level, -1);
    assert_eq!((parser.records.tbs[0].x, parser.records.tbs[0].y), (4, 8));
}
