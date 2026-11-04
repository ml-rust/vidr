//! The in-loop filters over bands of rows against the whole-picture form, on random
//! pictures. Any difference names a band split that breaks the order of 8.7.2.

use crate::decode::cpu::dsp::test_rng::Rng;
use crate::decode::cpu::hevc::layout::{LayoutSizes, PicLayout};
use crate::decode::cpu::hevc::params::CodingParams;
use crate::decode::cpu::hevc::picture::Picture;
use crate::decode::cpu::hevc::syntax::maps::{CtbFilter, ParseMaps};
use crate::decode::cpu::hevc::syntax::sao::{SaoCtb, SaoKind, SaoPlane};

use super::deblock::{DeblockContext, deblock_picture, deblock_picture_in_rows};
use super::edges::{EdgeMap, EdgeRun};
use super::sao::{SaoContext, sao_picture, sao_picture_in_rows};

struct Scene {
    params: CodingParams,
    layout: PicLayout,
    maps: ParseMaps,
    edges: EdgeMap,
}

/// A picture with random `QpY`, `bS`, and SAO values, at one CTB size.
fn scene(width: u32, height: u32, ctb_log2: u32, rng: &mut Rng) -> Scene {
    let params = CodingParams {
        width,
        height,
        ctb_log2,
        deblocking_disabled: false,
        ..CodingParams::for_tests()
    };
    let ctb = 1u32 << ctb_log2;
    let layout = PicLayout::new(&LayoutSizes {
        width,
        height,
        ctb_log2,
        min_tb_log2: 2,
        column_widths: &[width.div_ceil(ctb)],
        row_heights: &[height.div_ceil(ctb)],
    })
    .unwrap();
    let mut maps = ParseMaps::new(&params, &layout).unwrap();
    let mut edges = EdgeMap::new(width, height).unwrap();
    for y in (0..height).step_by(8) {
        for x in (0..width).step_by(8) {
            maps.set_qp_y(x, y, 8, 20 + rng.below(30) as i32);
            let strength = rng.below(3) as u8;
            edges.mark_vertical(EdgeRun {
                x,
                y,
                length: 8,
                strength,
            });
            let strength = rng.below(3) as u8;
            edges.mark_horizontal(EdgeRun {
                x,
                y,
                length: 8,
                strength,
            });
        }
    }
    for rs in 0..layout.ctb_count() {
        maps.set_ctb_slice(rs, 0);
        maps.begin_ctb_filter(
            rs,
            CtbFilter {
                beta_offset_div2: 0,
                tc_offset_div2: 0,
                across_slices: true,
            },
        );
        let mut values = SaoCtb::default();
        for plane in &mut values.planes {
            let kind = match rng.below(3) {
                0 => SaoKind::Off,
                1 => SaoKind::Band(rng.below(28) as u8),
                _ => SaoKind::Edge(rng.below(4) as u8),
            };
            *plane = SaoPlane {
                kind,
                offsets: [0, 3, 1, -1, -3],
            };
        }
        maps.set_sao(rs, values);
    }
    Scene {
        params,
        layout,
        maps,
        edges,
    }
}

fn random_picture(width: u32, height: u32, rng: &mut Rng) -> Picture<u8> {
    let mut picture = Picture::<u8>::new(width as usize, height as usize).unwrap();
    for c_idx in 0..3 {
        let plane = picture.plane_mut(c_idx);
        for y in 0..plane.height() {
            let mut around = 128;
            for sample in plane.row_mut(y).unwrap() {
                around = rng.sample(around, 8);
                *sample = around as u8;
            }
        }
    }
    picture
}

const CASES: [(u32, u32, u32); 5] = [
    (128, 96, 4),
    (136, 104, 4),
    (128, 128, 5),
    (200, 72, 6),
    (64, 16, 4),
];

#[test]
fn deblocking_in_rows_equals_the_whole_picture() {
    let mut rng = Rng::new(0x1234_5678);
    let mut changed = false;
    for (width, height, ctb_log2) in CASES {
        let scene = scene(width, height, ctb_log2, &mut rng);
        let context = DeblockContext {
            params: &scene.params,
            layout: &scene.layout,
            maps: &scene.maps,
            edges: &scene.edges,
        };
        let start = random_picture(width, height, &mut rng);
        let mut whole = start.clone();
        let mut rows = start.clone();
        deblock_picture(&mut whole, &context);
        deblock_picture_in_rows(&mut rows, &context);
        changed |= whole != start;
        assert_eq!(whole, rows, "{width}x{height} at CTB log2 {ctb_log2}");
    }
    assert!(changed, "the filter changes no picture");
}

#[test]
fn sao_in_rows_equals_the_whole_picture() {
    let mut rng = Rng::new(0x8765_4321);
    let mut changed = false;
    for (width, height, ctb_log2) in CASES {
        let scene = scene(width, height, ctb_log2, &mut rng);
        let context = SaoContext {
            params: &scene.params,
            layout: &scene.layout,
            maps: &scene.maps,
        };
        let start = random_picture(width, height, &mut rng);
        let mut whole = start.clone();
        let mut rows = start.clone();
        let mut copy = Picture::<u8>::new(width as usize, height as usize).unwrap();
        sao_picture(&mut whole, &mut copy, &context);
        let mut copy = Picture::<u8>::new(width as usize, height as usize).unwrap();
        sao_picture_in_rows(&mut rows, &mut copy, &context);
        changed |= whole != start;
        assert_eq!(whole, rows, "{width}x{height} at CTB log2 {ctb_log2}");
    }
    assert!(changed, "SAO changes no picture");
}
