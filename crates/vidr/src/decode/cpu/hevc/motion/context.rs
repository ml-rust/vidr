//! What motion derivation reads, the prediction block it works on, and the
//! availability of neighbours (6.4.2).

use crate::decode::cpu::hevc::layout::PicLayout;
use crate::decode::cpu::hevc::params::CodingParams;
use crate::decode::cpu::hevc::syntax::maps::ParseMaps;
use crate::decode::cpu::hevc::syntax::records::PartMode;

use super::field::MotionField;
use super::temporal::TemporalContext;
use super::types::{PbMotion, RefLists};

/// The inputs of motion derivation for one slice.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MotionContext<'a> {
    pub(crate) params: &'a CodingParams,
    pub(crate) layout: &'a PicLayout,
    pub(crate) maps: &'a ParseMaps,
    pub(crate) field: &'a MotionField,
    /// `RefPicList0` and `RefPicList1` of the slice.
    pub(crate) refs: &'a RefLists,
    pub(crate) temporal: TemporalContext<'a>,
    /// `PicOrderCntVal` of the current picture.
    pub(crate) poc: i32,
}

/// One prediction block in its coding block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PbGeometry {
    /// `xCb`.
    pub(crate) x_cb: u32,
    /// `yCb`.
    pub(crate) y_cb: u32,
    /// `nCbS`.
    pub(crate) n_cb: u32,
    /// `xPb`.
    pub(crate) x_pb: u32,
    /// `yPb`.
    pub(crate) y_pb: u32,
    /// `nPbW`.
    pub(crate) width: u32,
    /// `nPbH`.
    pub(crate) height: u32,
    /// `partIdx`.
    pub(crate) part_idx: u32,
    /// `PartMode`.
    pub(crate) part_mode: PartMode,
}

impl MotionContext<'_> {
    /// The motion of the prediction block covering `( x_nb, y_nb )`, or `None` when
    /// 6.4.2 finds it unavailable.
    pub(crate) fn neighbour(&self, pb: &PbGeometry, x_nb: i32, y_nb: i32) -> Option<PbMotion> {
        let (x_cb, y_cb, n_cb) = (pb.x_cb as i32, pb.y_cb as i32, pb.n_cb as i32);
        let same_cb = x_cb <= x_nb && y_cb <= y_nb && x_cb + n_cb > x_nb && y_cb + n_cb > y_nb;
        let available = if same_cb {
            let later_part = pb.width << 1 == pb.n_cb
                && pb.height << 1 == pb.n_cb
                && pb.part_idx == 1
                && y_cb + pb.height as i32 <= y_nb
                && x_cb + pb.width as i32 > x_nb;
            !later_part
        } else {
            self.maps
                .available(self.layout, (pb.x_pb, pb.y_pb), x_nb, y_nb)
        };
        if !available {
            return None;
        }
        let (x, y) = (x_nb as u32, y_nb as u32);
        if self.maps.cu_intra(x, y) {
            return None;
        }
        Some(self.field.at(x, y))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::cpu::hevc::motion::test_support::{Scene, inter, pb};
    use crate::decode::cpu::hevc::motion::types::Mv;

    #[test]
    fn a_block_outside_the_picture_is_unavailable() {
        let scene = Scene::new();
        let context = scene.context();
        let block = pb(0, 0, 16, 16);
        assert_eq!(context.neighbour(&block, -1, 0), None);
        assert_eq!(context.neighbour(&block, 0, -1), None);
    }

    #[test]
    fn an_earlier_inter_block_is_available() {
        let mut scene = Scene::new();
        let motion = inter(Mv::new(5, 6), 0);
        scene.set_inter(0, 0, 16, 16, motion);
        let context = scene.context();
        assert_eq!(context.neighbour(&pb(16, 0, 16, 16), 15, 0), Some(motion));
    }

    /// The last condition of 6.4.2.
    #[test]
    fn an_intra_block_is_unavailable() {
        let mut scene = Scene::new();
        scene.set_inter(0, 0, 16, 16, inter(Mv::new(1, 1), 0));
        scene.set_intra(0, 0, 16);
        let context = scene.context();
        assert_eq!(context.neighbour(&pb(16, 0, 16, 16), 15, 0), None);
    }

    /// z-scan order: the block to the right of the current CTB is not decoded yet.
    #[test]
    fn a_later_block_is_unavailable() {
        let mut scene = Scene::new();
        scene.set_inter(16, 0, 16, 16, inter(Mv::new(1, 1), 0));
        let context = scene.context();
        assert_eq!(context.neighbour(&pb(0, 0, 16, 16), 16, 0), None);
    }

    /// The second block of a PART_NxN unit cannot read the third (6.4.2).
    #[test]
    fn nxn_part_1_cannot_read_part_2() {
        let mut scene = Scene::new();
        scene.set_inter(0, 0, 16, 16, inter(Mv::new(1, 1), 0));
        let context = scene.context();
        let block = PbGeometry {
            x_cb: 0,
            y_cb: 0,
            n_cb: 16,
            x_pb: 8,
            y_pb: 0,
            width: 8,
            height: 8,
            part_idx: 1,
            part_mode: PartMode::NxN,
        };
        assert_eq!(context.neighbour(&block, 7, 8), None);
        // The first block of the same unit is available.
        assert!(context.neighbour(&block, 7, 7).is_some());
    }
}
