//! What sections 7.20 and 7.21 save for each of the eight reference names, as far as header
//! syntax reads it. A decoder owns the samples and the CDFs.

use super::types::{
    ALL_FRAMES, FilmGrain, FrameHeader, INTRA_ONLY_FRAME, KEY_FRAME, MAX_SEGMENTS, NUM_REF_FRAMES,
    SEG_LVL_MAX, TOTAL_REFS_PER_FRAME,
};
use crate::bitstream::av1::seq_header::SequenceHeader;

/// What one reference name holds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SavedRef {
    /// `RefValid`.
    pub(crate) valid: bool,
    /// Whether `RefValid` went to 0 through `ref_order_hint`: the name's frame is lost, as Annex C
    /// states. A later frame can still name it, and a decoder conceals what the frame held.
    pub(crate) lost: bool,
    /// `RefFrameId`.
    pub(crate) frame_id: u32,
    /// `RefUpscaledWidth`.
    pub(crate) upscaled_width: u32,
    /// `RefFrameWidth`.
    pub(crate) frame_width: u32,
    /// `RefFrameHeight`.
    pub(crate) frame_height: u32,
    /// `RefRenderWidth`.
    pub(crate) render_width: u32,
    /// `RefRenderHeight`.
    pub(crate) render_height: u32,
    /// `RefMiCols`.
    pub(crate) mi_cols: u32,
    /// `RefMiRows`.
    pub(crate) mi_rows: u32,
    /// `RefFrameType`.
    pub(crate) frame_type: u8,
    /// `RefOrderHint`.
    pub(crate) order_hint: u8,
    /// `SavedOrderHints`.
    pub(crate) saved_order_hints: [u8; TOTAL_REFS_PER_FRAME],
    /// `SavedGmParams`.
    pub(crate) gm_params: [[i32; 6]; TOTAL_REFS_PER_FRAME],
    pub(crate) loop_filter_ref_deltas: [i8; TOTAL_REFS_PER_FRAME],
    pub(crate) loop_filter_mode_deltas: [i8; 2],
    pub(crate) feature_enabled: [[bool; SEG_LVL_MAX]; MAX_SEGMENTS],
    pub(crate) feature_data: [[i16; SEG_LVL_MAX]; MAX_SEGMENTS],
    pub(crate) film_grain: FilmGrain,
    pub(crate) showable_frame: bool,
}

/// The eight reference names.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct RefTable {
    pub(crate) names: [SavedRef; NUM_REF_FRAMES],
}

impl RefTable {
    /// What `name` holds. `name` comes from a 3-bit field.
    pub(crate) fn get(&self, name: u8) -> &SavedRef {
        &self.names[usize::from(name & 7)]
    }

    /// `RefOrderHint` of every name.
    pub(crate) fn order_hints(&self) -> [u8; NUM_REF_FRAMES] {
        self.names.map(|saved| saved.order_hint)
    }

    /// The reference frame update process of section 7.20 for a frame that finished decoding.
    pub(crate) fn update(&mut self, header: &FrameHeader) {
        let saved = SavedRef {
            valid: true,
            lost: false,
            frame_id: header.current_frame_id,
            upscaled_width: header.upscaled_width,
            frame_width: header.frame_width,
            frame_height: header.frame_height,
            render_width: header.render_width,
            render_height: header.render_height,
            mi_cols: header.mi_cols,
            mi_rows: header.mi_rows,
            frame_type: header.frame_type,
            order_hint: header.order_hint,
            saved_order_hints: header.order_hints,
            gm_params: header.gm_params,
            loop_filter_ref_deltas: header.loop_filter.ref_deltas,
            loop_filter_mode_deltas: header.loop_filter.mode_deltas,
            feature_enabled: header.segmentation.feature_enabled,
            feature_data: header.segmentation.feature_data,
            film_grain: header.film_grain,
            showable_frame: header.showable_frame,
        };
        for (index, name) in self.names.iter_mut().enumerate() {
            if (header.refresh_frame_flags >> index) & 1 == 1 {
                *name = saved;
            }
        }
    }

    /// The header a `show_existing_frame` of `name` loads: section 7.21. A key frame refreshes
    /// every name, and a second show of it is not allowed, so it stops being showable.
    pub(crate) fn loaded_header(
        &self,
        name: u8,
        sequence: &SequenceHeader,
        temporal_id: u8,
        spatial_id: u8,
    ) -> FrameHeader {
        let saved = self.get(name);
        let key = saved.frame_type == KEY_FRAME;
        let mut header = FrameHeader {
            temporal_id,
            spatial_id,
            show_existing_frame: true,
            frame_to_show_map_idx: name & 7,
            frame_type: saved.frame_type,
            frame_is_intra: key || saved.frame_type == INTRA_ONLY_FRAME,
            show_frame: true,
            showable_frame: !key,
            current_frame_id: saved.frame_id,
            order_hint: saved.order_hint,
            refresh_frame_flags: if key { ALL_FRAMES } else { 0 },
            upscaled_width: saved.upscaled_width,
            frame_width: saved.frame_width,
            frame_height: saved.frame_height,
            render_width: saved.render_width,
            render_height: saved.render_height,
            mi_cols: saved.mi_cols,
            mi_rows: saved.mi_rows,
            order_hints: saved.saved_order_hints,
            gm_params: saved.gm_params,
            ..FrameHeader::default()
        };
        header.loop_filter.ref_deltas = saved.loop_filter_ref_deltas;
        header.loop_filter.mode_deltas = saved.loop_filter_mode_deltas;
        header.segmentation.feature_enabled = saved.feature_enabled;
        header.segmentation.feature_data = saved.feature_data;
        if sequence.film_grain_params_present {
            header.film_grain = saved.film_grain;
        }
        header
    }
}
