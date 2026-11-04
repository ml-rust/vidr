//! `uncompressed_header()` (AV1 specification 5.9.2).

use super::film_grain::film_grain_params;
use super::filters::{cdef_params, loop_filter_params, lr_params, read_tx_mode};
use super::global_motion::global_motion_params;
use super::quant::{
    coded_lossless, delta_lf_params, delta_q_params, quantization_params, segmentation_params,
};
use super::ref_table::RefTable;
use super::set_frame_refs::{relative_dist, set_frame_refs};
use super::size::{frame_size, frame_size_with_refs, render_size};
use super::skip_mode::skip_mode_params;
use super::tile_info::tile_info;
use super::types::{
    ALL_FRAMES, DEFAULT_REF_DELTAS, FrameHeader, IDENTITY_WARP, INTRA_ONLY_FRAME, KEY_FRAME,
    LAST_FRAME, NUM_REF_FRAMES, PRIMARY_REF_NONE, REFS_PER_FRAME, SWITCH_FRAME, SWITCHABLE,
    TOTAL_REFS_PER_FRAME,
};
use crate::bitstream::av1::obu::ObuExtension;
use crate::bitstream::av1::seq_header::{
    SELECT_INTEGER_MV, SELECT_SCREEN_CONTENT_TOOLS, SequenceHeader,
};
use crate::bitstream::bits::BitReader;
use crate::error::BitstreamError;

const STRUCTURE: &str = "AV1 frame header";

/// Parses one frame header. `refs` takes the changes the syntax itself states: `RefValid` and
/// `RefOrderHint`. The update of section 7.20 waits for [`RefTable::update`].
///
/// # Errors
///
/// [`BitstreamError`] for a payload that ends early or a field out of range. The same goes for a
/// field that names a reference that holds no frame.
pub(crate) fn parse_frame_header(
    sequence: &SequenceHeader,
    extension: Option<ObuExtension>,
    payload: &[u8],
    refs: &mut RefTable,
) -> Result<FrameHeader, BitstreamError> {
    let mut reader = BitReader::new(payload, STRUCTURE);
    let (temporal_id, spatial_id) =
        extension.map_or((0, 0), |layer| (layer.temporal_id, layer.spatial_id));
    let mut header = FrameHeader {
        temporal_id,
        spatial_id,
        ..FrameHeader::default()
    };
    if sequence.reduced_still_picture_header {
        header.frame_type = KEY_FRAME;
        header.frame_is_intra = true;
        header.show_frame = true;
        header.error_resilient_mode = true;
    } else {
        if reader.bit("show_existing_frame")? {
            return show_existing(&mut reader, sequence, refs, temporal_id, spatial_id);
        }
        frame_kind(&mut reader, sequence, &mut header)?;
    }
    let key_shown = header.frame_type == KEY_FRAME && header.show_frame;
    if key_shown {
        for name in &mut refs.names {
            name.valid = false;
            name.lost = false;
            name.order_hint = 0;
        }
    }
    screen_content(&mut reader, sequence, &mut header)?;
    frame_ids(&mut reader, sequence, refs, &mut header)?;
    header.frame_size_override_flag = if header.frame_type == SWITCH_FRAME {
        true
    } else if sequence.reduced_still_picture_header {
        false
    } else {
        reader.bit("frame_size_override_flag")?
    };
    header.order_hint = reader.u8(u32::from(sequence.order_hint_bits), "order_hint")?;
    header.primary_ref_frame = if header.frame_is_intra || header.error_resilient_mode {
        PRIMARY_REF_NONE
    } else {
        reader.u8(3, "primary_ref_frame")?
    };
    buffer_removal_times(&mut reader, sequence, &mut header)?;
    header.refresh_frame_flags = if header.frame_type == SWITCH_FRAME || key_shown {
        ALL_FRAMES
    } else {
        reader.u8(8, "refresh_frame_flags")?
    };
    check_refresh(&reader, &header)?;
    read_ref_order_hints(&mut reader, sequence, refs, &mut header)?;
    if header.frame_is_intra {
        frame_size(&mut reader, sequence, &mut header)?;
        render_size(&mut reader, &mut header)?;
        if header.allow_screen_content_tools && header.upscaled_width == header.frame_width {
            header.allow_intrabc = reader.bit("allow_intrabc")?;
        }
    } else {
        inter_references(&mut reader, sequence, refs, &mut header)?;
    }
    header.disable_frame_end_update_cdf =
        if sequence.reduced_still_picture_header || header.disable_cdf_update {
            true
        } else {
            reader.bit("disable_frame_end_update_cdf")?
        };
    let previous = load_previous(&reader, refs, &mut header)?;
    header.tile_info = tile_info(&mut reader, sequence, &header)?;
    quantization_params(&mut reader, sequence, &mut header)?;
    segmentation_params(&mut reader, &mut header)?;
    delta_q_params(&mut reader, &mut header)?;
    delta_lf_params(&mut reader, &mut header)?;
    coded_lossless(&mut header);
    loop_filter_params(&mut reader, sequence, &mut header)?;
    cdef_params(&mut reader, sequence, &mut header)?;
    lr_params(&mut reader, sequence, &mut header)?;
    read_tx_mode(&mut reader, &mut header)?;
    header.reference_select = !header.frame_is_intra && reader.bit("reference_select")?;
    skip_mode_params(&mut reader, sequence, refs, &mut header)?;
    header.allow_warped_motion = !header.frame_is_intra
        && !header.error_resilient_mode
        && sequence.enable_warped_motion
        && reader.bit("allow_warped_motion")?;
    header.reduced_tx_set = reader.bit("reduced_tx_set")?;
    global_motion_params(&mut reader, &previous, &mut header)?;
    film_grain_params(&mut reader, sequence, refs, &mut header)?;
    header.header_bits = reader.position();
    Ok(header)
}

/// The `show_existing_frame` branch.
fn show_existing(
    reader: &mut BitReader<'_>,
    sequence: &SequenceHeader,
    refs: &RefTable,
    temporal_id: u8,
    spatial_id: u8,
) -> Result<FrameHeader, BitstreamError> {
    let name = reader.u8(3, "frame_to_show_map_idx")?;
    let presentation = read_temporal_point(reader, sequence)?;
    let saved = refs.get(name);
    if !saved.valid {
        return Err(reader.invalid(
            "frame_to_show_map_idx",
            name.into(),
            "a reference name that holds a frame",
        ));
    }
    if !saved.showable_frame {
        return Err(reader.invalid(
            "frame_to_show_map_idx",
            name.into(),
            "a reference name whose frame has showable_frame 1",
        ));
    }
    let mut display_frame_id = 0;
    if sequence.frame_id_numbers_present_flag {
        display_frame_id = reader.bits(sequence.frame_id_length(), "display_frame_id")? as u32;
        if display_frame_id != saved.frame_id {
            return Err(reader.invalid(
                "display_frame_id",
                display_frame_id.into(),
                "RefFrameId of the name that frame_to_show_map_idx states",
            ));
        }
    }
    let mut header = refs.loaded_header(name, sequence, temporal_id, spatial_id);
    header.frame_presentation_time = presentation;
    header.display_frame_id = display_frame_id;
    header.header_bits = reader.position();
    Ok(header)
}

/// `temporal_point_info()`, when the sequence holds one.
fn read_temporal_point(
    reader: &mut BitReader<'_>,
    sequence: &SequenceHeader,
) -> Result<Option<u32>, BitstreamError> {
    let equal_interval = sequence
        .timing_info
        .is_some_and(|timing| timing.equal_picture_interval);
    match sequence.decoder_model_info {
        Some(model) if !equal_interval => {
            let bits = u32::from(model.frame_presentation_time_length_minus_1) + 1;
            Ok(Some(reader.bits(bits, "frame_presentation_time")? as u32))
        }
        _ => Ok(None),
    }
}

fn frame_kind(
    reader: &mut BitReader<'_>,
    sequence: &SequenceHeader,
    header: &mut FrameHeader,
) -> Result<(), BitstreamError> {
    header.frame_type = reader.u8(2, "frame_type")?;
    header.frame_is_intra = header.frame_type == INTRA_ONLY_FRAME || header.frame_type == KEY_FRAME;
    header.show_frame = reader.bit("show_frame")?;
    if header.show_frame {
        header.frame_presentation_time = read_temporal_point(reader, sequence)?;
    }
    header.showable_frame = if header.show_frame {
        header.frame_type != KEY_FRAME
    } else {
        reader.bit("showable_frame")?
    };
    header.error_resilient_mode = if header.frame_type == SWITCH_FRAME
        || (header.frame_type == KEY_FRAME && header.show_frame)
    {
        true
    } else {
        reader.bit("error_resilient_mode")?
    };
    Ok(())
}

fn screen_content(
    reader: &mut BitReader<'_>,
    sequence: &SequenceHeader,
    header: &mut FrameHeader,
) -> Result<(), BitstreamError> {
    header.disable_cdf_update = reader.bit("disable_cdf_update")?;
    header.allow_screen_content_tools =
        if sequence.seq_force_screen_content_tools == SELECT_SCREEN_CONTENT_TOOLS {
            reader.bit("allow_screen_content_tools")?
        } else {
            sequence.seq_force_screen_content_tools != 0
        };
    if header.allow_screen_content_tools {
        header.coded_force_integer_mv = if sequence.seq_force_integer_mv == SELECT_INTEGER_MV {
            reader.bit("force_integer_mv")?
        } else {
            sequence.seq_force_integer_mv != 0
        };
    }
    header.force_integer_mv = header.coded_force_integer_mv || header.frame_is_intra;
    Ok(())
}

/// `current_frame_id` and `mark_ref_frames()`.
fn frame_ids(
    reader: &mut BitReader<'_>,
    sequence: &SequenceHeader,
    refs: &mut RefTable,
    header: &mut FrameHeader,
) -> Result<(), BitstreamError> {
    if !sequence.frame_id_numbers_present_flag {
        return Ok(());
    }
    let id_len = sequence.frame_id_length();
    header.current_frame_id = reader.bits(id_len, "current_frame_id")? as u32;
    let current = i64::from(header.current_frame_id);
    let diff_span = 1_i64 << (u32::from(sequence.delta_frame_id_length_minus_2) + 2);
    for name in &mut refs.names {
        let id = i64::from(name.frame_id);
        let stale = if current > diff_span {
            id > current || id < current - diff_span
        } else {
            id > current && id < (1_i64 << id_len) + current - diff_span
        };
        if stale {
            name.valid = false;
            name.lost = false;
        }
    }
    Ok(())
}

fn buffer_removal_times(
    reader: &mut BitReader<'_>,
    sequence: &SequenceHeader,
    header: &mut FrameHeader,
) -> Result<(), BitstreamError> {
    let Some(model) = sequence.decoder_model_info else {
        return Ok(());
    };
    header.buffer_removal_time_present_flag = reader.bit("buffer_removal_time_present_flag")?;
    if !header.buffer_removal_time_present_flag {
        return Ok(());
    }
    let bits = u32::from(model.buffer_removal_time_length_minus_1) + 1;
    for (index, point) in sequence.operating_points.iter().enumerate() {
        if !point.decoder_model_present {
            continue;
        }
        let in_temporal = (point.idc >> header.temporal_id) & 1 == 1;
        let in_spatial = (point.idc >> (u16::from(header.spatial_id) + 8)) & 1 == 1;
        if point.idc == 0 || (in_temporal && in_spatial) {
            let time = reader.bits(bits, "buffer_removal_time")? as u32;
            header.buffer_removal_times.push((index as u8, time));
        }
    }
    Ok(())
}

/// Refuses what section 6.8.2 forbids: an intra-only frame that refreshes every name.
fn check_refresh(reader: &BitReader<'_>, header: &FrameHeader) -> Result<(), BitstreamError> {
    if header.frame_type == INTRA_ONLY_FRAME && header.refresh_frame_flags == ALL_FRAMES {
        return Err(reader.invalid(
            "refresh_frame_flags",
            u64::from(ALL_FRAMES),
            "a value other than 0xFF on an intra-only frame",
        ));
    }
    Ok(())
}

/// `ref_order_hint`. A name that differs goes invalid, and takes the coded hint as Annex C states.
/// A name that held a frame until here is lost, and `lost_names` tells the decoder that owns it.
fn read_ref_order_hints(
    reader: &mut BitReader<'_>,
    sequence: &SequenceHeader,
    refs: &mut RefTable,
    header: &mut FrameHeader,
) -> Result<(), BitstreamError> {
    let refreshes_all = header.refresh_frame_flags == ALL_FRAMES;
    let reads = (!header.frame_is_intra || !refreshes_all)
        && header.error_resilient_mode
        && sequence.enable_order_hint;
    if !reads {
        return Ok(());
    }
    let mut hints = [0; NUM_REF_FRAMES];
    for (index, (hint, name)) in hints.iter_mut().zip(&mut refs.names).enumerate() {
        *hint = reader.u8(u32::from(sequence.order_hint_bits), "ref_order_hint")?;
        if *hint != name.order_hint {
            if name.valid {
                name.lost = true;
                header.lost_names |= 1 << index;
            }
            name.valid = false;
            name.order_hint = *hint;
        }
    }
    header.ref_order_hint = Some(hints);
    Ok(())
}

/// Refuses a reference name section 6.8.2 forbids: one that holds no frame, or one whose frame
/// `mark_ref_frames()` retired. A name `ref_order_hint` states as lost passes, as Annex C states.
/// dav1d refuses an empty name, libaom refuses a retired one, and libaom greys a lost one.
fn check_reference(
    reader: &BitReader<'_>,
    refs: &RefTable,
    name: u8,
) -> Result<(), BitstreamError> {
    let saved = refs.get(name);
    if saved.valid || saved.lost {
        return Ok(());
    }
    Err(reader.invalid(
        "ref_frame_idx",
        name.into(),
        "a reference name with RefValid 1, so start the stream at a key frame",
    ))
}

/// The inter branch: references, frame size, and motion vector tools.
fn inter_references(
    reader: &mut BitReader<'_>,
    sequence: &SequenceHeader,
    refs: &RefTable,
    header: &mut FrameHeader,
) -> Result<(), BitstreamError> {
    if sequence.enable_order_hint {
        header.frame_refs_short_signaling = reader.bit("frame_refs_short_signaling")?;
        if header.frame_refs_short_signaling {
            header.last_frame_idx = reader.u8(3, "last_frame_idx")?;
            header.gold_frame_idx = reader.u8(3, "gold_frame_idx")?;
            header.ref_frame_idx = set_frame_refs(
                header.last_frame_idx,
                header.gold_frame_idx,
                &refs.order_hints(),
                header.order_hint,
                sequence.order_hint_bits,
            )?;
        }
    }
    for index in 0..REFS_PER_FRAME {
        if !header.frame_refs_short_signaling {
            header.ref_frame_idx[index] = reader.u8(3, "ref_frame_idx")?;
        }
        let name = header.ref_frame_idx[index];
        check_reference(reader, refs, name)?;
        if sequence.frame_id_numbers_present_flag {
            let bits = u32::from(sequence.delta_frame_id_length_minus_2) + 2;
            let delta = reader.bits(bits, "delta_frame_id_minus_1")? as u32;
            header.delta_frame_id_minus_1[index] = delta;
            let modulus = 1_u64 << sequence.frame_id_length();
            let expected =
                (u64::from(header.current_frame_id) + modulus - u64::from(delta) - 1) % modulus;
            header.expected_frame_id[index] = expected as u32;
            if expected as u32 != refs.get(name).frame_id {
                return Err(reader.invalid(
                    "delta_frame_id_minus_1",
                    delta.into(),
                    "a delta that gives RefFrameId of the name in ref_frame_idx",
                ));
            }
        }
    }
    if header.frame_size_override_flag && !header.error_resilient_mode {
        frame_size_with_refs(reader, sequence, refs, header)?;
    } else {
        frame_size(reader, sequence, header)?;
        render_size(reader, header)?;
    }
    header.allow_high_precision_mv =
        !header.force_integer_mv && reader.bit("allow_high_precision_mv")?;
    header.is_filter_switchable = reader.bit("is_filter_switchable")?;
    header.interpolation_filter = if header.is_filter_switchable {
        SWITCHABLE
    } else {
        reader.u8(2, "interpolation_filter")?
    };
    header.is_motion_mode_switchable = reader.bit("is_motion_mode_switchable")?;
    header.use_ref_frame_mvs = !header.error_resilient_mode
        && sequence.enable_ref_frame_mvs
        && reader.bit("use_ref_frame_mvs")?;
    for index in 0..REFS_PER_FRAME {
        let reference = LAST_FRAME + index;
        let hint = refs.get(header.ref_frame_idx[index]).order_hint;
        header.order_hints[reference] = hint;
        header.ref_frame_sign_bias[reference] = sequence.enable_order_hint
            && relative_dist(
                i32::from(hint),
                i32::from(header.order_hint),
                sequence.order_hint_bits,
            ) > 0;
    }
    Ok(())
}

/// `setup_past_independence()` or `load_previous()`. The result is `PrevGmParams`.
fn load_previous(
    reader: &BitReader<'_>,
    refs: &RefTable,
    header: &mut FrameHeader,
) -> Result<[[i32; 6]; TOTAL_REFS_PER_FRAME], BitstreamError> {
    if header.primary_ref_frame == PRIMARY_REF_NONE {
        header.loop_filter.delta_enabled = true;
        header.loop_filter.ref_deltas = DEFAULT_REF_DELTAS;
        header.loop_filter.mode_deltas = [0; 2];
        return Ok([IDENTITY_WARP; TOTAL_REFS_PER_FRAME]);
    }
    let name = header.ref_frame_idx[usize::from(header.primary_ref_frame)];
    let saved = refs.get(name);
    if !saved.valid {
        return Err(reader.invalid(
            "primary_ref_frame",
            header.primary_ref_frame.into(),
            "a reference whose name holds a frame",
        ));
    }
    header.loop_filter.ref_deltas = saved.loop_filter_ref_deltas;
    header.loop_filter.mode_deltas = saved.loop_filter_mode_deltas;
    header.segmentation.feature_enabled = saved.feature_enabled;
    header.segmentation.feature_data = saved.feature_data;
    Ok(saved.gm_params)
}
