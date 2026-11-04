//! `skip_mode_params()` (AV1 specification 5.9.22).

use super::ref_table::RefTable;
use super::set_frame_refs::relative_dist;
use super::types::{FrameHeader, LAST_FRAME, REFS_PER_FRAME};
use crate::bitstream::av1::seq_header::SequenceHeader;
use crate::bitstream::bits::BitReader;
use crate::error::BitstreamError;

/// `SkipModeFrame`, or `None` when `skipModeAllowed` is 0.
fn skip_mode_frames(hints: &[i32; REFS_PER_FRAME], order_hint: i32, bits: u8) -> Option<[u8; 2]> {
    let dist = |a, b| relative_dist(a, b, bits);
    let mut forward: Option<(usize, i32)> = None;
    let mut backward: Option<(usize, i32)> = None;
    for (index, &hint) in hints.iter().enumerate() {
        if dist(hint, order_hint) < 0 {
            if forward.is_none_or(|(_, best)| dist(hint, best) > 0) {
                forward = Some((index, hint));
            }
        } else if dist(hint, order_hint) > 0
            && backward.is_none_or(|(_, best)| dist(hint, best) < 0)
        {
            backward = Some((index, hint));
        }
    }
    let (forward_index, forward_hint) = forward?;
    let other = match backward {
        Some((index, _)) => index,
        None => {
            let mut second: Option<(usize, i32)> = None;
            for (index, &hint) in hints.iter().enumerate() {
                if dist(hint, forward_hint) < 0
                    && second.is_none_or(|(_, best)| dist(hint, best) > 0)
                {
                    second = Some((index, hint));
                }
            }
            second?.0
        }
    };
    Some([
        (LAST_FRAME + forward_index.min(other)) as u8,
        (LAST_FRAME + forward_index.max(other)) as u8,
    ])
}

/// Reads `skip_mode_params()`.
pub(super) fn skip_mode_params(
    reader: &mut BitReader<'_>,
    sequence: &SequenceHeader,
    refs: &RefTable,
    header: &mut FrameHeader,
) -> Result<(), BitstreamError> {
    if header.frame_is_intra || !header.reference_select || !sequence.enable_order_hint {
        return Ok(());
    }
    let hints = header
        .ref_frame_idx
        .map(|name| i32::from(refs.get(name).order_hint));
    let frames = skip_mode_frames(
        &hints,
        i32::from(header.order_hint),
        sequence.order_hint_bits,
    );
    if let Some(frames) = frames {
        header.skip_mode_allowed = true;
        header.skip_mode_frame = frames;
        header.skip_mode_present = reader.bit("skip_mode_present")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_forward_and_a_backward_reference_are_the_pair() {
        // Current 4: the closest before is index 1, and the closest after is index 3.
        let frames = skip_mode_frames(&[1, 3, 2, 6, 7, 1, 2], 4, 4);
        assert_eq!(frames, Some([2, 4]));
    }

    #[test]
    fn two_forward_references_are_the_pair_with_no_backward_one() {
        // Current 4: the closest before is index 2, and the next is index 0.
        let frames = skip_mode_frames(&[2, 1, 3, 1, 1, 1, 1], 4, 4);
        assert_eq!(frames, Some([1, 3]));
    }

    #[test]
    fn one_forward_hint_alone_allows_no_skip_mode() {
        assert_eq!(skip_mode_frames(&[3; 7], 4, 4), None);
        assert_eq!(skip_mode_frames(&[6; 7], 4, 4), None);
    }
}
