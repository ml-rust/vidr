//! VP9 colour: `color_space` and `color_range` from the uncompressed header.
//!
//! A VP9 stream states its colour only in the header of a key frame, or of an
//! intra-only frame of profile above 0. The container's `vpcC` is a copy of
//! it. NVDEC reports an all-zero video signal description for VP9, so a
//! decoder reads the header itself.
//!
//! The syntax follows section 6.2 of the VP9 bitstream specification, v0.6.
//! A chunk can be a superframe: frames followed by an index. [`vp9_color`]
//! reads one frame, and [`vp9_packet_color`] reads every frame of a chunk.

use super::bits::BitReader;
use crate::error::BitstreamError;

pub(super) const STRUCTURE: &str = "VP9 uncompressed header";
const SUPERFRAME: &str = "VP9 superframe index";

/// `color_space` 7: sRGB. The range is full and the matrix is identity.
pub(super) const CS_RGB: u8 = 7;
/// The three bytes of `frame_sync_code`.
const SYNC_CODE: [u8; 3] = [0x49, 0x83, 0x42];

/// The colour a VP9 frame header states.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Vp9Color {
    /// `color_space`: 0 unknown, 1 BT.601, 2 BT.709, 3 SMPTE 170M,
    /// 4 SMPTE 240M, 5 BT.2020, 6 reserved, 7 sRGB.
    pub color_space: u8,
    /// `color_range`: true for full range. `color_space` 7 is always full.
    pub full_range: bool,
    /// Bits per sample: 8 for profiles 0 and 1, 10 or 12 for 2 and 3.
    pub bit_depth: u8,
}

/// Reads the colour from one VP9 frame.
///
/// Returns `Ok(None)` for a frame that states no colour: a
/// `show_existing_frame` header, an inter frame, and an intra-only frame of
/// profile 0. The last of these implies BT.601 at 8 bits but leaves the
/// range from the previous key frame, so the caller keeps its values.
///
/// # Errors
///
/// [`BitstreamError::Truncated`] when the frame ends inside the header
/// fields up to `color_range`, and [`BitstreamError::InvalidField`] for a
/// `frame_marker` other than 2, a non-zero reserved bit, or a wrong sync code.
pub fn vp9_color(frame: &[u8]) -> Result<Option<Vp9Color>, BitstreamError> {
    let mut reader = BitReader::new(frame, STRUCTURE);
    let marker = reader.u8(2, "frame_marker")?;
    if marker != 2 {
        return Err(invalid("frame_marker", u64::from(marker), "2"));
    }
    let low = reader.u8(1, "profile_low_bit")?;
    let high = reader.u8(1, "profile_high_bit")?;
    let profile = (high << 1) | low;
    if profile == 3 && reader.bit("reserved_zero")? {
        return Err(invalid("reserved_zero", 1, "0"));
    }
    if reader.bit("show_existing_frame")? {
        return Ok(None);
    }
    let key_frame = !reader.bit("frame_type")?;
    let show_frame = reader.bit("show_frame")?;
    let error_resilient = reader.bit("error_resilient_mode")?;
    if !key_frame {
        let intra_only = !show_frame && reader.bit("intra_only")?;
        if !intra_only {
            return Ok(None);
        }
        if !error_resilient {
            reader.u8(2, "reset_frame_context")?;
        }
    }
    sync_code(&mut reader)?;
    if !key_frame && profile == 0 {
        return Ok(None);
    }
    color_config(&mut reader, profile).map(Some)
}

/// Reads the colour from a chunk of one frame or a superframe.
///
/// Returns the colour of the last frame that states one, or `Ok(None)` when
/// no frame does. A chunk whose last byte is not a superframe marker is one
/// frame. A frame of size 0 is skipped.
///
/// # Errors
///
/// As [`vp9_color`], for the first frame that fails, and
/// [`BitstreamError::InvalidField`] for a frame size that runs into the index.
pub fn vp9_packet_color(packet: &[u8]) -> Result<Option<Vp9Color>, BitstreamError> {
    let mut found = None;
    for frame in split_superframe(packet)? {
        if let Some(color) = vp9_color(frame)? {
            found = Some(color);
        }
    }
    Ok(found)
}

/// Splits a chunk into its frames (Annex B of the VP9 specification). A chunk
/// whose last byte is not a superframe marker is one frame. A frame of size 0
/// is dropped.
///
/// # Errors
///
/// [`BitstreamError::InvalidField`] for a frame size that runs into the index.
pub(super) fn split_superframe(packet: &[u8]) -> Result<Vec<&[u8]>, BitstreamError> {
    let Some(index) = SuperframeIndex::of(packet) else {
        return Ok(vec![packet]);
    };
    let mut frames = Vec::with_capacity(index.frames);
    let mut start = 0usize;
    for position in 0..index.frames {
        let size = index.size(packet, position);
        let frame = start
            .checked_add(size)
            .and_then(|end| packet.get(start..end).filter(|_| end <= index.start))
            .ok_or(BitstreamError::InvalidField {
                structure: SUPERFRAME,
                field: "frame_sizes",
                value: size as u64,
                allowed: "frame sizes that end before the superframe index",
            })?;
        start += size;
        if !frame.is_empty() {
            frames.push(frame);
        }
    }
    Ok(frames)
}

fn invalid(field: &'static str, value: u64, allowed: &'static str) -> BitstreamError {
    BitstreamError::InvalidField {
        structure: STRUCTURE,
        field,
        value,
        allowed,
    }
}

/// The field name and allowed value of each `frame_sync_code` byte.
const SYNC_FIELDS: [(&str, &str); 3] = [
    ("frame_sync_byte_0", "0x49"),
    ("frame_sync_byte_1", "0x83"),
    ("frame_sync_byte_2", "0x42"),
];

/// Reads `frame_sync_code()` (6.2.1).
pub(super) fn sync_code(reader: &mut BitReader<'_>) -> Result<(), BitstreamError> {
    for (expected, (field, allowed)) in SYNC_CODE.into_iter().zip(SYNC_FIELDS) {
        let byte = reader.u8(8, field)?;
        if byte != expected {
            return Err(BitstreamError::InvalidField {
                structure: STRUCTURE,
                field,
                value: u64::from(byte),
                allowed,
            });
        }
    }
    Ok(())
}

/// Reads `color_config` up to `color_range`. The subsampling bits after it
/// are not needed.
fn color_config(reader: &mut BitReader<'_>, profile: u8) -> Result<Vp9Color, BitstreamError> {
    let bit_depth = if profile >= 2 {
        if reader.bit("ten_or_twelve_bit")? {
            12
        } else {
            10
        }
    } else {
        8
    };
    let color_space = reader.u8(3, "color_space")?;
    let full_range = color_space == CS_RGB || reader.bit("color_range")?;
    Ok(Vp9Color {
        color_space,
        full_range,
        bit_depth,
    })
}

/// A superframe index at the end of a chunk.
struct SuperframeIndex {
    /// Offset of the index in the chunk.
    start: usize,
    frames: usize,
    /// Bytes per frame size.
    size_bytes: usize,
}

impl SuperframeIndex {
    /// The index of `packet`, or `None` when the chunk is one frame.
    ///
    /// The last byte must hold marker `0b110`, and the first byte of the
    /// index must equal it.
    fn of(packet: &[u8]) -> Option<Self> {
        let marker = *packet.last()?;
        if marker >> 5 != 0b110 {
            return None;
        }
        let frames = usize::from(marker & 7) + 1;
        let size_bytes = usize::from((marker >> 3) & 3) + 1;
        let length = 2 + frames * size_bytes;
        let start = packet.len().checked_sub(length)?;
        (packet.get(start) == Some(&marker)).then_some(Self {
            start,
            frames,
            size_bytes,
        })
    }

    /// The size of frame `position`, little-endian in the index.
    fn size(&self, packet: &[u8], position: usize) -> usize {
        let first = self.start + 1 + position * self.size_bytes;
        packet
            .get(first..first + self.size_bytes)
            .unwrap_or_default()
            .iter()
            .rev()
            .fold(0usize, |size, byte| (size << 8) | usize::from(*byte))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Packs `(value, width)` fields into bytes, most significant bit first,
    /// padding the last byte with zeros.
    fn pack(fields: &[(u32, u32)]) -> Vec<u8> {
        let mut bits = Vec::new();
        for &(value, width) in fields {
            for shift in (0..width).rev() {
                bits.push((value >> shift) & 1);
            }
        }
        bits.chunks(8)
            .map(|chunk| {
                chunk.iter().enumerate().fold(0u8, |byte, (index, bit)| {
                    byte | ((*bit as u8) << (7 - index))
                })
            })
            .collect()
    }

    fn profile_fields(profile: u32) -> Vec<(u32, u32)> {
        let mut fields = vec![(2, 2), (profile & 1, 1), (profile >> 1, 1)];
        if profile == 3 {
            fields.push((0, 1));
        }
        fields
    }

    fn sync() -> Vec<(u32, u32)> {
        SYNC_CODE.iter().map(|byte| (u32::from(*byte), 8)).collect()
    }

    /// A key frame header up to and including the bits after `color_range`.
    fn key_frame(profile: u32, space: u32, range: u32) -> Vec<u8> {
        let mut fields = profile_fields(profile);
        // show_existing_frame 0, frame_type 0, show_frame 1, error_res 0.
        fields.extend([(0, 1), (0, 1), (1, 1), (0, 1)]);
        fields.extend(sync());
        if profile >= 2 {
            fields.push((0, 1));
        }
        fields.push((space, 3));
        if space != 7 {
            fields.push((range, 1));
            if profile == 1 || profile == 3 {
                fields.extend([(1, 1), (1, 1), (0, 1)]);
            }
        } else if profile == 1 || profile == 3 {
            fields.push((0, 1));
        }
        // Frame size bits follow in a real header.
        fields.push((0x1234, 16));
        pack(&fields)
    }

    /// An intra-only frame: show_frame 0, intra_only 1, reset_frame_context 0.
    fn intra_only(profile: u32, space: u32, range: u32) -> Vec<u8> {
        let mut fields = profile_fields(profile);
        fields.extend([(0, 1), (1, 1), (0, 1), (0, 1), (1, 1), (0, 2)]);
        fields.extend(sync());
        if profile > 0 {
            if profile >= 2 {
                fields.push((1, 1));
            }
            fields.extend([(space, 3), (range, 1)]);
            if profile == 1 || profile == 3 {
                fields.extend([(1, 1), (1, 1), (0, 1)]);
            }
        }
        fields.push((0xff, 8));
        pack(&fields)
    }

    fn color(space: u8, full_range: bool, bit_depth: u8) -> Option<Vp9Color> {
        Some(Vp9Color {
            color_space: space,
            full_range,
            bit_depth,
        })
    }

    #[test]
    fn every_color_space_reads_back() {
        for space in 0..=6u32 {
            for range in 0..=1u32 {
                assert_eq!(
                    vp9_color(&key_frame(0, space, range)),
                    Ok(color(space as u8, range == 1, 8)),
                    "space {space} range {range}"
                );
            }
        }
    }

    #[test]
    fn rgb_is_always_full_range() {
        // Profile 1 and 3 hold sRGB. The range bit does not exist for it.
        for profile in [1, 3] {
            let depth = if profile == 3 { 10 } else { 8 };
            assert_eq!(
                vp9_color(&key_frame(profile, 7, 0)),
                Ok(color(7, true, depth)),
                "profile {profile}"
            );
        }
    }

    #[test]
    fn every_profile_reads_its_bit_depth() {
        for (profile, depth) in [(0, 8), (1, 8), (2, 10), (3, 10)] {
            assert_eq!(
                vp9_color(&key_frame(profile, 2, 1)),
                Ok(color(2, true, depth)),
                "profile {profile}"
            );
        }
    }

    #[test]
    fn twelve_bit_is_read() {
        let mut fields = profile_fields(2);
        fields.extend([(0, 1), (0, 1), (1, 1), (0, 1)]);
        fields.extend(sync());
        fields.extend([(1, 1), (5, 3), (0, 1), (0, 16)]);
        assert_eq!(vp9_color(&pack(&fields)), Ok(color(5, false, 12)));
    }

    #[test]
    fn truncation_at_every_byte_is_an_error_or_none() {
        for profile in 0..=3 {
            let frame = key_frame(profile, 2, 1);
            // Bits up to color_range: header, sync, depth, space, range.
            let needed_bits =
                2 + 2 + u32::from(profile == 3) + 4 + 24 + u32::from(profile >= 2) + 4;
            let needed = needed_bits.div_ceil(8) as usize;
            for length in 0..frame.len() {
                let result = vp9_color(&frame[..length]);
                if length < needed {
                    assert!(
                        matches!(result, Err(BitstreamError::Truncated { .. })),
                        "profile {profile} length {length}: {result:?}"
                    );
                }
            }
        }
        let frame = intra_only(1, 2, 1);
        for length in 0..frame.len() {
            let _ = vp9_color(&frame[..length]);
        }
        assert!(matches!(
            vp9_packet_color(&[]),
            Err(BitstreamError::Truncated { .. })
        ));
    }

    #[test]
    fn show_existing_frame_states_no_colour() {
        // frame_marker 2, profile 0, show_existing_frame 1, index 3.
        let frame = pack(&[(2, 2), (0, 2), (1, 1), (3, 3)]);
        assert_eq!(vp9_color(&frame), Ok(None));
        let frame = pack(&[(2, 2), (1, 1), (1, 1), (0, 1), (1, 1), (0, 3)]);
        assert_eq!(vp9_color(&frame), Ok(None));
    }

    #[test]
    fn inter_frames_state_no_colour() {
        // frame_type 1, show_frame 1, error_res 0, reset_frame_context 0.
        let shown = pack(&[
            (2, 2),
            (0, 2),
            (0, 1),
            (1, 1),
            (1, 1),
            (0, 1),
            (0, 2),
            (0, 8),
        ]);
        assert_eq!(vp9_color(&shown), Ok(None));
        // show_frame 0 and intra_only 0.
        let hidden = pack(&[
            (2, 2),
            (0, 2),
            (0, 1),
            (1, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 8),
        ]);
        assert_eq!(vp9_color(&hidden), Ok(None));
    }

    #[test]
    fn intra_only_reads_colour_above_profile_0() {
        assert_eq!(vp9_color(&intra_only(0, 2, 1)), Ok(None));
        assert_eq!(vp9_color(&intra_only(1, 5, 1)), Ok(color(5, true, 8)));
        assert_eq!(vp9_color(&intra_only(2, 2, 0)), Ok(color(2, false, 12)));
        assert_eq!(vp9_color(&intra_only(3, 1, 1)), Ok(color(1, true, 12)));
    }

    #[test]
    fn error_resilient_intra_only_skips_reset_frame_context() {
        // show_frame 0, error_res 1, intra_only 1, then the sync code.
        let mut fields = profile_fields(1);
        fields.extend([(0, 1), (1, 1), (0, 1), (1, 1), (1, 1)]);
        fields.extend(sync());
        fields.extend([(2, 3), (0, 1), (0, 16)]);
        assert_eq!(vp9_color(&pack(&fields)), Ok(color(2, false, 8)));
    }

    #[test]
    fn bad_headers_name_the_field() {
        let mut frame = key_frame(0, 2, 0);
        frame[1] ^= 0xff;
        assert!(matches!(
            vp9_color(&frame),
            Err(BitstreamError::InvalidField {
                field: "frame_sync_byte_0",
                ..
            })
        ));
        let mut frame = key_frame(0, 2, 0);
        frame[0] &= 0x3f;
        assert!(matches!(
            vp9_color(&frame),
            Err(BitstreamError::InvalidField {
                field: "frame_marker",
                value: 0,
                ..
            })
        ));
        // Profile 3 with the reserved bit set.
        let frame = pack(&[(2, 2), (1, 1), (1, 1), (1, 1), (0, 8)]);
        assert!(matches!(
            vp9_color(&frame),
            Err(BitstreamError::InvalidField {
                field: "reserved_zero",
                ..
            })
        ));
    }

    #[test]
    fn a_wrong_sync_code_is_refused() {
        let mut fields = profile_fields(0);
        fields.extend([(0, 1), (0, 1), (1, 1), (0, 1)]);
        fields.extend([(0x49, 8), (0x83, 8), (0x41, 8), (0, 8)]);
        assert!(matches!(
            vp9_color(&pack(&fields)),
            Err(BitstreamError::InvalidField {
                field: "frame_sync_byte_2",
                value: 0x41,
                allowed: "0x42",
                ..
            })
        ));
    }

    /// A superframe of `frames` with a `size_bytes`-byte index.
    fn superframe(frames: &[Vec<u8>], size_bytes: usize) -> Vec<u8> {
        let marker = 0xc0 | (((size_bytes - 1) as u8) << 3) | (frames.len() - 1) as u8;
        let mut chunk: Vec<u8> = frames.concat();
        chunk.push(marker);
        for frame in frames {
            chunk.extend(&frame.len().to_le_bytes()[..size_bytes]);
        }
        chunk.push(marker);
        chunk
    }

    #[test]
    fn a_superframe_gives_the_last_frame_with_colour() {
        let inter = pack(&[
            (2, 2),
            (0, 2),
            (0, 1),
            (1, 1),
            (1, 1),
            (0, 1),
            (0, 2),
            (0, 8),
        ]);
        let key = key_frame(0, 2, 0);
        let other = key_frame(0, 5, 1);
        for size_bytes in 1..=4 {
            let chunk = superframe(&[key.clone(), inter.clone()], size_bytes);
            assert_eq!(vp9_packet_color(&chunk), Ok(color(2, false, 8)));
            // The key frame is not first.
            let chunk = superframe(&[inter.clone(), key.clone()], size_bytes);
            assert_eq!(vp9_packet_color(&chunk), Ok(color(2, false, 8)));
            let chunk = superframe(&[key.clone(), other.clone()], size_bytes);
            assert_eq!(vp9_packet_color(&chunk), Ok(color(5, true, 8)));
        }
    }

    #[test]
    fn a_plain_frame_and_a_bad_index_are_handled() {
        let key = key_frame(0, 1, 0);
        assert_eq!(vp9_packet_color(&key), Ok(color(1, false, 8)));
        // A last byte of 0xc0 with a mismatched first index byte is one frame.
        let mut chunk = key.clone();
        chunk.extend([0, 0, 0xc0]);
        assert_eq!(vp9_packet_color(&chunk), Ok(color(1, false, 8)));
        // A frame size past the index is an error, never a panic.
        let mut chunk = superframe(std::slice::from_ref(&key), 1);
        let size_at = chunk.len() - 2;
        chunk[size_at] = 200;
        assert!(matches!(
            vp9_packet_color(&chunk),
            Err(BitstreamError::InvalidField {
                field: "frame_sizes",
                ..
            })
        ));
        // Every prefix of a superframe returns without a panic.
        let chunk = superframe(&[key.clone(), key], 2);
        for length in 0..=chunk.len() {
            let _ = vp9_packet_color(&chunk[..length]);
        }
    }

    #[test]
    fn splitting_a_superframe_returns_each_frame_and_drops_empty_ones() {
        let first = key_frame(0, 1, 0);
        let second = key_frame(0, 2, 0);
        let chunk = superframe(&[first.clone(), second.clone()], 2);
        assert_eq!(
            split_superframe(&chunk),
            Ok(vec![first.as_slice(), second.as_slice()])
        );
        assert_eq!(split_superframe(&first), Ok(vec![first.as_slice()]));
        let chunk = superframe(&[first.clone(), Vec::new()], 1);
        assert_eq!(split_superframe(&chunk), Ok(vec![first.as_slice()]));
    }

    #[test]
    fn a_size_that_overflows_is_an_error() {
        let key = key_frame(0, 1, 0);
        let mut chunk = key;
        // Marker 0b110, four size bytes, one frame; size 0xffffffff.
        let marker = 0xc0 | (3 << 3);
        chunk.push(marker);
        chunk.extend([0xff; 4]);
        chunk.push(marker);
        assert!(matches!(
            vp9_packet_color(&chunk),
            Err(BitstreamError::InvalidField {
                field: "frame_sizes",
                ..
            })
        ));
    }
}
