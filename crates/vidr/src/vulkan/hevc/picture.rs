//! The slice segments of one picture, in the form Vulkan Video reads.

use crate::bitstream::hevc_slice::SliceSegmentHeader;

const START_CODE: [u8; 3] = [0, 0, 1];

/// One picture's slice segments, gathered from one packet.
pub(super) struct Picture<'u> {
    /// The header of the first slice segment.
    pub(super) header: SliceSegmentHeader,
    pub(super) pts: i64,
    /// Each segment with its start code, one after another.
    pub(super) parts: Vec<&'u [u8]>,
    /// Where each segment starts in the bytes of `parts`.
    pub(super) offsets: Vec<u32>,
    bytes: u32,
}

impl<'u> Picture<'u> {
    pub(super) const fn new(header: SliceSegmentHeader, pts: i64) -> Self {
        Self {
            header,
            pts,
            parts: Vec::new(),
            offsets: Vec::new(),
            bytes: 0,
        }
    }

    pub(super) fn push_segment(&mut self, nal: &'u [u8]) {
        self.offsets.push(self.bytes);
        self.bytes += (START_CODE.len() + nal.len()) as u32;
        self.parts.push(&START_CODE);
        self.parts.push(nal);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_offset_points_at_a_start_code() {
        let mut picture = Picture::new(SliceSegmentHeader::default(), 3);
        picture.push_segment(&[1, 2, 3, 4]);
        picture.push_segment(&[5]);
        assert_eq!(picture.offsets, [0, 7]);
        let joined: Vec<u8> = picture.parts.concat();
        assert_eq!(joined, [0, 0, 1, 1, 2, 3, 4, 0, 0, 1, 5]);
    }
}
