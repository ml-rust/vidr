//! What the NAL unit walk asks of the code that decodes pictures. One form decodes on the
//! calling thread. The other hands each picture to a worker pool.

use crate::bitstream::hevc_pps::PicParameterSet;
use crate::bitstream::hevc_slice::SliceSegmentHeader;
use crate::bitstream::hevc_sps_full::SeqParameterSet;
use crate::decode::cpu::dsp::sample::Sample;

use super::error::HevcError;
use super::hash::picture_hash::PictureHash;
use super::picture_decoder::PictureBegin;

/// The NAL unit that a picture error belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum UnitSite {
    /// One NAL unit of the packet at `pts`.
    Nal {
        /// The size of the NAL unit.
        bytes: usize,
        /// The pts of its packet.
        pts: i64,
    },
    /// The end of the stream.
    End,
}

/// Decodes pictures of one sample type, as the NAL unit walk drives them.
pub(crate) trait PictureStage<S: Sample> {
    /// What leaves in output order.
    type Output;

    /// True while a picture has slices and no filters yet.
    fn is_active(&self) -> bool;

    /// Starts a picture from its first slice segment.
    ///
    /// # Errors
    ///
    /// Any error of the buffer steps, the checks of the parameter sets, or an allocation.
    fn begin(
        &mut self,
        sps: &SeqParameterSet,
        pps: &PicParameterSet,
        header: &SliceSegmentHeader,
        begin: PictureBegin,
    ) -> Result<(), HevcError>;

    /// Takes one slice segment of the active picture. A call with no active picture does
    /// nothing. `data` is the NAL unit without emulation prevention bytes.
    ///
    /// # Errors
    ///
    /// Any error of the reference lists, and on the calling thread any error of the slice
    /// data.
    fn decode_slice(&mut self, header: &SliceSegmentHeader, data: Vec<u8>)
    -> Result<(), HevcError>;

    /// Records the hash the picture has to match.
    fn set_hash(&mut self, hash: PictureHash);

    /// Ends the active picture.
    ///
    /// # Errors
    ///
    /// Any error of the end of a picture, on the calling thread. A worker pool reports it
    /// in the picture's output order place instead.
    fn finish(&mut self, check_hash: bool) -> Result<(), HevcError>;

    /// Drops the active picture after an error.
    fn abandon(&mut self);

    /// Outputs every waiting picture, at the end of the stream.
    fn flush_output(&mut self);

    /// Empties the buffer when the stream moves to the other sample type. With `discard`,
    /// waiting pictures leave with no output.
    fn close_sequence(&mut self, discard: bool);

    /// The oldest output.
    fn pop_output(&mut self) -> Option<Self::Output>;

    /// Names the unit that later errors of this stage belong to.
    fn enter_unit(&mut self, _site: UnitSite) {}
}
