//! MP4-style edit lists, applied to a track's frames.
//!
//! The input is container-agnostic: one [`EditEntry`] per edit, with the
//! segment duration in the movie time base and the media time in the media
//! time base. Leading empty edits give a start offset. One media edit trims
//! the media. Every edit must play at rate 1.

use super::rational::Rational;
use super::rounding::{Rounding, rescale};
use super::ticks::{checked_tick_add, checked_tick_sub};
use super::timing::{FrameTiming, presentation_order_violations};
use crate::error::{EditListError, TimeError, TimingError};

/// One edit, as an MP4 `elst` entry holds it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EditEntry {
    /// Length of the edit on the presentation timeline, in movie ticks. Zero
    /// on a media edit means the edit runs to the end of the media.
    pub segment_duration: i64,
    /// First media tick the edit plays, or [`EditEntry::EMPTY_MEDIA_TIME`]
    /// for an empty edit.
    pub media_time: i64,
    /// Playback rate. Only `1/1` is supported.
    pub rate: Rational,
}

impl EditEntry {
    /// The media time of an empty edit, which shows nothing for its duration.
    pub const EMPTY_MEDIA_TIME: i64 = -1;

    /// Creates an empty edit of `segment_duration` movie ticks at rate 1.
    pub const fn empty(segment_duration: i64) -> Self {
        Self {
            segment_duration,
            media_time: Self::EMPTY_MEDIA_TIME,
            rate: Rational::ONE,
        }
    }

    /// Creates a media edit at rate 1.
    pub const fn media(segment_duration: i64, media_time: i64) -> Self {
        Self {
            segment_duration,
            media_time,
            rate: Rational::ONE,
        }
    }

    /// Returns `true` for an empty edit.
    pub const fn is_empty_edit(&self) -> bool {
        self.media_time == Self::EMPTY_MEDIA_TIME
    }
}

/// The presentation timeline an edit list gives.
///
/// Frames whose media pts fall in `media_start..media_end` play, shifted so
/// `media_start` becomes pts 0. The track starts `start_offset` seconds into
/// the presentation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EditTimeline {
    start_offset: Rational,
    media_start: i64,
    media_end: Option<i64>,
}

impl EditTimeline {
    /// The timeline of a track without an edit list: no offset, no trim.
    pub const IDENTITY: Self = Self {
        start_offset: Rational::ZERO,
        media_start: 0,
        media_end: None,
    };

    /// Resolves an edit list.
    ///
    /// Supported layout: any number of leading empty edits, then at most one
    /// media edit. An empty list gives [`EditTimeline::IDENTITY`].
    ///
    /// # Errors
    ///
    /// - [`EditListError::UnsupportedRate`] for a rate other than 1.
    /// - [`EditListError::NegativeSegmentDuration`] for a negative duration.
    /// - [`EditListError::InvalidMediaTime`] for a media time below -1.
    /// - [`EditListError::SegmentAfterMedia`] for any edit after the media edit.
    /// - [`EditListError::NoMediaSegment`] for a list of empty edits only.
    /// - [`EditListError::Time`] for a time base of zero or less, or overflow.
    pub fn resolve(
        entries: &[EditEntry],
        movie_time_base: Rational,
        media_time_base: Rational,
    ) -> Result<Self, EditListError> {
        for time_base in [movie_time_base, media_time_base] {
            if !time_base.is_positive() {
                return Err(TimeError::NonPositiveTimeBase { time_base }.into());
            }
        }
        let mut start_offset = Rational::ZERO;
        let mut media: Option<(usize, EditEntry)> = None;
        for (index, entry) in entries.iter().copied().enumerate() {
            if entry.rate != Rational::ONE {
                return Err(EditListError::UnsupportedRate {
                    index,
                    rate: entry.rate,
                });
            }
            if entry.segment_duration < 0 {
                return Err(EditListError::NegativeSegmentDuration {
                    index,
                    segment_duration: entry.segment_duration,
                });
            }
            if entry.media_time < EditEntry::EMPTY_MEDIA_TIME {
                return Err(EditListError::InvalidMediaTime {
                    index,
                    media_time: entry.media_time,
                });
            }
            if let Some((media_index, _)) = media {
                return Err(EditListError::SegmentAfterMedia { index, media_index });
            }
            if entry.is_empty_edit() {
                let gap = Rational::from_ticks(entry.segment_duration, movie_time_base)?;
                start_offset = start_offset.checked_add(gap)?;
            } else {
                media = Some((index, entry));
            }
        }
        let Some((_, entry)) = media else {
            return if entries.is_empty() {
                Ok(Self::IDENTITY)
            } else {
                Err(EditListError::NoMediaSegment {
                    entries: entries.len(),
                })
            };
        };
        let media_end = if entry.segment_duration == 0 {
            None
        } else {
            // Ceil: an integer pts lies below the exact segment end exactly
            // when it lies below the end rounded up, so no frame moves across
            // the boundary.
            let length = rescale(
                entry.segment_duration,
                movie_time_base,
                media_time_base,
                Rounding::Ceil,
            )?;
            Some(checked_tick_add(entry.media_time, length)?)
        };
        Ok(Self {
            start_offset,
            media_start: entry.media_time,
            media_end,
        })
    }

    /// Seconds of presentation before the first frame, from empty edits.
    pub const fn start_offset(&self) -> Rational {
        self.start_offset
    }

    /// First media tick the timeline plays.
    pub const fn media_start(&self) -> i64 {
        self.media_start
    }

    /// Media tick where the timeline ends, exclusive. `None` runs to the end
    /// of the media.
    pub const fn media_end(&self) -> Option<i64> {
        self.media_end
    }

    /// Applies the timeline to frames in presentation order, in media ticks.
    ///
    /// - The frame on screen at `media_start` plays first. When it starts
    ///   earlier, its pts moves to `media_start` and its known duration
    ///   shrinks by the cut.
    /// - Frames from `media_end` on are dropped. The last frame's known
    ///   duration shrinks to end at `media_end`.
    /// - Every pts and dts shifts by `-media_start`, so playback starts at 0.
    ///
    /// # Errors
    ///
    /// - [`EditListError::Timing`] when pts do not strictly increase.
    /// - [`EditListError::NoPresentedFrames`] when no frame falls in range.
    /// - [`EditListError::Time`] on overflow.
    pub fn present(&self, frames: &[FrameTiming]) -> Result<Vec<FrameTiming>, EditListError> {
        if let Some(violation) = presentation_order_violations(frames).next() {
            return Err(TimingError::from(violation).into());
        }
        let none = EditListError::NoPresentedFrames {
            media_start: self.media_start,
            media_end: self.media_end,
            frames: frames.len(),
        };
        // The last frame whose pts does not exceed the start is on screen there.
        let mut first = frames
            .partition_point(|frame| frame.pts <= self.media_start)
            .saturating_sub(1);
        let end = self.media_end.map_or(frames.len(), |end| {
            frames.partition_point(|frame| frame.pts < end)
        });
        // The final frame of the media is on screen only until its known
        // duration ends.
        if first + 1 == frames.len()
            && let Some(frame) = frames.get(first)
            && frame.pts < self.media_start
            && frame.duration > 0
            && checked_tick_sub(self.media_start, frame.pts)? >= frame.duration
        {
            first += 1;
        }
        let mut presented = frames
            .get(first..end)
            .filter(|range| !range.is_empty())
            .ok_or(none)?
            .to_vec();
        if let Some(head) = presented.first_mut()
            && head.pts < self.media_start
        {
            let cut = checked_tick_sub(self.media_start, head.pts)?;
            head.pts = self.media_start;
            if head.duration > 0 {
                head.duration = head.duration.saturating_sub(cut).max(0);
            }
        }
        if let (Some(end), Some(tail)) = (self.media_end, presented.last_mut())
            && tail.duration > 0
        {
            tail.duration = tail.duration.min(checked_tick_sub(end, tail.pts)?);
        }
        for frame in &mut presented {
            frame.pts = checked_tick_sub(frame.pts, self.media_start)?;
            frame.dts = frame
                .dts
                .map(|dts| checked_tick_sub(dts, self.media_start))
                .transpose()?;
        }
        Ok(presented)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(num: i64, den: u64) -> Rational {
        Rational::new(num, den).unwrap()
    }

    fn frames(pts: &[i64], duration: i64) -> Vec<FrameTiming> {
        pts.iter()
            .map(|&pts| FrameTiming::new(pts, Some(pts), duration))
            .collect()
    }

    #[test]
    fn empty_list_is_identity() {
        let timeline = EditTimeline::resolve(&[], r(1, 600), r(1, 90000)).unwrap();
        assert_eq!(timeline, EditTimeline::IDENTITY);
        let list = frames(&[0, 3003], 3003);
        assert_eq!(timeline.present(&list).unwrap(), list);
    }

    #[test]
    fn empty_edits_sum_into_start_offset() {
        let entries = [
            EditEntry::empty(300),
            EditEntry::empty(300),
            EditEntry::media(0, 0),
        ];
        let timeline = EditTimeline::resolve(&entries, r(1, 600), r(1, 90000)).unwrap();
        assert_eq!(timeline.start_offset(), Rational::ONE);
        assert_eq!(timeline.media_end(), None);
    }

    #[test]
    fn segment_end_rounds_up_into_media_ticks() {
        // 100 ticks of 1/600 s are 166.67 ticks of 1/1000 s, which round up to 167.
        let entries = [EditEntry::media(100, 10)];
        let timeline = EditTimeline::resolve(&entries, r(1, 600), r(1, 1000)).unwrap();
        assert_eq!(timeline.media_end(), Some(177));
    }

    #[test]
    fn trim_cuts_the_covering_frame() {
        let timeline = EditTimeline {
            start_offset: Rational::ZERO,
            media_start: 50,
            media_end: Some(130),
        };
        let presented = timeline
            .present(&frames(&[0, 40, 80, 120, 160], 40))
            .unwrap();
        let pts: Vec<_> = presented.iter().map(|frame| frame.pts).collect();
        assert_eq!(pts, [0, 30, 70]);
        assert_eq!(presented[0].duration, 30);
        assert_eq!(presented[0].dts, Some(-10));
        assert_eq!(presented[2].duration, 10);
    }

    #[test]
    fn edit_past_the_last_frame_presents_nothing() {
        let timeline = EditTimeline {
            start_offset: Rational::ZERO,
            media_start: 100,
            media_end: None,
        };
        assert_eq!(
            timeline.present(&frames(&[0, 40], 40)),
            Err(EditListError::NoPresentedFrames {
                media_start: 100,
                media_end: None,
                frames: 2,
            })
        );
    }

    #[test]
    fn edit_after_media_is_refused() {
        let entries = [EditEntry::media(10, 0), EditEntry::empty(5)];
        assert_eq!(
            EditTimeline::resolve(&entries, r(1, 600), r(1, 600)),
            Err(EditListError::SegmentAfterMedia {
                index: 1,
                media_index: 0,
            })
        );
    }

    #[test]
    fn only_empty_edits_are_refused() {
        assert_eq!(
            EditTimeline::resolve(&[EditEntry::empty(5)], r(1, 600), r(1, 600)),
            Err(EditListError::NoMediaSegment { entries: 1 })
        );
    }

    #[test]
    fn bad_fields_are_refused() {
        let base = r(1, 600);
        let negative = [EditEntry::media(-1, 0)];
        assert!(matches!(
            EditTimeline::resolve(&negative, base, base),
            Err(EditListError::NegativeSegmentDuration { index: 0, .. })
        ));
        let media_time = [EditEntry::media(1, -2)];
        assert!(matches!(
            EditTimeline::resolve(&media_time, base, base),
            Err(EditListError::InvalidMediaTime { index: 0, .. })
        ));
    }
}
