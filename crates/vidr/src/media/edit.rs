//! containr's raw edit list as vidr's [`EditTimeline`].

use containr::{EditList, Timebase, Track};

use crate::error::{EditListError, MediaError, TimeError};
use crate::time::{EditEntry, EditTimeline, Rational};

/// 1.0 in the 16.16 fixed point of `media_rate`.
const RATE_DENOMINATOR: u64 = 0x10000;

pub(super) fn edit_timeline(track: &Track) -> Result<EditTimeline, MediaError> {
    resolve(track.id, &track.edit_list, track.timebase)
}

/// Segment durations count in the movie timescale, media times in the track
/// timebase. A track with no edit list has no movie timescale to read.
fn resolve(track_id: u32, list: &EditList, media: Timebase) -> Result<EditTimeline, MediaError> {
    if list.is_empty() {
        return Ok(EditTimeline::IDENTITY);
    }
    let edit_error = |error: EditListError| MediaError::EditList { track_id, error };
    let time_error = |error: TimeError| edit_error(error.into());
    let mut entries = Vec::with_capacity(list.entries.len());
    for (index, entry) in list.entries.iter().enumerate() {
        let segment_duration =
            i64::try_from(entry.segment_duration).map_err(|_| MediaError::EditDuration {
                track_id,
                index,
                segment_duration: entry.segment_duration,
            })?;
        entries.push(EditEntry {
            segment_duration,
            media_time: entry.media_time,
            rate: Rational::new(i64::from(entry.media_rate), RATE_DENOMINATOR)
                .map_err(time_error)?,
        });
    }
    let movie = Rational::new(1, u64::from(list.movie_timescale)).map_err(time_error)?;
    let media = Rational::new(i64::from(media.num), u64::from(media.den)).map_err(time_error)?;
    EditTimeline::resolve(&entries, movie, media).map_err(edit_error)
}

#[cfg(test)]
mod tests {
    use containr::EditEntry as RawEntry;

    use super::*;

    const MEDIA: Timebase = Timebase { num: 1, den: 12800 };

    fn list(movie_timescale: u32, entries: Vec<RawEntry>) -> EditList {
        EditList {
            movie_timescale,
            entries,
        }
    }

    fn raw(segment_duration: u64, media_time: i64, media_rate: i32) -> RawEntry {
        RawEntry {
            segment_duration,
            media_time,
            media_rate,
        }
    }

    #[test]
    fn no_edit_list_is_the_identity_and_needs_no_timescale() {
        assert_eq!(
            resolve(1, &EditList::default(), MEDIA),
            Ok(EditTimeline::IDENTITY)
        );
    }

    #[test]
    fn movie_ticks_set_the_offset_and_track_ticks_set_the_trim() {
        let edits = list(
            1000,
            vec![
                raw(500, RawEntry::EMPTY_MEDIA_TIME, RawEntry::NORMAL_RATE),
                raw(480, 1024, RawEntry::NORMAL_RATE),
            ],
        );
        let timeline = resolve(1, &edits, MEDIA).unwrap();
        assert_eq!(timeline.start_offset(), Rational::new(1, 2).unwrap());
        assert_eq!(timeline.media_start(), 1024);
        // 480 ms are 6144 ticks of 1/12800 s.
        assert_eq!(timeline.media_end(), Some(1024 + 6144));
    }

    #[test]
    fn a_dwell_and_a_reverse_rate_are_refused() {
        for rate in [0, -0x10000, 0x20000] {
            let edits = list(1000, vec![raw(480, 0, rate)]);
            assert!(
                matches!(
                    resolve(7, &edits, MEDIA),
                    Err(MediaError::EditList {
                        track_id: 7,
                        error: EditListError::UnsupportedRate { index: 0, .. },
                    })
                ),
                "rate {rate}"
            );
        }
    }

    #[test]
    fn a_duration_above_the_signed_range_is_refused() {
        let duration = u64::MAX;
        let edits = list(1000, vec![raw(duration, 0, RawEntry::NORMAL_RATE)]);
        assert_eq!(
            resolve(7, &edits, MEDIA),
            Err(MediaError::EditDuration {
                track_id: 7,
                index: 0,
                segment_duration: duration,
            })
        );
    }

    #[test]
    fn zero_time_bases_are_refused() {
        let edits = list(0, vec![raw(480, 0, RawEntry::NORMAL_RATE)]);
        assert!(matches!(
            resolve(7, &edits, MEDIA),
            Err(MediaError::EditList { track_id: 7, .. })
        ));
        let edits = list(1000, vec![raw(480, 0, RawEntry::NORMAL_RATE)]);
        assert!(matches!(
            resolve(7, &edits, Timebase { num: 1, den: 0 }),
            Err(MediaError::EditList { track_id: 7, .. })
        ));
    }

    #[test]
    fn an_edit_after_the_media_edit_is_refused() {
        let edits = list(
            1000,
            vec![
                raw(480, 0, RawEntry::NORMAL_RATE),
                raw(480, 512, RawEntry::NORMAL_RATE),
            ],
        );
        assert!(matches!(
            resolve(7, &edits, MEDIA),
            Err(MediaError::EditList {
                error: EditListError::SegmentAfterMedia { .. },
                ..
            })
        ));
    }
}
