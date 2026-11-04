//! Encoded video timestamps on the output clock, in integers.
//!
//! The encoder's time base can differ from the output's. MP4 stores it exactly.
//! Matroska rounds each time to its `TimestampScale`. Both count from the first
//! packet, whose pts is the earliest, so [`check_output`], which compares pts
//! relative to the first frame, sees what the output stores.
//!
//! [`check_output`]: crate::check::check_output

use crate::encode::EncodedPacket;
use crate::error::{MediaError, TimeError};

fn packet_error(packet: &EncodedPacket, reason: &'static str) -> MediaError {
    MediaError::VideoPacket {
        pts: packet.pts,
        dts: packet.dts,
        duration: packet.duration,
        reason,
    }
}

fn time_error(what: &'static str) -> impl Fn(TimeError) -> MediaError {
    move |error| MediaError::Time { what, error }
}

const NOT_FIRST: &str = "its pts comes before the pts of the first packet. Send the first \
                         displayed frame first.";
const NO_DURATION: &str = "its duration is not above zero. Pass the encoder a positive duration.";

#[cfg(feature = "mp4")]
pub(super) use mp4_clock::{Mp4Clock, Mp4Placed};

#[cfg(feature = "mp4")]
mod mp4_clock {
    use containr::{EditEntry, EditList};

    use super::{NO_DURATION, NOT_FIRST, packet_error, time_error};
    use crate::encode::EncodedPacket;
    use crate::error::MediaError;
    use crate::media::VideoOutput;
    use crate::time::{Rational, Rounding};

    const OVERFLOW: &str = "its timestamps overflow the MP4 clock. Use a coarser time base.";

    /// Where one packet lands in the MP4 track, in track ticks.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(in crate::media) struct Mp4Placed {
        pub(in crate::media) pts: i64,
        pub(in crate::media) dts: i64,
        pub(in crate::media) duration: u64,
    }

    /// The MP4 video clock.
    ///
    /// A track tick is `1 / time_base.den` s, so every value stays exact. The
    /// first dts stores as 0, as MP4 requires. A B-frame delay becomes the edit
    /// list's media time, and the start offset its empty edit.
    #[derive(Clone, Copy, Debug)]
    pub(in crate::media) struct Mp4Clock {
        factor: i64,
        timescale: u32,
        movie_timescale: u32,
        /// The empty edit, in movie ticks.
        empty: u64,
        /// The first packet's pts and dts.
        first: Option<(i64, i64)>,
        /// The largest presentation end, relative to the first pts, in the
        /// encoder's time base.
        end: i64,
    }

    impl Mp4Clock {
        /// The clock for `video`. The start offset rounds half-even to the
        /// movie clock, so the empty edit is within half a movie tick of it.
        pub(in crate::media) fn new(
            video: &VideoOutput,
            movie_timescale: u32,
        ) -> Result<Self, MediaError> {
            let timescale =
                u32::try_from(video.time_base.den()).map_err(|_| MediaError::VideoOutput {
                    field: "time_base",
                    value: video.time_base.to_string(),
                    reason: "MP4 counts in a 32-bit timescale. Use a time base denominator \
                             below 2^32.",
                })?;
            let movie = Rational::new(1, u64::from(movie_timescale))
                .map_err(time_error("the movie timescale"))?;
            let empty = video
                .start_offset
                .to_ticks(movie, Rounding::HalfEven)
                .map_err(time_error("the video start offset"))?;
            Ok(Self {
                factor: video.time_base.num(),
                timescale,
                movie_timescale,
                empty: empty.unsigned_abs(),
                first: None,
                end: 0,
            })
        }

        /// The track ticks per second.
        pub(in crate::media) const fn timescale(&self) -> u32 {
            self.timescale
        }

        /// Places `packet`, and records where it ends.
        pub(in crate::media) fn place(
            &mut self,
            packet: &EncodedPacket,
        ) -> Result<Mp4Placed, MediaError> {
            if packet.duration <= 0 {
                return Err(packet_error(packet, NO_DURATION));
            }
            let (first_pts, first_dts) = *self.first.get_or_insert((packet.pts, packet.dts));
            let overflow = || packet_error(packet, OVERFLOW);
            let relative = packet.pts.checked_sub(first_pts).ok_or_else(overflow)?;
            if relative < 0 {
                return Err(packet_error(packet, NOT_FIRST));
            }
            let end = relative.checked_add(packet.duration).ok_or_else(overflow)?;
            self.end = self.end.max(end);
            let stored = |value: i64, origin: i64| {
                value
                    .checked_sub(origin)
                    .and_then(|ticks| ticks.checked_mul(self.factor))
                    .ok_or_else(overflow)
            };
            Ok(Mp4Placed {
                pts: stored(packet.pts, first_dts)?,
                dts: stored(packet.dts, first_dts)?,
                duration: stored(packet.duration, 0)?.unsigned_abs(),
            })
        }

        /// The edit list the track needs, once every packet is placed. `None`
        /// when the first frame presents at 0 with no delay.
        ///
        /// The media edit rounds up to the movie clock, as FFmpeg's does, so
        /// it never cuts the last frame.
        pub(in crate::media) fn edit_list(&self) -> Option<EditList> {
            let (first_pts, first_dts) = self.first?;
            let media_time = first_pts
                .saturating_sub(first_dts)
                .saturating_mul(self.factor);
            if media_time == 0 && self.empty == 0 {
                return None;
            }
            let span = u128::from(self.end.unsigned_abs()) * u128::from(self.factor.unsigned_abs());
            let segment = (span * u128::from(self.movie_timescale))
                .div_ceil(u128::from(self.timescale.max(1)));
            let mut entries = Vec::with_capacity(2);
            if self.empty > 0 {
                entries.push(EditEntry {
                    segment_duration: self.empty,
                    media_time: EditEntry::EMPTY_MEDIA_TIME,
                    media_rate: EditEntry::NORMAL_RATE,
                });
            }
            entries.push(EditEntry {
                segment_duration: u64::try_from(segment).unwrap_or(u64::MAX),
                media_time,
                media_rate: EditEntry::NORMAL_RATE,
            });
            Some(EditList {
                movie_timescale: self.movie_timescale,
                entries,
            })
        }
    }
}

#[cfg(feature = "mkv")]
pub(super) use mkv_clock::MkvClock;

#[cfg(feature = "mkv")]
mod mkv_clock {
    use std::collections::BTreeSet;

    use super::{NO_DURATION, NOT_FIRST, packet_error, time_error};
    use crate::encode::EncodedPacket;
    use crate::error::MediaError;
    use crate::media::VideoOutput;
    use crate::time::{Rational, Rounding, checked_tick_add, checked_tick_sub, rescale};

    const COLLISION: &str = "its pts rounds onto the pts of another frame at the Matroska clock. \
                             Write MP4 output (`.mp4`), whose clock keeps them apart.";

    /// The Matroska video clock.
    ///
    /// Each pts is `base + round(pts - first pts)`, half-even, in output ticks.
    /// `base` is the start offset, rounded half-even.
    ///
    /// Relative to the first frame, a pts equals the source pts rescaled
    /// half-even, the exact value `check_output` compares. On the absolute
    /// time line a pts is within one tick of exact: half a tick from `base`,
    /// and half a tick of its own. A duration is the distance between two
    /// rounded times, so the track's total stays within half a tick, inside
    /// [`DURATION_TOLERANCE`](crate::check::DURATION_TOLERANCE).
    #[derive(Clone, Debug)]
    pub(in crate::media) struct MkvClock {
        time_base: Rational,
        output_base: Rational,
        base: i64,
        first_pts: Option<i64>,
        /// Recent output pts. A later packet presents at or after its dts, so
        /// a pts below the newest dts leaves the set, which holds the
        /// reorder depth at most.
        pub(super) recent: BTreeSet<i64>,
    }

    impl MkvClock {
        /// The clock for `video` at `TimestampScale` `scale` nanoseconds.
        pub(in crate::media) fn new(video: &VideoOutput, scale: u64) -> Result<Self, MediaError> {
            let scale = i64::try_from(scale).map_err(|_| MediaError::VideoOutput {
                field: "time_base",
                value: scale.to_string(),
                reason: "the `TimestampScale` does not fit 63 bits. Report this as a vidr bug.",
            })?;
            let output_base =
                Rational::new(scale, 1_000_000_000).map_err(time_error("the Matroska clock"))?;
            let base = video
                .start_offset
                .to_ticks(output_base, Rounding::HalfEven)
                .map_err(time_error("the video start offset"))?;
            Ok(Self {
                time_base: video.time_base,
                output_base,
                base,
                first_pts: None,
                recent: BTreeSet::new(),
            })
        }

        /// True once a packet is placed.
        pub(in crate::media) const fn started(&self) -> bool {
            self.first_pts.is_some()
        }

        /// One output tick, in seconds.
        pub(in crate::media) const fn output_base(&self) -> Rational {
            self.output_base
        }

        /// The output pts and duration of `packet`, in output ticks.
        pub(in crate::media) fn place(
            &mut self,
            packet: &EncodedPacket,
        ) -> Result<(i64, u64), MediaError> {
            if packet.duration <= 0 {
                return Err(packet_error(packet, NO_DURATION));
            }
            let first_pts = *self.first_pts.get_or_insert(packet.pts);
            let relative =
                checked_tick_sub(packet.pts, first_pts).map_err(time_error("the video pts"))?;
            if relative < 0 {
                return Err(packet_error(packet, NOT_FIRST));
            }
            let end = checked_tick_add(relative, packet.duration)
                .map_err(time_error("the video frame end"))?;
            let start_ticks = self.to_output(relative, Rounding::HalfEven)?;
            let end_ticks = self.to_output(end, Rounding::HalfEven)?;
            let pts =
                checked_tick_add(self.base, start_ticks).map_err(time_error("the video pts"))?;
            if !self.recent.insert(pts) {
                return Err(packet_error(packet, COLLISION));
            }
            let dts =
                checked_tick_sub(packet.dts, first_pts).map_err(time_error("the video dts"))?;
            let floor = checked_tick_add(self.base, self.to_output(dts, Rounding::Floor)?)
                .map_err(time_error("the video dts"))?;
            while self.recent.first().is_some_and(|&oldest| oldest < floor) {
                self.recent.pop_first();
            }
            Ok((pts, end_ticks.saturating_sub(start_ticks).unsigned_abs()))
        }

        fn to_output(&self, ticks: i64, rounding: Rounding) -> Result<i64, MediaError> {
            rescale(ticks, self.time_base, self.output_base, rounding)
                .map_err(time_error("the video time on the Matroska clock"))
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::encode::{EncodeCodec, EncodedPacket};
    use crate::media::VideoOutput;
    use crate::time::Rational;

    fn r(num: i64, den: u64) -> Rational {
        Rational::new(num, den).unwrap()
    }

    fn video(time_base: Rational, start_offset: Rational) -> VideoOutput {
        let mut video = VideoOutput::new(EncodeCodec::H264, vec![1], 64, 48, time_base);
        video.start_offset = start_offset;
        video
    }

    fn packet(pts: i64, dts: i64, duration: i64) -> EncodedPacket {
        EncodedPacket {
            data: Vec::new(),
            pts,
            dts,
            duration,
            keyframe: false,
        }
    }

    #[cfg(feature = "mp4")]
    mod mp4 {
        use super::super::{Mp4Clock, Mp4Placed};
        use super::*;
        use containr::EditEntry;

        #[test]
        fn no_delay_and_no_offset_need_no_edit() {
            let mut clock = Mp4Clock::new(&video(r(1, 1000), Rational::ZERO), 1000).unwrap();
            let placed = clock.place(&packet(0, 0, 40)).unwrap();
            assert_eq!(
                placed,
                Mp4Placed {
                    pts: 0,
                    dts: 0,
                    duration: 40
                }
            );
            assert_eq!(clock.edit_list(), None);
        }

        #[test]
        fn a_b_frame_delay_becomes_the_media_time_and_the_offset_an_empty_edit() {
            let mut clock = Mp4Clock::new(&video(r(1, 12800), r(1, 2)), 1000).unwrap();
            // Decode order: 0, 3, 1, 2. The first dts sits two frames back.
            let frames = [(0, -1024), (1536, -512), (512, 0), (1024, 512)];
            let placed: Vec<_> = frames
                .iter()
                .map(|&(pts, dts)| clock.place(&packet(pts, dts, 512)).unwrap())
                .collect();
            assert_eq!((placed[0].pts, placed[0].dts), (1024, 0));
            assert_eq!((placed[1].pts, placed[1].dts), (2560, 512));
            let edits = clock.edit_list().unwrap();
            assert_eq!(edits.movie_timescale, 1000);
            assert_eq!(edits.entries[0].segment_duration, 500);
            assert_eq!(edits.entries[0].media_time, EditEntry::EMPTY_MEDIA_TIME);
            // Four frames last 2048 / 12800 s, which is 160 ms exactly.
            assert_eq!(edits.entries[1].segment_duration, 160);
            assert_eq!(edits.entries[1].media_time, 1024);
        }

        #[test]
        fn a_time_base_with_a_numerator_stays_exact() {
            let mut clock = Mp4Clock::new(&video(r(1001, 30000), Rational::ZERO), 1000).unwrap();
            clock.place(&packet(0, 0, 1)).unwrap();
            let placed = clock.place(&packet(1, 1, 1)).unwrap();
            assert_eq!(placed.pts, 1001);
            assert_eq!(placed.duration, 1001);
            assert_eq!(clock.timescale(), 30000);
        }

        #[test]
        fn a_pts_before_the_first_is_refused() {
            let mut clock = Mp4Clock::new(&video(r(1, 1000), Rational::ZERO), 1000).unwrap();
            clock.place(&packet(40, 40, 40)).unwrap();
            assert!(matches!(
                clock.place(&packet(0, 80, 40)),
                Err(crate::MediaError::VideoPacket { pts: 0, .. })
            ));
        }
    }

    #[cfg(feature = "mkv")]
    mod mkv {
        use super::super::MkvClock;
        use super::*;
        use crate::time::{Rounding, rescale};

        const MS: u64 = 1_000_000;

        #[test]
        fn relative_pts_equal_the_half_even_rescale_check_output_compares() {
            let time_base = r(1, 30000);
            let offset = r(3, 7);
            let mut clock = MkvClock::new(&video(time_base, offset), MS).unwrap();
            let first = 1001;
            let mut origin = None;
            for frame in 0..1000 {
                let pts = first + frame * 1001;
                let (out, _) = clock.place(&packet(pts, pts, 1001)).unwrap();
                let origin = *origin.get_or_insert(out);
                let expected =
                    rescale(pts - first, time_base, r(1, 1000), Rounding::HalfEven).unwrap();
                assert_eq!(out - origin, expected, "frame {frame}");
            }
        }

        #[test]
        fn durations_sum_to_the_rounded_end() {
            let mut clock = MkvClock::new(&video(r(1, 30000), Rational::ZERO), MS).unwrap();
            let mut total = 0;
            for frame in 0..1000 {
                let pts = frame * 1001;
                let (out, duration) = clock.place(&packet(pts, pts, 1001)).unwrap();
                assert_eq!(out, i64::try_from(total).unwrap(), "frame {frame}");
                total += duration;
            }
            // 1000 frames of 1001 / 30000 s are 33 366.67 ms.
            assert_eq!(total, 33367);
        }

        #[test]
        fn two_frames_on_one_tick_are_refused_by_name() {
            let mut clock = MkvClock::new(&video(r(1, 10000), Rational::ZERO), MS).unwrap();
            clock.place(&packet(0, 0, 1)).unwrap();
            let error = clock.place(&packet(1, 1, 1)).unwrap_err();
            assert!(error.to_string().contains("MP4"), "{error}");
        }

        #[test]
        fn the_recent_set_holds_the_reorder_depth() {
            let mut clock = MkvClock::new(&video(r(1, 1000), Rational::ZERO), MS).unwrap();
            for frame in 0..10_000 {
                let pts = frame * 40;
                clock.place(&packet(pts + 80, pts, 40)).unwrap();
                assert!(clock.recent.len() <= 4, "{}", clock.recent.len());
            }
        }
    }
}
