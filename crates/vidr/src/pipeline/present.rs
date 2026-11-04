//! [`EditTimeline::present`] one frame at a time, with each frame's output
//! duration.
//!
//! The rules are those of `present`, and a test compares the two. A frame's
//! fate and its duration both depend on the next frame's pts, so this type
//! holds one frame back. T2 runs it before the transform, so a dropped frame
//! costs no work.

use super::PipelineError;
use crate::error::{EditListError, TimeError, TimingError};
use crate::time::{
    DurationHints, EditTimeline, FrameTiming, Rational, Rounding, StartOffset, TrackTiming,
    checked_tick_sub,
};

/// A frame the timeline keeps, with its output timing.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct Presented<T> {
    pub(super) item: T,
    pub(super) pts: i64,
    pub(super) duration: i64,
}

/// What one pushed frame releases.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Pushed<T> {
    /// Nothing yet: the type holds its first frame.
    Held,
    /// One frame leaves the timeline.
    Dropped,
    /// The frame held before this one, now with its duration.
    Emit(Presented<T>),
}

/// The streaming form of [`EditTimeline::present`].
pub(super) struct Presenter<T> {
    time_base: Rational,
    timeline: Option<EditTimeline>,
    held: Option<(FrameTiming, T)>,
    /// The output timing of the last emitted frame, for the chain's last step.
    previous: Option<FrameTiming>,
    last_pts: Option<i64>,
    seen: usize,
    past_end: bool,
}

impl<T> Presenter<T> {
    pub(super) fn new(time_base: Rational, timeline: Option<EditTimeline>) -> Self {
        Self {
            time_base,
            timeline,
            held: None,
            previous: None,
            last_pts: None,
            seen: 0,
            past_end: false,
        }
    }

    /// Takes the next frame in presentation order. `timing.duration` is the
    /// duration of the frame's own packet.
    pub(super) fn push(&mut self, timing: FrameTiming, item: T) -> Result<Pushed<T>, TimingError> {
        let index = self.seen;
        self.seen += 1;
        if let Some(previous) = self.last_pts
            && timing.pts <= previous
        {
            return Err(TimingError::NonIncreasingPts {
                index,
                previous,
                pts: timing.pts,
            });
        }
        self.last_pts = Some(timing.pts);
        if let Some(end) = self.timeline.and_then(|timeline| timeline.media_end())
            && timing.pts >= end
        {
            self.past_end = true;
            return Ok(Pushed::Dropped);
        }
        let Some((held, held_item)) = self.held.replace((timing, item)) else {
            return Ok(Pushed::Held);
        };
        // The last frame at or before the start is on screen there.
        if self
            .timeline
            .is_some_and(|timeline| timing.pts <= timeline.media_start())
        {
            return Ok(Pushed::Dropped);
        }
        let pts = self.output_pts(held.pts)?;
        let duration = checked_tick_sub(self.output_pts(timing.pts)?, pts)?;
        self.previous = Some(FrameTiming::new(pts, None, duration));
        Ok(Pushed::Emit(Presented {
            item: held_item,
            pts,
            duration,
        }))
    }

    /// Ends the stream and returns the last frame, with its duration from the
    /// chain in `time/chain.rs`.
    pub(super) fn finish(mut self, hints: DurationHints) -> Result<Presented<T>, PipelineError> {
        let none = |this: &Self| -> PipelineError {
            match this.timeline {
                Some(timeline) => EditListError::NoPresentedFrames {
                    media_start: timeline.media_start(),
                    media_end: timeline.media_end(),
                    frames: this.seen,
                }
                .into(),
                None => TimingError::NoFrames.into(),
            }
        };
        let Some((timing, item)) = self.held.take() else {
            return Err(none(&self));
        };
        let mut last = FrameTiming::new(timing.pts, None, timing.duration);
        if let Some(timeline) = self.timeline {
            let start = timeline.media_start();
            if last.pts < start {
                let cut = checked_tick_sub(start, last.pts)?;
                // The final frame of the media is on screen only until its
                // known duration ends.
                if !self.past_end && last.duration > 0 && cut >= last.duration {
                    return Err(none(&self));
                }
                last.pts = start;
                if last.duration > 0 {
                    last.duration = last.duration.saturating_sub(cut).max(0);
                }
            }
            if let Some(end) = timeline.media_end()
                && last.duration > 0
            {
                last.duration = last.duration.min(checked_tick_sub(end, last.pts)?);
            }
            last.pts = checked_tick_sub(last.pts, start)?;
        }
        let frames = match self.previous {
            Some(previous) => vec![previous, last],
            None => vec![last],
        };
        let track = TrackTiming::new(self.time_base, StartOffset::default(), frames, hints)?;
        let duration = track.last_duration_ticks(Rounding::HalfEven)?;
        if duration <= 0 {
            return Err(PipelineError::LastDurationBelowTick {
                pts: last.pts,
                duration: track.last_duration(),
                time_base: self.time_base,
            });
        }
        Ok(Presented {
            item,
            pts: last.pts,
            duration,
        })
    }

    /// The output pts of a frame at media tick `pts`.
    fn output_pts(&self, pts: i64) -> Result<i64, TimeError> {
        match self.timeline {
            Some(timeline) => {
                let start = timeline.media_start();
                checked_tick_sub(pts.max(start), start)
            }
            None => Ok(pts),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::EditEntry;

    fn base() -> Rational {
        Rational::new(1, 1000).unwrap()
    }

    /// Runs `frames` through a [`Presenter`] and returns the kept frames' pts,
    /// their durations, and the last duration.
    fn stream(
        timeline: Option<EditTimeline>,
        frames: &[FrameTiming],
        hints: DurationHints,
    ) -> Result<Vec<(i64, i64)>, PipelineError> {
        let mut presenter = Presenter::new(base(), timeline);
        let mut kept = Vec::new();
        for frame in frames {
            if let Pushed::Emit(emitted) = presenter.push(*frame, ())? {
                kept.push((emitted.pts, emitted.duration));
            }
        }
        let last = presenter.finish(hints)?;
        kept.push((last.pts, last.duration));
        Ok(kept)
    }

    /// The same frames through [`EditTimeline::present`] and [`TrackTiming`].
    fn batch(
        timeline: EditTimeline,
        frames: &[FrameTiming],
        hints: DurationHints,
    ) -> Result<(Vec<i64>, i64), PipelineError> {
        let presented = timeline.present(frames)?;
        let pts = presented.iter().map(|frame| frame.pts).collect();
        let track = TrackTiming::new(base(), StartOffset::default(), presented, hints)?;
        Ok((pts, track.last_duration_ticks(Rounding::HalfEven)?))
    }

    #[test]
    fn matches_present_for_every_window() {
        let lists: [&[i64]; 4] = [
            &[100],
            &[0, 100, 200, 300],
            &[50, 100, 250, 300, 310],
            &[-100, 0, 40, 400],
        ];
        for pts in lists {
            for own in [0, 30, 100] {
                let frames: Vec<FrameTiming> = pts
                    .iter()
                    .map(|&pts| FrameTiming::new(pts, None, own))
                    .collect();
                for start in [0, 40, 100, 150, 299, 300, 305, 400, 900] {
                    for length in [0, 1, 60, 100, 160, 1000] {
                        let entry = EditEntry::media(length, start);
                        let timeline = EditTimeline::resolve(&[entry], base(), base()).unwrap();
                        let hints = DurationHints {
                            packet_duration: Some(17),
                            ..DurationHints::default()
                        };
                        let batch = batch(timeline, &frames, hints);
                        let stream = stream(Some(timeline), &frames, hints);
                        let name = format!("pts {pts:?} own {own} start {start} length {length}");
                        match (batch, stream) {
                            (Ok((expected, last)), Ok(kept)) => {
                                let found: Vec<i64> = kept.iter().map(|pair| pair.0).collect();
                                assert_eq!(found, expected, "{name}");
                                assert_eq!(kept.last().map(|pair| pair.1), Some(last), "{name}");
                            }
                            (Err(expected), Err(found)) => assert_eq!(found, expected, "{name}"),
                            (expected, found) => panic!("{name}: {expected:?} and {found:?}"),
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn each_duration_runs_to_the_next_pts() {
        let frames = [
            FrameTiming::new(0, None, 5),
            FrameTiming::new(10, None, 5),
            FrameTiming::new(30, None, 5),
        ];
        let kept = stream(None, &frames, DurationHints::default()).unwrap();
        assert_eq!(kept, vec![(0, 10), (10, 20), (30, 5)]);
    }

    #[test]
    fn no_timeline_keeps_negative_pts() {
        let frames = [
            FrameTiming::new(-20, None, 0),
            FrameTiming::new(-10, None, 0),
        ];
        let kept = stream(None, &frames, DurationHints::default()).unwrap();
        assert_eq!(kept, vec![(-20, 10), (-10, 10)]);
    }

    #[test]
    fn a_pts_that_does_not_increase_is_refused() {
        let mut presenter = Presenter::new(base(), None);
        presenter.push(FrameTiming::new(5, None, 0), ()).unwrap();
        assert_eq!(
            presenter.push(FrameTiming::new(5, None, 0), ()),
            Err(TimingError::NonIncreasingPts {
                index: 1,
                previous: 5,
                pts: 5,
            })
        );
    }
}
