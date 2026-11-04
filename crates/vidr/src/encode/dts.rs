//! Decode timestamps for packets that come out in decode order.
//!
//! An encoder such as NVENC reports the pts of each picture and no dts. With `b`
//! consecutive B-frames, the `k`th picture in decode order has at most `b` pictures before
//! it that display after it. So its display index is at least `k - b`, and the
//! `(k - b)`th input pts never exceeds its pts. That value is its dts. The first
//! `b` packets have no such input, so they count back from the first pts in
//! units of the first duration, and their dts precede the first pts.
//!
//! Input pts increase strictly, so dts increase strictly. All arithmetic is
//! exact integer arithmetic.

use std::collections::VecDeque;

use crate::error::EncodeError;

/// The dts of each packet of one stream.
#[derive(Clone, Debug)]
pub(super) struct DtsClock {
    /// The reorder delay in frames: the config's `b_frames`.
    delay: u32,
    /// The first frame's `(pts, duration)`.
    first: Option<(i64, i64)>,
    /// Input pts the output has not used yet.
    inputs: VecDeque<i64>,
    /// Packets given a dts so far.
    emitted: u64,
}

impl DtsClock {
    pub(super) const fn new(delay: u32) -> Self {
        Self {
            delay,
            first: None,
            inputs: VecDeque::new(),
            emitted: 0,
        }
    }

    /// Records a frame sent to the encoder.
    pub(super) fn push(&mut self, pts: i64, duration: i64) {
        self.first.get_or_insert((pts, duration));
        self.inputs.push_back(pts);
    }

    /// The dts of the next packet in decode order, whose pts is `pts`.
    pub(super) fn next(&mut self, pts: i64) -> Result<i64, EncodeError> {
        let dts = if self.emitted < u64::from(self.delay) {
            let (first_pts, first_duration) =
                self.first.ok_or(EncodeError::UnexpectedOutput { pts })?;
            let frames = i64::from(self.delay) - self.emitted as i64;
            frames
                .checked_mul(first_duration)
                .and_then(|back| first_pts.checked_sub(back))
                .ok_or(EncodeError::TimestampOverflow {
                    pts: first_pts,
                    duration: first_duration,
                    delay: self.delay,
                })?
        } else {
            self.inputs
                .pop_front()
                .ok_or(EncodeError::UnexpectedOutput { pts })?
        };
        if dts > pts {
            return Err(EncodeError::DtsAfterPts {
                dts,
                pts,
                delay: self.delay,
            });
        }
        self.emitted += 1;
        Ok(dts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(delay: u32, inputs: &[i64], outputs: &[i64]) -> Vec<i64> {
        let mut clock = DtsClock::new(delay);
        for &pts in inputs {
            clock.push(pts, 1001);
        }
        outputs
            .iter()
            .map(|&pts| clock.next(pts).unwrap_or(i64::MIN))
            .collect()
    }

    #[test]
    fn without_b_frames_dts_equals_pts() {
        assert_eq!(run(0, &[7, 1008, 2009], &[7, 1008, 2009]), [7, 1008, 2009]);
    }

    /// The order NVENC gives on the RTX 3060 at `frameIntervalP` 3 with the
    /// middle B-frame as a reference: I0 P3 B2 B1 P6 B5 B4.
    #[test]
    fn two_b_frames_precede_the_first_pts_then_follow_the_inputs() {
        let inputs: Vec<i64> = (0..7).map(|frame| 7 + frame * 1001).collect();
        let order = [0, 3, 2, 1, 6, 5, 4];
        let outputs: Vec<i64> = order.iter().map(|&frame| inputs[frame]).collect();
        let dts = run(2, &inputs, &outputs);
        assert_eq!(
            dts,
            [
                7 - 2002,
                7 - 1001,
                inputs[0],
                inputs[1],
                inputs[2],
                inputs[3],
                inputs[4],
            ]
        );
        assert!(dts.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(dts.iter().zip(&outputs).all(|(dts, pts)| dts <= pts));
    }

    #[test]
    fn variable_frame_rate_stays_exact() {
        let inputs = [0, 10, 25, 26, 100];
        let order = [0, 25, 10, 100, 26];
        assert_eq!(run(1, &inputs, &order), [-1001, 0, 10, 25, 26]);
    }

    #[test]
    fn a_reorder_deeper_than_the_delay_is_refused() {
        let mut clock = DtsClock::new(0);
        clock.push(0, 1);
        clock.push(10, 1);
        assert_eq!(clock.next(0), Ok(0));
        clock.push(5, 1);
        assert_eq!(
            clock.next(5),
            Err(EncodeError::DtsAfterPts {
                dts: 10,
                pts: 5,
                delay: 0,
            })
        );
    }

    #[test]
    fn output_without_input_is_refused() {
        let mut clock = DtsClock::new(0);
        assert_eq!(clock.next(3), Err(EncodeError::UnexpectedOutput { pts: 3 }));
        let mut delayed = DtsClock::new(2);
        assert_eq!(
            delayed.next(3),
            Err(EncodeError::UnexpectedOutput { pts: 3 })
        );
    }

    #[test]
    fn overflow_is_an_error() {
        let mut clock = DtsClock::new(2);
        clock.push(i64::MIN, 1);
        assert_eq!(
            clock.next(i64::MIN),
            Err(EncodeError::TimestampOverflow {
                pts: i64::MIN,
                duration: 1,
                delay: 2,
            })
        );
    }
}
