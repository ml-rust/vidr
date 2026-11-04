//! Output order for frames, ITU-T H.264 C.4.

/// Decoded frames that wait for output, smallest order count first.
#[derive(Debug)]
pub(super) struct OutputQueue<T> {
    waiting: Vec<Waiting<T>>,
    reorder_frames: usize,
    buffer_frames: usize,
}

#[derive(Debug)]
struct Waiting<T> {
    poc: i32,
    id: u64,
    frame: T,
}

impl<T> OutputQueue<T> {
    /// A queue for one sequence.
    pub(super) const fn new(reorder_frames: usize, buffer_frames: usize) -> Self {
        Self {
            waiting: Vec::new(),
            reorder_frames,
            buffer_frames,
        }
    }

    /// Takes the limits of a new sequence.
    pub(super) const fn set_limits(&mut self, reorder_frames: usize, buffer_frames: usize) {
        self.reorder_frames = reorder_frames;
        self.buffer_frames = buffer_frames;
    }

    /// Adds a frame.
    pub(super) fn push(&mut self, id: u64, poc: i32, frame: T) {
        self.waiting.push(Waiting { poc, id, frame });
    }

    /// Frames waiting.
    #[cfg(test)]
    fn len(&self) -> usize {
        self.waiting.len()
    }

    fn pop_smallest(&mut self) -> Option<T> {
        let index = self
            .waiting
            .iter()
            .enumerate()
            .min_by_key(|(_, waiting)| waiting.poc)
            .map(|(index, _)| index)?;
        Some(self.waiting.remove(index).frame)
    }

    /// The next frame that must leave. A frame must leave once more frames wait than
    /// `max_num_reorder_frames`, or once the DPB holds more frames than
    /// `max_dec_frame_buffering`. `is_reference` says whether a frame also holds a slot.
    pub(super) fn pop_ready(
        &mut self,
        references: usize,
        is_reference: impl Fn(u64) -> bool,
    ) -> Option<T> {
        let waiting_references = self
            .waiting
            .iter()
            .filter(|waiting| is_reference(waiting.id))
            .count();
        let held = self.waiting.len() + references.saturating_sub(waiting_references);
        if self.waiting.len() > self.reorder_frames || held > self.buffer_frames {
            self.pop_smallest()
        } else {
            None
        }
    }

    /// Every frame, in order. An IDR picture, operation 5, and the stream's end take this.
    pub(super) fn drain(&mut self) -> Vec<T> {
        let mut frames = Vec::with_capacity(self.waiting.len());
        while let Some(frame) = self.pop_smallest() {
            frames.push(frame);
        }
        frames
    }

    /// Empties the queue without output: `no_output_of_prior_pics_flag` 1 of C.4.4.
    pub(super) fn discard(&mut self) {
        self.waiting.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Decode-order order counts, as frames push and pop. `frame` holds the count.
    fn run(reorder: usize, buffer: usize, pocs: &[i32]) -> Vec<i32> {
        let mut queue = OutputQueue::new(reorder, buffer);
        let mut out = Vec::new();
        for (id, &poc) in pocs.iter().enumerate() {
            queue.push(id as u64, poc, poc);
            while let Some(frame) = queue.pop_ready(0, |_| false) {
                out.push(frame);
            }
        }
        out.extend(queue.drain());
        out
    }

    #[test]
    fn zero_reorder_outputs_in_decode_order() {
        assert_eq!(run(0, 1, &[0, 2, 4, 6]), [0, 2, 4, 6]);
    }

    #[test]
    fn b_frames_come_out_sorted() {
        // I0 P6 B2 B4 P12 B8 B10, with two frames of reorder.
        let order = run(2, 4, &[0, 6, 2, 4, 12, 8, 10]);
        assert_eq!(order, [0, 2, 4, 6, 8, 10, 12]);
    }

    #[test]
    fn a_b_pyramid_comes_out_sorted() {
        // I0 P16 B8 b4 b12, then P32 B24 b20 b28.
        let order = run(3, 4, &[0, 16, 8, 4, 12, 32, 24, 20, 28]);
        assert_eq!(order, [0, 4, 8, 12, 16, 20, 24, 28, 32]);
    }

    #[test]
    fn a_frame_waits_until_the_reorder_depth_passes() {
        let mut queue = OutputQueue::new(1, 4);
        queue.push(0, 0, "first");
        assert_eq!(queue.pop_ready(1, |id| id == 0), None);
        queue.push(1, 2, "second");
        assert_eq!(queue.pop_ready(2, |_| true), Some("first"));
        assert_eq!(queue.len(), 1);
    }

    /// Two references out of the queue and two waiting fill a buffer of four, so one
    /// more frame forces the smallest out. That frame waits below the reorder depth.
    #[test]
    fn a_full_buffer_bumps_below_the_reorder_depth() {
        let mut queue = OutputQueue::new(16, 4);
        queue.push(0, 8, "later");
        queue.push(1, 4, "sooner");
        assert_eq!(queue.pop_ready(2, |_| false), None);
        queue.push(2, 6, "middle");
        assert_eq!(queue.pop_ready(2, |_| false), Some("sooner"));
        assert_eq!(queue.pop_ready(2, |_| false), None);
    }

    #[test]
    fn a_waiting_reference_counts_once() {
        let mut queue = OutputQueue::new(16, 2);
        queue.push(0, 0, "held");
        queue.push(1, 2, "held too");
        // Both waiting frames are the two references, so the buffer holds two frames.
        assert_eq!(queue.pop_ready(2, |_| true), None);
        assert_eq!(queue.drain(), ["held", "held too"]);
    }

    #[test]
    fn a_discard_outputs_nothing_and_the_queue_stays_usable() {
        let mut queue = OutputQueue::new(2, 4);
        queue.push(0, 6, 6);
        queue.push(1, 8, 8);
        queue.discard();
        assert_eq!(queue.len(), 0);
        assert_eq!(queue.pop_ready(0, |_| false), None);
        queue.push(2, 0, 0);
        assert_eq!(queue.drain(), [0]);
    }

    #[test]
    fn new_limits_take_effect_and_drain_sorts() {
        let mut queue = OutputQueue::new(4, 4);
        queue.push(0, 9, 9);
        queue.push(1, 3, 3);
        queue.set_limits(0, 1);
        assert_eq!(queue.pop_ready(0, |_| false), Some(3));
        assert_eq!(queue.drain(), [9]);
    }
}
