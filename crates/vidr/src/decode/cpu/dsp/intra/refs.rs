//! Reference samples of one transform block (8.4.4.2.2, 8.4.4.2.3), and the
//! prediction output array.

/// The largest `nTbS`.
const MAX_TB_SIZE: usize = 32;
/// `nTbS * 4 + 1` at the largest `nTbS`.
const MAX_REFS: usize = 4 * MAX_TB_SIZE + 1;
/// `INTRA_DC` (Table 8-1).
const INTRA_DC: u8 = 1;

/// The samples `p[ x ][ y ]` in one line: from `p[ -1 ][ 2 * nTbS - 1 ]` up to the
/// corner `p[ -1 ][ -1 ]`, then right to `p[ 2 * nTbS - 1 ][ -1 ]`. Substitution
/// and smoothing both walk this order.
#[derive(Clone, Debug)]
pub(crate) struct RefSamples {
    samples: [u16; MAX_REFS],
    available: [bool; MAX_REFS],
    size: usize,
}

impl RefSamples {
    /// An empty set for the largest block.
    pub(crate) const fn new() -> Self {
        Self {
            samples: [0; MAX_REFS],
            available: [false; MAX_REFS],
            size: MAX_TB_SIZE,
        }
    }

    /// Starts a block of `Log2( nTbS )` `log2_size`, with every sample marked not
    /// available. A size above 32 acts as 32.
    pub(crate) fn begin(&mut self, log2_size: u32) {
        self.size = (1usize << log2_size.min(5)).min(MAX_TB_SIZE);
        self.available.fill(false);
    }

    /// `nTbS`.
    pub(crate) const fn size(&self) -> usize {
        self.size
    }

    /// The line position of `p[ -1 ][ -1 ]`.
    pub(crate) const fn corner_index(&self) -> usize {
        2 * self.size
    }

    /// Samples in the line.
    const fn count(&self) -> usize {
        4 * self.size + 1
    }

    fn set(&mut self, index: usize, value: u16) {
        if let (Some(sample), Some(flag)) =
            (self.samples.get_mut(index), self.available.get_mut(index))
        {
            *sample = value;
            *flag = true;
        }
    }

    /// Stores `p[ -1 ][ y ]`, and marks it available.
    pub(crate) fn set_left(&mut self, y: usize, value: u16) {
        let corner = self.corner_index();
        self.set(corner.wrapping_sub(1 + y), value);
    }

    /// Stores `p[ -1 ][ -1 ]`, and marks it available.
    pub(crate) fn set_corner(&mut self, value: u16) {
        self.set(self.corner_index(), value);
    }

    /// Stores `p[ x ][ -1 ]`, and marks it available.
    pub(crate) fn set_top(&mut self, x: usize, value: u16) {
        self.set(self.corner_index() + 1 + x, value);
    }

    /// The sample at line position `index`.
    #[inline]
    pub(crate) fn at(&self, index: usize) -> u16 {
        self.samples.get(index).copied().unwrap_or(0)
    }

    /// `p[ -1 ][ y ]`.
    #[inline]
    pub(crate) fn left(&self, y: usize) -> u16 {
        self.at(self.corner_index().wrapping_sub(1 + y))
    }

    /// `p[ -1 ][ -1 ]`.
    #[inline]
    pub(crate) fn corner(&self) -> u16 {
        self.at(self.corner_index())
    }

    /// `p[ x ][ -1 ]`.
    #[inline]
    pub(crate) fn top(&self, x: usize) -> u16 {
        self.at(self.corner_index() + 1 + x)
    }

    /// The reference sample substitution process (8.4.4.2.2).
    pub(crate) fn substitute(&mut self, bit_depth: u32) {
        let count = self.count();
        let samples = &mut self.samples[..count];
        let available = &self.available[..count];
        let Some(first) = available.iter().position(|&flag| flag) else {
            samples.fill(1 << (bit_depth - 1));
            return;
        };
        if first > 0 {
            samples[0] = samples[first];
        }
        for index in 1..count {
            if !available[index] {
                samples[index] = samples[index - 1];
            }
        }
    }

    /// The filtering process of neighbouring samples (8.4.4.2.3) for a luma block.
    pub(crate) fn filter_for_mode(&mut self, mode: u8, strong_enabled: bool, bit_depth: u32) {
        let threshold = match self.size {
            8 => 7,
            16 => 1,
            32 => 0,
            _ => return,
        };
        if mode == INTRA_DC {
            return;
        }
        let mode = i32::from(mode);
        if (mode - 26).abs().min((mode - 10).abs()) <= threshold {
            return;
        }
        if strong_enabled && self.size == 32 && self.is_flat(bit_depth) {
            self.smooth_strong();
        } else {
            self.smooth();
        }
    }

    /// The last two conditions on `biIntFlag` (8.4.4.2.3).
    fn is_flat(&self, bit_depth: u32) -> bool {
        let n = self.size;
        let corner = i32::from(self.corner());
        let limit = 1 << (bit_depth - 5);
        let far_top = i32::from(self.top(2 * n - 1));
        let far_left = i32::from(self.left(2 * n - 1));
        (corner + far_top - 2 * i32::from(self.top(n - 1))).abs() < limit
            && (corner + far_left - 2 * i32::from(self.left(n - 1))).abs() < limit
    }

    /// (8-41) to (8-45). The line order makes each sample's neighbours adjacent,
    /// the corner included.
    fn smooth(&mut self) {
        let count = self.count();
        let samples = &mut self.samples[..count];
        let mut previous = u32::from(samples[0]);
        for index in 1..count - 1 {
            let current = u32::from(samples[index]);
            let next = u32::from(samples[index + 1]);
            samples[index] = ((previous + 2 * current + next + 2) >> 2) as u16;
            previous = current;
        }
    }

    /// (8-36) to (8-40), for an `nTbS` of 32.
    fn smooth_strong(&mut self) {
        let corner = u32::from(self.corner());
        let far_left = u32::from(self.left(63));
        let far_top = u32::from(self.top(63));
        for position in 0..63u32 {
            let left = ((63 - position) * corner + (position + 1) * far_left + 32) >> 6;
            let top = ((63 - position) * corner + (position + 1) * far_top + 32) >> 6;
            self.samples[63 - position as usize] = left as u16;
            self.samples[65 + position as usize] = top as u16;
        }
    }
}

/// `predSamples` of one block, row by row with no gaps.
#[derive(Clone, Debug)]
pub(crate) struct PredBlock {
    samples: [u16; MAX_TB_SIZE * MAX_TB_SIZE],
}

impl PredBlock {
    /// A block of zeros.
    pub(crate) const fn new() -> Self {
        Self {
            samples: [0; MAX_TB_SIZE * MAX_TB_SIZE],
        }
    }

    /// The rows of a block of `size` samples.
    pub(crate) fn rows(&self, size: usize) -> impl Iterator<Item = &[u16]> {
        self.samples
            .chunks_exact(size.clamp(1, MAX_TB_SIZE))
            .take(size)
    }

    /// The rows of a block of `size` samples, for writing.
    pub(crate) fn rows_mut(&mut self, size: usize) -> impl Iterator<Item = &mut [u16]> {
        self.samples
            .chunks_exact_mut(size.clamp(1, MAX_TB_SIZE))
            .take(size)
    }

    /// Swaps rows and columns of a block of `size` samples.
    pub(crate) fn transpose(&mut self, size: usize) {
        let size = size.min(MAX_TB_SIZE);
        for y in 0..size {
            for x in 0..y {
                self.samples.swap(y * size + x, x * size + y);
            }
        }
    }

    /// The sample at row `y`, column `x`, of a block of `size` samples.
    pub(crate) fn get(&self, size: usize, x: usize, y: usize) -> u16 {
        self.samples.get(y * size + x).copied().unwrap_or(0)
    }

    /// Writes one sample of a block of `size` samples.
    pub(crate) fn set(&mut self, size: usize, x: usize, y: usize, value: u16) {
        if let Some(sample) = self.samples.get_mut(y * size + x) {
            *sample = value;
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A set for `log2_size` with `left`, `corner`, and `top` values for every
    /// sample.
    pub(crate) fn filled(log2_size: u32, left: u16, corner: u16, top: u16) -> RefSamples {
        let mut refs = RefSamples::new();
        refs.begin(log2_size);
        for index in 0..2 * refs.size() {
            refs.set_left(index, left);
            refs.set_top(index, top);
        }
        refs.set_corner(corner);
        refs
    }

    #[test]
    fn the_line_runs_from_the_far_left_to_the_far_top() {
        let mut refs = RefSamples::new();
        refs.begin(2);
        for index in 0..8 {
            refs.set_left(index, 100 + index as u16);
            refs.set_top(index, 200 + index as u16);
        }
        refs.set_corner(50);
        assert_eq!(refs.at(0), 107);
        assert_eq!(refs.at(7), 100);
        assert_eq!(refs.at(8), 50);
        assert_eq!(refs.at(9), 200);
        assert_eq!(refs.at(16), 207);
        assert_eq!(refs.left(3), 103);
        assert_eq!(refs.top(5), 205);
    }

    #[test]
    fn with_nothing_available_every_sample_takes_mid_grey() {
        let mut refs = RefSamples::new();
        refs.begin(2);
        refs.substitute(8);
        assert!((0..17).all(|index| refs.at(index) == 128));
        refs.begin(3);
        refs.substitute(10);
        assert!((0..33).all(|index| refs.at(index) == 512));
    }

    /// (8.4.4.2.2): the first sample found fills the start, and each gap copies
    /// the sample before it in the line.
    #[test]
    fn gaps_copy_the_sample_before_them() {
        let mut refs = RefSamples::new();
        refs.begin(2);
        // Available: left rows 0 and 1, and top columns 2 and 3.
        refs.set_left(1, 11);
        refs.set_left(0, 10);
        refs.set_top(2, 22);
        refs.set_top(3, 23);
        refs.substitute(8);
        let line: Vec<u16> = (0..17).map(|index| refs.at(index)).collect();
        assert_eq!(
            line,
            vec![
                11, 11, 11, 11, 11, 11, 11, 10, 10, 10, 10, 22, 23, 23, 23, 23, 23
            ]
        );
    }

    /// Worked from (8-41) to (8-45) on a line of squares.
    #[test]
    fn smoothing_keeps_the_ends_and_filters_the_rest() {
        let mut refs = RefSamples::new();
        refs.begin(3);
        for index in 0..8 {
            refs.set_left(index, 100);
            refs.set_top(index, 100);
        }
        for index in 8..16 {
            refs.set_left(index, 50);
            refs.set_top(index, 150);
        }
        refs.set_corner(100);
        // Planar: minDistVerHor 10 is above threshold 7.
        refs.filter_for_mode(0, false, 8);
        // The ends stay.
        assert_eq!(refs.left(15), 50);
        assert_eq!(refs.top(15), 150);
        // ( 50 + 2 * 50 + 100 + 2 ) >> 2 = 63 where the steps meet.
        assert_eq!(refs.left(8), 63);
        // ( 50 + 2 * 100 + 100 + 2 ) >> 2 = 88.
        assert_eq!(refs.left(7), 88);
        // ( 100 + 2 * 100 + 150 + 2 ) >> 2 = 113.
        assert_eq!(refs.top(7), 113);
        // ( 100 + 2 * 150 + 150 + 2 ) >> 2 = 138.
        assert_eq!(refs.top(8), 138);
        assert_eq!(refs.corner(), 100);
    }

    #[test]
    fn no_filter_runs_below_the_threshold() {
        let cases = [
            (2, 0),  // nTbS 4 never filters.
            (3, 1),  // DC never filters.
            (3, 26), // Distance 0 at nTbS 8.
            (3, 3),  // Distance 7 is not above 7.
            (4, 9),  // Distance 1 is not above 1.
            (5, 10), // Distance 0 is not above 0.
        ];
        for (log2_size, mode) in cases {
            let mut refs = filled(log2_size, 10, 90, 200);
            let before: Vec<u16> = (0..=4usize << log2_size)
                .map(|index| refs.at(index))
                .collect();
            refs.filter_for_mode(mode, false, 8);
            let after: Vec<u16> = (0..=4usize << log2_size)
                .map(|index| refs.at(index))
                .collect();
            assert_eq!(before, after, "size {log2_size}, mode {mode}");
        }
    }

    #[test]
    fn the_filter_runs_above_the_threshold() {
        for (log2_size, mode) in [(3, 2), (3, 34), (3, 18), (4, 8), (5, 11)] {
            let mut refs = filled(log2_size, 10, 90, 200);
            let corner = refs.corner();
            refs.filter_for_mode(mode, false, 8);
            assert_ne!(refs.corner(), corner, "size {log2_size}, mode {mode}");
        }
    }

    /// (8-36) to (8-40) on a corner of 100 with far samples of 104 and 96.
    #[test]
    fn strong_smoothing_interpolates_between_the_corners() {
        let mut refs = RefSamples::new();
        refs.begin(5);
        for index in 0..64 {
            refs.set_left(index, if index == 63 { 96 } else { 98 });
            refs.set_top(index, if index == 63 { 104 } else { 102 });
        }
        refs.set_corner(100);
        refs.filter_for_mode(0, true, 8);
        assert_eq!(refs.corner(), 100);
        assert_eq!(refs.left(63), 96);
        assert_eq!(refs.top(63), 104);
        // y = 0: ( 63 * 100 + 1 * 96 + 32 ) >> 6 = 100.
        assert_eq!(refs.left(0), 100);
        // y = 31: ( 32 * 100 + 32 * 96 + 32 ) >> 6 = 98.
        assert_eq!(refs.left(31), 98);
        // x = 31: ( 32 * 100 + 32 * 104 + 32 ) >> 6 = 102.
        assert_eq!(refs.top(31), 102);
        // x = 62: ( 1 * 100 + 63 * 104 + 32 ) >> 6 = 104.
        assert_eq!(refs.top(62), 104);
    }

    /// A step above `1 << ( BitDepthY - 5 )` selects the three-tap filter.
    #[test]
    fn a_step_turns_strong_smoothing_off() {
        let mut refs = RefSamples::new();
        refs.begin(5);
        for index in 0..64 {
            refs.set_left(index, 100);
            refs.set_top(index, if index < 31 { 100 } else { 108 });
        }
        refs.set_corner(100);
        refs.filter_for_mode(0, true, 8);
        // Three taps: ( 100 + 2 * 100 + 108 + 2 ) >> 2 = 102 at x = 30.
        assert_eq!(refs.top(30), 102);
    }

    #[test]
    fn transposing_swaps_rows_and_columns() {
        let mut block = PredBlock::new();
        for y in 0..4 {
            for x in 0..4 {
                block.set(4, x, y, (10 * y + x) as u16);
            }
        }
        block.transpose(4);
        assert_eq!(block.get(4, 3, 1), 31);
        assert_eq!(block.get(4, 1, 3), 13);
        assert_eq!(block.get(4, 2, 2), 22);
    }
}
