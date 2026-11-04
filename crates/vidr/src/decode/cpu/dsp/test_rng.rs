//! A small seeded generator for the kernel comparison tests.

/// Xorshift64.
pub(crate) struct Rng(u64);

impl Rng {
    /// A generator from a seed that is not zero.
    pub(crate) const fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// The next 64 random bits.
    pub(crate) fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// A value from `0` to `limit - 1`.
    pub(crate) fn below(&mut self, limit: u64) -> u64 {
        self.next() % limit
    }

    /// A sample at a bit depth: flat areas, small steps, and the full range.
    pub(crate) fn sample(&mut self, around: i32, bit_depth: u32) -> i32 {
        let max = (1i32 << bit_depth) - 1;
        match self.below(4) {
            0 => around.clamp(0, max),
            1 => (around + self.below(17) as i32 - 8).clamp(0, max),
            2 => self.below(max as u64 + 1) as i32,
            _ => {
                if self.below(2) == 0 {
                    0
                } else {
                    max
                }
            }
        }
    }
}
