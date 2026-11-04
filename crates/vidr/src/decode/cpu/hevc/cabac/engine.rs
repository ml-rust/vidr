//! The arithmetic decoding engine (9.3.4.3) with its own fast reader.
//!
//! `offset` holds `ivlOffset` above `available` bits read ahead of the
//! specification's bitstream pointer. A comparison scales `ivlCurrRange` up by
//! `available`, so a renormalization only lowers `available`. Bits past the end
//! of the data read as zero. A caller finds that through [`Cabac::exhausted`],
//! so the hot paths stay free of error returns.

use super::context::ContextTable;
use super::tables::{NEXT_STATE_LPS, NEXT_STATE_MPS, RANGE_TAB_LPS};
use crate::decode::cpu::hevc::error::HevcError;

/// A refill runs below this many bits. A decision takes at most 7 bits and a
/// bypass run at most [`BYPASS_RUN`].
const MIN_AVAILABLE: u32 = 24;
/// Bypass bins decoded between refills.
const BYPASS_RUN: u32 = 16;
/// The bits of `ivlOffset` (9.3.2.6).
const OFFSET_BITS: u32 = 9;

/// The engine registers and its read position.
pub(crate) struct CabacEngine<'a> {
    data: &'a [u8],
    /// The next byte to load. It passes the end of `data` as zero bytes load.
    next: usize,
    offset: u64,
    available: u32,
    range: u32,
}

impl<'a> CabacEngine<'a> {
    /// Starts decoding at byte `start` of `data` (9.3.2.6).
    ///
    /// # Errors
    ///
    /// [`HevcError::InvalidSyntax`] for an `ivlOffset` of 510 or 511.
    pub(crate) fn new(data: &'a [u8], start: usize) -> Result<Self, HevcError> {
        let mut engine = Self {
            data,
            next: start,
            offset: 0,
            available: 0,
            range: 510,
        };
        engine.restart_at(start)?;
        Ok(engine)
    }

    /// Runs the initialization of 9.3.2.6 at byte `start`, after PCM samples or
    /// at an entry point.
    ///
    /// # Errors
    ///
    /// [`HevcError::InvalidSyntax`] for an `ivlOffset` of 510 or 511.
    pub(crate) fn restart_at(&mut self, start: usize) -> Result<(), HevcError> {
        self.next = start;
        self.range = 510;
        self.offset = u64::from(self.next_word());
        self.available = u32::BITS - OFFSET_BITS;
        let offset = self.offset >> self.available;
        if offset >= 510 {
            return Err(HevcError::InvalidSyntax {
                element: "ivlOffset",
                value: offset as i64,
                allowed: "0 to 509",
            });
        }
        Ok(())
    }

    /// The next four bytes, with zeros past the end of the data.
    fn next_word(&mut self) -> u32 {
        let rest = self.data.get(self.next..).unwrap_or_default();
        let word = match rest.first_chunk::<4>() {
            Some(bytes) => u32::from_be_bytes(*bytes),
            None => {
                let mut bytes = [0u8; 4];
                for (target, &source) in bytes.iter_mut().zip(rest) {
                    *target = source;
                }
                u32::from_be_bytes(bytes)
            }
        };
        self.next = self.next.saturating_add(4);
        word
    }

    #[inline]
    fn refill(&mut self) {
        if self.available < MIN_AVAILABLE {
            let word = self.next_word();
            self.offset = (self.offset << u32::BITS) | u64::from(word);
            self.available += u32::BITS;
        }
    }

    /// DecodeDecision (9.3.4.3.2) on the packed context `state`.
    #[inline]
    pub(crate) fn decode_decision(&mut self, state: &mut u8) -> bool {
        self.refill();
        let packed = usize::from(*state & 127);
        let q_range_idx = ((self.range >> 6) & 3) as usize;
        let lps_range = u32::from(RANGE_TAB_LPS[packed >> 1][q_range_idx]);
        let mps_range = self.range - lps_range;
        let scaled = u64::from(mps_range) << self.available;
        let val_mps = packed & 1 == 1;
        if self.offset < scaled {
            *state = NEXT_STATE_MPS[packed];
            // rangeTabLps leaves at least 128 here, so one doubling is enough.
            if mps_range < 256 {
                self.range = mps_range << 1;
                self.available -= 1;
            } else {
                self.range = mps_range;
            }
            val_mps
        } else {
            self.offset -= scaled;
            *state = NEXT_STATE_LPS[packed];
            let shift = lps_range.leading_zeros() - (u32::BITS - OFFSET_BITS);
            self.range = lps_range << shift;
            self.available -= shift;
            !val_mps
        }
    }

    /// DecodeBypass (9.3.4.3.4).
    #[inline]
    pub(crate) fn decode_bypass(&mut self) -> bool {
        self.refill();
        self.bypass_bin()
    }

    /// One bypass bin, after a refill that covers it.
    #[inline]
    fn bypass_bin(&mut self) -> bool {
        self.available -= 1;
        let scaled = u64::from(self.range) << self.available;
        if self.offset >= scaled {
            self.offset -= scaled;
            true
        } else {
            false
        }
    }

    /// `count` bypass bins, first bin in the highest bit (9.3.3.5). A count above
    /// 32 keeps the last 32 bins.
    pub(crate) fn decode_bypass_bits(&mut self, count: u32) -> u32 {
        let mut value = 0u32;
        let mut left = count;
        while left > 0 {
            let run = left.min(BYPASS_RUN);
            self.refill();
            for _ in 0..run {
                value = (value << 1) | u32::from(self.bypass_bin());
            }
            left -= run;
        }
        value
    }

    /// DecodeTerminate (9.3.4.3.5).
    ///
    /// After a true result, the read position follows the last bit in
    /// `ivlOffset`. The next call on this engine is [`Self::restart_at`].
    pub(crate) fn decode_terminate(&mut self) -> bool {
        self.refill();
        self.range -= 2;
        let scaled = u64::from(self.range) << self.available;
        if self.offset >= scaled {
            return true;
        }
        if self.range < 256 {
            self.range <<= 1;
            self.available -= 1;
        }
        false
    }

    /// Bits the specification's bitstream pointer has passed.
    pub(crate) fn bit_position(&self) -> u64 {
        (self.next as u64 * 8).saturating_sub(u64::from(self.available))
    }

    /// True when the bitstream pointer has passed the end of the data.
    pub(crate) fn exhausted(&self) -> bool {
        self.bit_position() > self.data.len() as u64 * 8
    }

    /// The data this engine reads.
    pub(crate) const fn data(&self) -> &'a [u8] {
        self.data
    }

    /// The error for reading past the end of the data.
    pub(crate) fn truncated(&self) -> HevcError {
        HevcError::SliceDataTruncated {
            read_bits: self.bit_position(),
            size: self.data.len(),
        }
    }

    /// Skips `pcm_alignment_zero_bit` (7.3.8.5) after a `pcm_flag` of 1, and
    /// returns the byte that follows.
    ///
    /// # Errors
    ///
    /// - [`HevcError::SliceDataTruncated`] past the end of the data.
    /// - [`HevcError::InvalidSyntax`] for an alignment bit of 1.
    pub(crate) fn aligned_byte(&self) -> Result<usize, HevcError> {
        self.aligned_byte_after("pcm_alignment_zero_bit")
    }

    /// Skips zero bits up to a byte boundary after a terminating bin, and returns
    /// the byte that follows. `element` names the bits in the error.
    ///
    /// # Errors
    ///
    /// - [`HevcError::SliceDataTruncated`] past the end of the data.
    /// - [`HevcError::InvalidSyntax`] for a bit of 1.
    pub(crate) fn aligned_byte_after(&self, element: &'static str) -> Result<usize, HevcError> {
        if self.exhausted() {
            return Err(self.truncated());
        }
        let position = self.bit_position();
        let byte = (position / 8) as usize;
        let used = (position % 8) as u32;
        if used == 0 {
            return Ok(byte);
        }
        let partial = self.data.get(byte).copied().unwrap_or(0);
        let padding = partial & (0xff >> used);
        if padding != 0 {
            return Err(HevcError::InvalidSyntax {
                element,
                value: i64::from(padding),
                allowed: "0",
            });
        }
        Ok(byte + 1)
    }
}

/// The engine with the context variables of one slice segment.
pub(crate) struct Cabac<'a> {
    pub(crate) engine: CabacEngine<'a>,
    pub(crate) contexts: ContextTable,
}

impl<'a> Cabac<'a> {
    /// Starts at byte `start` of `data` with the context table `contexts`.
    ///
    /// # Errors
    ///
    /// [`HevcError::InvalidSyntax`] for an `ivlOffset` of 510 or 511.
    pub(crate) fn new(
        data: &'a [u8],
        start: usize,
        contexts: ContextTable,
    ) -> Result<Self, HevcError> {
        Ok(Self {
            engine: CabacEngine::new(data, start)?,
            contexts,
        })
    }

    /// One bin decoded with context `index`.
    #[inline]
    pub(crate) fn decision(&mut self, index: usize) -> bool {
        self.engine.decode_decision(self.contexts.state_mut(index))
    }

    /// One bypass bin.
    #[inline]
    pub(crate) fn bypass(&mut self) -> bool {
        self.engine.decode_bypass()
    }

    /// `count` bypass bins as a fixed-length value.
    #[inline]
    pub(crate) fn bypass_bits(&mut self, count: u32) -> u32 {
        self.engine.decode_bypass_bits(count)
    }

    /// One bin before termination.
    #[inline]
    pub(crate) fn terminate(&mut self) -> bool {
        self.engine.decode_terminate()
    }

    /// True when the parser has read past the end of the slice data.
    #[inline]
    pub(crate) fn exhausted(&self) -> bool {
        self.engine.exhausted()
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_encoder::CabacEncoder;
    use super::*;

    /// Reproducible values for the round trips.
    struct Xorshift(u64);

    impl Xorshift {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    #[derive(Clone, Copy, Debug)]
    enum Bin {
        Context(usize, bool),
        Bypass(bool),
        Run(u32, u32),
        Terminate,
    }

    /// Bins with a mix of skewed and even contexts, in every decoding mode.
    fn random_bins(rng: &mut Xorshift, count: usize) -> Vec<Bin> {
        (0..count)
            .map(|_| {
                let value = rng.next();
                match value % 8 {
                    0..=3 => {
                        let context = ((value >> 8) % 40) as usize;
                        let one_in = 1 + context as u64 % 7;
                        Bin::Context(context, (value >> 32).is_multiple_of(one_in))
                    }
                    4 | 5 => Bin::Bypass((value >> 40) & 1 == 1),
                    6 => {
                        let count = 1 + ((value >> 8) % 32) as u32;
                        let bits = (value >> 16) as u32;
                        let mask = if count == 32 {
                            u32::MAX
                        } else {
                            (1 << count) - 1
                        };
                        Bin::Run(count, bits & mask)
                    }
                    _ => Bin::Terminate,
                }
            })
            .collect()
    }

    fn encode(bins: &[Bin], contexts: &ContextTable) -> Vec<u8> {
        let mut encoder = CabacEncoder::new();
        let mut contexts = contexts.clone();
        for bin in bins {
            match *bin {
                Bin::Context(index, value) => {
                    encoder.encode_decision(contexts.state_mut(index), value);
                }
                Bin::Bypass(value) => encoder.encode_bypass(value),
                Bin::Run(count, value) => encoder.encode_bypass_bits(value, count),
                Bin::Terminate => encoder.encode_terminate(false),
            }
        }
        encoder.encode_terminate(true);
        encoder.finish()
    }

    fn decode_and_check(bins: &[Bin], bytes: &[u8], contexts: &ContextTable) {
        let mut cabac = Cabac::new(bytes, 0, contexts.clone()).unwrap();
        for (index, bin) in bins.iter().enumerate() {
            match *bin {
                Bin::Context(context, value) => {
                    assert_eq!(cabac.decision(context), value, "bin {index}");
                }
                Bin::Bypass(value) => assert_eq!(cabac.bypass(), value, "bin {index}"),
                Bin::Run(count, value) => {
                    assert_eq!(cabac.bypass_bits(count), value, "bin {index}");
                }
                Bin::Terminate => assert!(!cabac.terminate(), "bin {index}"),
            }
        }
        assert!(cabac.terminate());
        assert!(!cabac.exhausted());
    }

    #[test]
    fn random_bins_round_trip_through_the_encoder() {
        let mut rng = Xorshift(0x9e37_79b9_7f4a_7c15);
        for (trial, qp) in (0..200).zip((0..52).cycle()) {
            let bins = random_bins(&mut rng, 1 + trial * 13);
            let contexts = ContextTable::new(trial % 3, qp);
            let bytes = encode(&bins, &contexts);
            decode_and_check(&bins, &bytes, &contexts);
        }
    }

    /// Long runs on one context drive `pStateIdx` to both ends of its range.
    #[test]
    fn skewed_runs_round_trip() {
        let mut bins = vec![Bin::Context(0, true); 500];
        bins.extend(std::iter::repeat_n(Bin::Context(0, false), 500));
        bins.extend(std::iter::repeat_n(Bin::Context(1, false), 300));
        bins.push(Bin::Context(1, true));
        let contexts = ContextTable::new(0, 26);
        let bytes = encode(&bins, &contexts);
        decode_and_check(&bins, &bytes, &contexts);
    }

    /// Bytes and bins from a separate model of 9.3.4.3 and 9.3.5. The fixed
    /// bytes hold the engine to the specification with no help from the encoder
    /// in this crate.
    #[test]
    fn a_fixed_vector_decodes() {
        const BYTES: [u8; 8] = [111, 186, 123, 71, 180, 123, 200, 128];
        const BINS: [u8; 48] = [
            0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0,
            0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1,
        ];
        let mut engine = CabacEngine::new(&BYTES, 0).unwrap();
        // pStateIdx 3 with valMps 0.
        let mut state = 3u8 << 1;
        let mut bins = BINS.iter();
        for step in 0..64 {
            if step % 4 == 3 {
                assert!(!engine.decode_terminate(), "step {step}");
                continue;
            }
            let decoded = if step % 4 == 2 {
                engine.decode_bypass()
            } else {
                engine.decode_decision(&mut state)
            };
            assert_eq!(Some(u8::from(decoded)), bins.next().copied(), "step {step}");
        }
        assert!(engine.decode_terminate());
        // The model's flush ends after 57 bits.
        assert_eq!(engine.bit_position(), 57);
    }

    #[test]
    fn terminate_leaves_the_pointer_after_the_last_flushed_bit() {
        let mut rng = Xorshift(0x1234_5678_9abc_def1);
        for trial in 0..100 {
            let bins = random_bins(&mut rng, trial);
            let contexts = ContextTable::new(1, 30);
            let mut encoder = CabacEncoder::new();
            let mut encoder_contexts = contexts.clone();
            for bin in &bins {
                match *bin {
                    Bin::Context(index, value) => {
                        encoder.encode_decision(encoder_contexts.state_mut(index), value);
                    }
                    Bin::Bypass(value) => encoder.encode_bypass(value),
                    Bin::Run(count, value) => encoder.encode_bypass_bits(value, count),
                    Bin::Terminate => encoder.encode_terminate(false),
                }
            }
            encoder.encode_terminate(true);
            let flushed_bits = encoder.bit_len() as u64;
            let bytes = encoder.finish();
            let mut cabac = Cabac::new(&bytes, 0, contexts).unwrap();
            for bin in &bins {
                match *bin {
                    Bin::Context(index, _) => {
                        cabac.decision(index);
                    }
                    Bin::Bypass(_) => {
                        cabac.bypass();
                    }
                    Bin::Run(count, _) => {
                        cabac.bypass_bits(count);
                    }
                    Bin::Terminate => {
                        cabac.terminate();
                    }
                }
            }
            assert!(cabac.terminate());
            assert_eq!(cabac.engine.bit_position(), flushed_bits);
        }
    }

    #[test]
    fn restart_reads_from_the_given_byte() {
        let mut encoder = CabacEncoder::new();
        encoder.encode_bypass(true);
        encoder.encode_terminate(true);
        let first_len = encoder.bit_len().div_ceil(8);
        encoder.align_zero();
        encoder.restart();
        encoder.encode_bypass(false);
        encoder.encode_bypass(true);
        encoder.encode_terminate(true);
        let bytes = encoder.finish();
        let mut engine = CabacEngine::new(&bytes, 0).unwrap();
        assert!(engine.decode_bypass());
        assert!(engine.decode_terminate());
        let next = engine.aligned_byte().unwrap();
        assert_eq!(next, first_len);
        engine.restart_at(next).unwrap();
        assert!(!engine.decode_bypass());
        assert!(engine.decode_bypass());
        assert!(engine.decode_terminate());
    }

    #[test]
    fn a_set_alignment_bit_is_refused() {
        let mut encoder = CabacEncoder::new();
        encoder.encode_terminate(true);
        let mut bytes = encoder.finish();
        // The flush ends mid-byte, so the padding holds zeros. Set one of them.
        let last = bytes.len() - 1;
        bytes[last] |= 1;
        let mut engine = CabacEngine::new(&bytes, 0).unwrap();
        assert!(engine.decode_terminate());
        assert!(matches!(
            engine.aligned_byte(),
            Err(HevcError::InvalidSyntax {
                element: "pcm_alignment_zero_bit",
                ..
            })
        ));
    }

    #[test]
    fn an_offset_of_510_is_refused() {
        // 510 in 9 bits is 1 1111 1110.
        let bytes = [0xff, 0x00, 0, 0];
        assert!(matches!(
            CabacEngine::new(&bytes, 0),
            Err(HevcError::InvalidSyntax {
                element: "ivlOffset",
                value: 510,
                ..
            })
        ));
    }

    #[test]
    fn reading_past_the_end_reports_exhausted() {
        let bytes = [0u8; 2];
        let mut cabac = Cabac::new(&bytes, 0, ContextTable::new(0, 26)).unwrap();
        assert!(!cabac.exhausted());
        let _ = cabac.bypass_bits(7);
        assert!(!cabac.exhausted());
        let _ = cabac.bypass();
        assert!(cabac.exhausted());
        assert_eq!(
            cabac.engine.truncated(),
            HevcError::SliceDataTruncated {
                read_bits: 17,
                size: 2
            }
        );
    }

    #[test]
    fn an_empty_buffer_decodes_zeros_and_reports_exhausted() {
        let mut cabac = Cabac::new(&[], 0, ContextTable::new(0, 26)).unwrap();
        assert!(cabac.exhausted());
        let _ = cabac.decision(0);
        let _ = cabac.bypass_bits(32);
        assert!(!cabac.terminate());
    }
}
