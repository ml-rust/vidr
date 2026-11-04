//! A CABAC encoder for tests, written from 9.3.5 and Figures 9-10 to 9-15.

use super::tables::{NEXT_STATE_LPS, NEXT_STATE_MPS, RANGE_TAB_LPS};

/// The encoder registers (9.3.5.2) and the bits written so far.
pub(crate) struct CabacEncoder {
    low: u32,
    range: u32,
    first_bit: bool,
    bits_outstanding: u32,
    bits: Vec<bool>,
}

impl CabacEncoder {
    pub(crate) const fn new() -> Self {
        Self {
            low: 0,
            range: 510,
            first_bit: true,
            bits_outstanding: 0,
            bits: Vec::new(),
        }
    }

    /// Runs InitEncoder again, as after PCM samples (9.3.5.2).
    pub(crate) fn restart(&mut self) {
        self.low = 0;
        self.range = 510;
        self.first_bit = true;
        self.bits_outstanding = 0;
    }

    /// PutBit (Figure 9-12).
    fn put_bit(&mut self, bit: bool) {
        if self.first_bit {
            self.first_bit = false;
        } else {
            self.bits.push(bit);
        }
        while self.bits_outstanding > 0 {
            self.bits.push(!bit);
            self.bits_outstanding -= 1;
        }
    }

    /// RenormE (Figure 9-11).
    fn renorm(&mut self) {
        while self.range < 256 {
            if self.low < 256 {
                self.put_bit(false);
            } else if self.low >= 512 {
                self.low -= 512;
                self.put_bit(true);
            } else {
                self.low -= 256;
                self.bits_outstanding += 1;
            }
            self.range <<= 1;
            self.low <<= 1;
        }
    }

    /// EncodeDecision (9.3.5.3) on a packed context state.
    pub(crate) fn encode_decision(&mut self, state: &mut u8, bin: bool) {
        let packed = usize::from(*state & 127);
        let q_range_idx = ((self.range >> 6) & 3) as usize;
        let lps_range = u32::from(RANGE_TAB_LPS[packed >> 1][q_range_idx]);
        self.range -= lps_range;
        let val_mps = packed & 1 == 1;
        if bin == val_mps {
            *state = NEXT_STATE_MPS[packed];
        } else {
            self.low += self.range;
            self.range = lps_range;
            *state = NEXT_STATE_LPS[packed];
        }
        self.renorm();
    }

    /// EncodeBypass (Figure 9-13).
    pub(crate) fn encode_bypass(&mut self, bin: bool) {
        self.low <<= 1;
        if bin {
            self.low += self.range;
        }
        if self.low >= 1024 {
            self.put_bit(true);
            self.low -= 1024;
        } else if self.low < 512 {
            self.put_bit(false);
        } else {
            self.low -= 512;
            self.bits_outstanding += 1;
        }
    }

    /// `count` bypass bins, highest bit first.
    pub(crate) fn encode_bypass_bits(&mut self, value: u32, count: u32) {
        for shift in (0..count).rev() {
            self.encode_bypass((value >> shift) & 1 == 1);
        }
    }

    /// EncodeTerminate (Figure 9-14), with EncodeFlush (Figure 9-15) for a bin
    /// of 1.
    pub(crate) fn encode_terminate(&mut self, bin: bool) {
        self.range -= 2;
        if bin {
            self.low += self.range;
            self.flush();
        } else {
            self.renorm();
        }
    }

    /// EncodeFlush (Figure 9-15).
    fn flush(&mut self) {
        self.range = 2;
        self.renorm();
        self.put_bit((self.low >> 9) & 1 == 1);
        // WriteBits( ( ( ivlLow >> 7 ) & 3 ) | 1, 2 ).
        self.bits.push((self.low >> 8) & 1 == 1);
        self.bits.push(true);
    }

    /// Writes raw bits, such as PCM samples, highest bit first.
    pub(crate) fn write_raw_bits(&mut self, value: u32, count: u32) {
        for shift in (0..count).rev() {
            self.bits.push((value >> shift) & 1 == 1);
        }
    }

    /// Writes zero bits up to a byte boundary.
    pub(crate) fn align_zero(&mut self) {
        while !self.bits.len().is_multiple_of(8) {
            self.bits.push(false);
        }
    }

    /// Bits written so far.
    pub(crate) fn bit_len(&self) -> usize {
        self.bits.len()
    }

    /// The bits as bytes, with zeros to the last byte boundary.
    pub(crate) fn finish(mut self) -> Vec<u8> {
        self.align_zero();
        self.bits
            .chunks(8)
            .map(|byte| {
                byte.iter()
                    .fold(0u8, |acc, &bit| (acc << 1) | u8::from(bit))
            })
            .collect()
    }
}
