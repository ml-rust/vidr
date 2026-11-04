//! A bit writer and NAL unit framing for hand-built header tests.

/// Writes syntax elements, most significant bit first.
#[derive(Clone, Debug, Default)]
pub(crate) struct BitWriter {
    bits: Vec<bool>,
}

impl BitWriter {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Writes `value` in `count` bits.
    pub(crate) fn u(&mut self, count: u32, value: u64) -> &mut Self {
        for shift in (0..count).rev() {
            self.bits.push((value >> shift) & 1 == 1);
        }
        self
    }

    pub(crate) fn flag(&mut self, value: bool) -> &mut Self {
        self.bits.push(value);
        self
    }

    /// Writes an unsigned Exp-Golomb code.
    #[cfg(any(feature = "hevc", feature = "vulkan-video"))]
    pub(crate) fn ue(&mut self, value: u32) -> &mut Self {
        let code = u64::from(value) + 1;
        let length = 64 - code.leading_zeros();
        self.u(length - 1, 0).u(length, code)
    }

    /// Writes a signed Exp-Golomb code.
    #[cfg(any(feature = "hevc", feature = "vulkan-video"))]
    pub(crate) fn se(&mut self, value: i64) -> &mut Self {
        let code = if value > 0 { 2 * value - 1 } else { -2 * value };
        self.ue(code as u32)
    }

    /// Writes a stop bit, then zero bits to a byte boundary.
    #[cfg(any(feature = "hevc", feature = "vulkan-video"))]
    pub(crate) fn trailing(&mut self) -> &mut Self {
        self.bits.push(true);
        self.align()
    }

    /// Writes zero bits to a byte boundary.
    pub(crate) fn align(&mut self) -> &mut Self {
        while !self.bits.len().is_multiple_of(8) {
            self.bits.push(false);
        }
        self
    }

    /// The bits as bytes, with zero bits after the last one.
    pub(crate) fn bytes(&self) -> Vec<u8> {
        let mut out = vec![0u8; self.bits.len().div_ceil(8)];
        for (index, bit) in self.bits.iter().enumerate() {
            if *bit {
                out[index / 8] |= 0x80 >> (index % 8);
            }
        }
        out
    }
}

/// `bytes` with an emulation prevention byte before each byte that needs one.
#[cfg(any(feature = "hevc", feature = "vulkan-video"))]
pub(crate) fn escape(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len());
    let mut zeros = 0usize;
    for &byte in bytes {
        if zeros >= 2 && byte <= 3 {
            out.push(3);
            zeros = 0;
        }
        out.push(byte);
        zeros = if byte == 0 { zeros + 1 } else { 0 };
    }
    out
}

/// Builds an HEVC NAL unit with emulation prevention, at layer 0 and
/// `TemporalId` 0.
#[cfg(feature = "hevc")]
pub(crate) fn hevc_nal(nal_unit_type: u8, rbsp: &[u8]) -> Vec<u8> {
    let mut nal = vec![nal_unit_type << 1, 1];
    nal.extend(escape(rbsp));
    nal
}
