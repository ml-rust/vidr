//! `pcm_sample()` (7.3.8.7) and the restart that follows it (9.3.1).

use crate::decode::cpu::hevc::error::HevcError;

use super::parser::CtuParser;

impl CtuParser<'_, '_> {
    /// Reads the samples after a `pcm_flag` of 1, then restarts the engine. The
    /// sample counts are multiples of 8, so the samples end on a byte boundary.
    pub(super) fn pcm_sample(&mut self, log2_size: u32) -> Result<(), HevcError> {
        let Some(pcm) = self.params.pcm else {
            return Ok(());
        };
        let start = self.cabac.engine.aligned_byte()?;
        let luma_count = 1usize << (2 * log2_size);
        // ( 2 << ( log2CbSize << 1 ) ) / ( SubWidthC * SubHeightC ).
        let chroma_count = luma_count / 2;
        let bits =
            luma_count * pcm.bit_depth_luma as usize + chroma_count * pcm.bit_depth_chroma as usize;
        let end = start + bits / 8;
        let data = self.cabac.engine.data();
        let Some(bytes) = data.get(start..end) else {
            return Err(HevcError::SliceDataTruncated {
                read_bits: end as u64 * 8,
                size: data.len(),
            });
        };
        let samples = &mut self.records.pcm_samples;
        samples
            .try_reserve(luma_count + chroma_count)
            .map_err(|_| HevcError::Allocation {
                what: "PCM samples",
                count: luma_count + chroma_count,
            })?;
        let mut reader = PcmReader {
            bytes: bytes.iter(),
            held: 0,
            held_bits: 0,
        };
        samples.extend((0..luma_count).map(|_| reader.read(pcm.bit_depth_luma)));
        samples.extend((0..chroma_count).map(|_| reader.read(pcm.bit_depth_chroma)));
        self.cabac.engine.restart_at(end)
    }
}

/// Reads fixed-length values of at most 16 bits, most significant bit first.
struct PcmReader<'a> {
    bytes: std::slice::Iter<'a, u8>,
    held: u32,
    held_bits: u32,
}

impl PcmReader<'_> {
    fn read(&mut self, count: u32) -> u16 {
        while self.held_bits < count {
            let byte = self.bytes.next().copied().unwrap_or(0);
            self.held = (self.held << 8) | u32::from(byte);
            self.held_bits += 8;
        }
        self.held_bits -= count;
        let value = self.held >> self.held_bits;
        self.held &= (1 << self.held_bits) - 1;
        (value & 0xffff) as u16
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_cross_byte_boundaries() {
        let bytes = [0b1011_0011, 0b1000_1111, 0xff];
        let mut reader = PcmReader {
            bytes: bytes.iter(),
            held: 0,
            held_bits: 0,
        };
        assert_eq!(reader.read(3), 0b101);
        assert_eq!(reader.read(10), 0b10_0111_0001);
        assert_eq!(reader.read(3), 0b111);
        assert_eq!(reader.read(8), 0xff);
        // Past the end, bits read as zero.
        assert_eq!(reader.read(4), 0);
    }
}
