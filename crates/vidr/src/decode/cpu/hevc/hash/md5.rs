//! MD5 (IETF RFC 1321), for the picture hash. The picture hash is a check against
//! corruption here, never a security function.

/// Each step's left rotation (RFC 1321, 3.4).
const SHIFTS: [u32; 64] = [
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9,
    14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10, 15,
    21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
];
/// The sine table of RFC 1321, 3.4.
const SINES: [u32; 64] = [
    0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
    0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
    0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
    0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
    0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
    0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
    0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
    0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391,
];
/// Bytes in one input block.
const BLOCK_BYTES: usize = 64;

/// A running digest.
#[derive(Clone, Debug)]
pub(crate) struct Md5 {
    state: [u32; 4],
    pending: [u8; BLOCK_BYTES],
    pending_len: usize,
    total_bytes: u64,
}

impl Md5 {
    /// `MD5Init`.
    pub(crate) const fn new() -> Self {
        Self {
            state: [0x6745_2301, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476],
            pending: [0; BLOCK_BYTES],
            pending_len: 0,
            total_bytes: 0,
        }
    }

    /// `MD5Update`.
    pub(crate) fn update(&mut self, mut bytes: &[u8]) {
        self.total_bytes = self.total_bytes.wrapping_add(bytes.len() as u64);
        if self.pending_len > 0 {
            let taken = bytes.len().min(BLOCK_BYTES - self.pending_len);
            let (head, rest) = bytes.split_at(taken);
            self.pending[self.pending_len..self.pending_len + taken].copy_from_slice(head);
            self.pending_len += taken;
            bytes = rest;
            if self.pending_len < BLOCK_BYTES {
                return;
            }
            let block = self.pending;
            self.compress(&block);
            self.pending_len = 0;
        }
        let (blocks, tail) = bytes.as_chunks::<BLOCK_BYTES>();
        for block in blocks {
            self.compress(block);
        }
        self.pending[..tail.len()].copy_from_slice(tail);
        self.pending_len = tail.len();
    }

    /// `MD5Final`.
    pub(crate) fn finish(mut self) -> [u8; 16] {
        let bits = self.total_bytes.wrapping_mul(8);
        self.update(&[0x80]);
        while self.pending_len != BLOCK_BYTES - 8 {
            self.update(&[0]);
        }
        self.update(&bits.to_le_bytes());
        let mut digest = [0u8; 16];
        for (out, word) in digest.as_chunks_mut::<4>().0.iter_mut().zip(self.state) {
            out.copy_from_slice(&word.to_le_bytes());
        }
        digest
    }

    /// One block (RFC 1321, 3.4).
    fn compress(&mut self, block: &[u8; BLOCK_BYTES]) {
        let mut words = [0u32; 16];
        for (word, bytes) in words.iter_mut().zip(block.as_chunks::<4>().0) {
            *word = u32::from_le_bytes(*bytes);
        }
        let [mut a, mut b, mut c, mut d] = self.state;
        for step in 0..BLOCK_BYTES {
            let (mix, word) = match step / 16 {
                0 => ((b & c) | (!b & d), step),
                1 => ((d & b) | (!d & c), (5 * step + 1) % 16),
                2 => (b ^ c ^ d, (3 * step + 5) % 16),
                _ => (c ^ (b | !d), (7 * step) % 16),
            };
            let sum = a
                .wrapping_add(mix)
                .wrapping_add(SINES[step])
                .wrapping_add(words[word]);
            (a, d, c) = (d, c, b);
            b = b.wrapping_add(sum.rotate_left(SHIFTS[step]));
        }
        for (stored, value) in self.state.iter_mut().zip([a, b, c, d]) {
            *stored = stored.wrapping_add(value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(bytes: &[u8]) -> [u8; 16] {
        let mut md5 = Md5::new();
        md5.update(bytes);
        md5.finish()
    }

    fn hex(bytes: &[u8; 16]) -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    /// The test suite of RFC 1321, A.5.
    #[test]
    fn rfc_1321_vectors_hold() {
        let cases: [(&str, &str); 7] = [
            ("", "d41d8cd98f00b204e9800998ecf8427e"),
            ("a", "0cc175b9c0f1b6a831c399e269772661"),
            ("abc", "900150983cd24fb0d6963f7d28e17f72"),
            ("message digest", "f96b697d7cb7938d525a2f31aaf161d0"),
            (
                "abcdefghijklmnopqrstuvwxyz",
                "c3fcd3d76192e4007dfb496cca67e13b",
            ),
            (
                "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
                "d174ab98d277d9f5a5611c2c9f419d9f",
            ),
            (
                "12345678901234567890123456789012345678901234567890123456789012345678901234567890",
                "57edf4a22be3c955ac49da2e2107b67a",
            ),
        ];
        for (text, expected) in cases {
            assert_eq!(hex(&digest(text.as_bytes())), expected, "{text}");
        }
    }

    #[test]
    fn the_digest_ignores_how_the_input_splits() {
        let text = [0x5au8; 200];
        let whole = digest(&text);
        for split in [0, 1, 55, 56, 63, 64, 65, 128, 199, 200] {
            let (left, right) = text.split_at(split);
            let mut md5 = Md5::new();
            md5.update(left);
            md5.update(right);
            assert_eq!(md5.finish(), whole, "split at {split}");
        }
    }

    /// Lengths around the padding boundary, where a second block starts.
    #[test]
    fn padding_lengths_run() {
        for length in 54..=66 {
            let bytes = vec![b'x'; length];
            assert_ne!(digest(&bytes), digest(&bytes[1..]));
        }
    }
}
