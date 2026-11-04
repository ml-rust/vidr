//! MD5 (RFC 1321), for libvpx's per-frame hashes and JCT-VC's whole-stream hashes.

/// Each round's left rotation.
const SHIFTS: [u32; 64] = [
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9,
    14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10, 15,
    21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
];

/// The table's value `floor(2^32 * |sin(index + 1)|)`.
fn constant(index: usize) -> u32 {
    (((index + 1) as f64).sin().abs() * 4_294_967_296.0) as u32
}

fn block(state: &mut [u32; 4], constants: &[u32; 64], bytes: &[u8; 64]) {
    let mut words = [0_u32; 16];
    for (word, chunk) in words.iter_mut().zip(bytes.as_chunks::<4>().0) {
        *word = u32::from_le_bytes(*chunk);
    }
    let [mut a, mut b, mut c, mut d] = *state;
    for (index, shift) in SHIFTS.into_iter().enumerate() {
        let (mix, word) = match index / 16 {
            0 => ((b & c) | (!b & d), index),
            1 => ((d & b) | (!d & c), (5 * index + 1) % 16),
            2 => (b ^ c ^ d, (3 * index + 5) % 16),
            _ => (c ^ (b | !d), (7 * index) % 16),
        };
        let sum = a
            .wrapping_add(mix)
            .wrapping_add(constants[index])
            .wrapping_add(words[word]);
        (a, d, c) = (d, c, b);
        b = b.wrapping_add(sum.rotate_left(shift));
    }
    for (stored, value) in state.iter_mut().zip([a, b, c, d]) {
        *stored = stored.wrapping_add(value);
    }
}

/// An MD5 that takes its input in parts, and holds at most one block of it.
pub(super) struct Md5 {
    state: [u32; 4],
    constants: [u32; 64],
    pending: Vec<u8>,
    total: u64,
}

impl Md5 {
    pub(super) fn new() -> Self {
        Self {
            state: [0x6745_2301, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476],
            constants: std::array::from_fn(constant),
            pending: Vec::with_capacity(128),
            total: 0,
        }
    }

    /// Adds the next part of the input.
    pub(super) fn update(&mut self, part: &[u8]) {
        self.total = self.total.wrapping_add(part.len() as u64);
        let mut rest = part;
        if !self.pending.is_empty() {
            let taken = rest.len().min(64 - self.pending.len());
            self.pending.extend_from_slice(&rest[..taken]);
            rest = &rest[taken..];
            if self.pending.len() < 64 {
                return;
            }
            if let Some(bytes) = self.pending.first_chunk::<64>() {
                block(&mut self.state, &self.constants, bytes);
            }
            self.pending.clear();
        }
        let (blocks, tail) = rest.as_chunks::<64>();
        for bytes in blocks {
            block(&mut self.state, &self.constants, bytes);
        }
        self.pending.extend_from_slice(tail);
    }

    /// The digest in lowercase hexadecimal.
    pub(super) fn finish(mut self) -> String {
        self.pending.push(0x80);
        while self.pending.len() % 64 != 56 {
            self.pending.push(0);
        }
        self.pending
            .extend_from_slice(&self.total.wrapping_mul(8).to_le_bytes());
        for bytes in self.pending.as_chunks::<64>().0 {
            block(&mut self.state, &self.constants, bytes);
        }
        self.state
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }
}

/// The MD5 of `parts` in order, in lowercase hexadecimal.
pub(super) fn hex_digest(parts: &[&[u8]]) -> String {
    let mut md5 = Md5::new();
    for part in parts {
        md5.update(part);
    }
    md5.finish()
}

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
    for (text, digest) in cases {
        assert_eq!(hex_digest(&[text.as_bytes()]), digest, "`{text}`");
    }
}

/// The digest does not depend on how the input splits into parts.
#[test]
fn parts_join_at_any_boundary() {
    let text = [0x5a_u8; 200];
    let whole = hex_digest(&[&text]);
    for split in [0, 1, 55, 63, 64, 65, 128, 199, 200] {
        let (left, right) = text.split_at(split);
        assert_eq!(hex_digest(&[left, right]), whole, "split at {split}");
    }
}

/// A long input in small parts equals its one-part digest.
#[test]
fn updates_of_any_size_equal_one_update() {
    let text: Vec<u8> = (0..1000_u32).map(|value| value as u8).collect();
    for size in [1, 7, 63, 64, 65, 999] {
        let mut md5 = Md5::new();
        for part in text.chunks(size) {
            md5.update(part);
        }
        assert_eq!(md5.finish(), hex_digest(&[&text]), "parts of {size}");
    }
}
