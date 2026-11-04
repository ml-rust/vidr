//! The `NAME.samples` framing the fixture scripts write.

/// Splits the `NAME.samples` framing: a 4-byte big-endian size before each
/// sample. Both fixture loaders read it.
pub(crate) fn unframe(bytes: &[u8]) -> Vec<Vec<u8>> {
    let mut samples = Vec::new();
    let mut rest = bytes;
    while rest.len() >= 4 {
        let size = u32::from_be_bytes([rest[0], rest[1], rest[2], rest[3]]) as usize;
        assert!(
            rest.len() >= 4 + size,
            "fixture sample framing is cut short"
        );
        samples.push(rest[4..4 + size].to_vec());
        rest = &rest[4 + size..];
    }
    assert!(rest.is_empty(), "fixture sample framing has trailing bytes");
    samples
}
