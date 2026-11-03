//! Raw byte sequence payloads: NAL unit payloads without emulation
//! prevention bytes.

/// Returns `nal` with every emulation prevention byte removed: the `03` of
/// each `00 00 03`.
pub(crate) fn unescape(nal: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(nal.len());
    let mut zeros = 0usize;
    for &byte in nal {
        if zeros >= 2 && byte == 3 {
            zeros = 0;
            continue;
        }
        zeros = if byte == 0 { zeros + 1 } else { 0 };
        out.push(byte);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn removes_each_emulation_prevention_byte() {
        assert_eq!(
            unescape(&[1, 0, 0, 3, 1, 0, 0, 3, 0, 0, 3]),
            vec![1, 0, 0, 1, 0, 0, 0, 0]
        );
        // The byte after a removed `03` starts a new zero count.
        assert_eq!(unescape(&[0, 0, 3, 0, 3]), vec![0, 0, 0, 3]);
        assert_eq!(unescape(&[0, 3, 0, 0, 0]), vec![0, 3, 0, 0, 0]);
    }
}
