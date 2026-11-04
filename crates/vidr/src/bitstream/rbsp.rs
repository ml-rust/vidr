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

/// The offset in `nal` of the byte at offset `rbsp_offset` of `unescape(nal)`.
/// It counts each emulation prevention byte before that byte.
#[cfg(feature = "hevc")]
pub(crate) fn escaped_offset(nal: &[u8], rbsp_offset: usize) -> usize {
    let mut kept = 0usize;
    let mut zeros = 0usize;
    for (index, &byte) in nal.iter().enumerate() {
        if zeros >= 2 && byte == 3 {
            zeros = 0;
            continue;
        }
        if kept == rbsp_offset {
            return index;
        }
        kept += 1;
        zeros = if byte == 0 { zeros + 1 } else { 0 };
    }
    nal.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "hevc")]
    #[test]
    fn escaped_offset_counts_removed_bytes() {
        let nal = [1, 0, 0, 3, 1, 0, 0, 3, 0];
        let rbsp = unescape(&nal);
        assert_eq!(rbsp, vec![1, 0, 0, 1, 0, 0, 0]);
        for (offset, &byte) in rbsp.iter().enumerate() {
            assert_eq!(nal[escaped_offset(&nal, offset)], byte, "offset {offset}");
        }
        assert_eq!(escaped_offset(&nal, 3), 4);
        assert_eq!(escaped_offset(&nal, 6), 8);
        assert_eq!(escaped_offset(&nal, 7), nal.len());
    }

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
