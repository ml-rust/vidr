//! What a conformance zip states of its decoded output, for each decoder's sweep.

use std::path::Path;

#[cfg(feature = "cpu-hevc")]
use vidr::bitstream::HevcNalType;

use super::hevc::{BITSTREAMS, HashKind, md5_token, read_stream, unzip};
use super::md5::hex_digest;

/// `payloadType` of the decoded picture hash SEI message (D.2.1).
#[cfg(feature = "cpu-hevc")]
const PICTURE_HASH_PAYLOAD: u8 = 132;

/// True for a suffix SEI NAL unit of the base layer whose first message is a picture hash.
#[cfg(feature = "cpu-hevc")]
pub(super) fn holds_picture_hash(nal: &[u8]) -> bool {
    let [first, second, payload, ..] = nal else {
        return false;
    };
    HevcNalType::from_header(*first) == HevcNalType::SEI_SUFFIX
        && first & 1 == 0
        && second >> 3 == 0
        && *payload == PICTURE_HASH_PAYLOAD
}

/// The whole-stream MD5 a hash file publishes. A line that names a `.yuv` file wins.
/// A file named for the bitstream states none. In a file with no such name, the first
/// digest wins whose line names no bitstream. That digest also differs from the bitstream's.
fn published_yuv_md5(kind: HashKind, text: &str, bitstream_md5: &str) -> Option<String> {
    let named = text
        .lines()
        .filter(|line| line.to_ascii_lowercase().contains("yuv"))
        .find_map(md5_token);
    if named.is_some() {
        return named;
    }
    match kind {
        HashKind::Bitstream => None,
        HashKind::Yuv => md5_token(text),
        HashKind::Unnamed => text
            .lines()
            .filter(|line| {
                let lower = line.to_ascii_lowercase();
                !BITSTREAMS.iter().any(|ending| lower.contains(ending))
            })
            .filter_map(md5_token)
            .find(|digest| digest != bitstream_md5),
    }
}

/// The bitstream, and the whole-stream MD5 with the member that states it.
pub(super) type ZipStream = (Vec<u8>, Option<(String, String)>);

pub(super) fn read_zip(zip: &Path) -> Result<ZipStream, String> {
    let (stream, hash) = read_stream(zip)?;
    let Some((hash, kind)) = hash else {
        return Ok((stream, None));
    };
    let text = unzip("-p", zip, Some(&hash))?;
    let published = published_yuv_md5(kind, &text, &hex_digest(&[&stream]));
    Ok((stream, published.map(|digest| (hash, digest))))
}

#[test]
fn each_hash_file_form_gives_its_digest() {
    let bit = "ffc9e0afd09cb90dad7706b252f15dcd";
    let yuv = "277006085088cf68b9cafbbba0c19443";
    // `X.yuv.md5` with the bare digest in uppercase.
    assert_eq!(
        published_yuv_md5(HashKind::Yuv, &yuv.to_ascii_uppercase(), bit).as_deref(),
        Some(yuv)
    );
    // `X.md5` with the bare digest, which differs from the bitstream's.
    assert_eq!(
        published_yuv_md5(HashKind::Unnamed, yuv, bit).as_deref(),
        Some(yuv)
    );
    // `X.md5` holding the bitstream's own digest states no decoded hash.
    assert_eq!(published_yuv_md5(HashKind::Unnamed, bit, bit), None);
    // `X.bit.md5` states none.
    assert_eq!(published_yuv_md5(HashKind::Bitstream, bit, bit), None);
    // `X.md5` with a named file.
    let named = format!("{yuv} *BUMPING_A_ericsson.yuv\n");
    assert_eq!(
        published_yuv_md5(HashKind::Unnamed, &named, bit).as_deref(),
        Some(yuv)
    );
    // Two lines: the one that names the decoded file wins.
    let both = format!("{bit}  x.bit\n{yuv}  x.yuv\n");
    assert_eq!(
        published_yuv_md5(HashKind::Unnamed, &both, bit).as_deref(),
        Some(yuv)
    );
    assert_eq!(published_yuv_md5(HashKind::Unnamed, "none\n", bit), None);
    // MD5summer's comment lines, then a line that names the bitstream.
    let summer = format!("# Generated 4/16/2013\n{yuv} *x.bit\n");
    assert_eq!(published_yuv_md5(HashKind::Unnamed, &summer, bit), None);
}
