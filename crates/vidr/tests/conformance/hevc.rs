//! The layout of the JCT-VC HEVC_v1 zips: which member is the bitstream, and which
//! hash goes with it.
//!
//! The test here needs no decoder, so it compiles with no features. It reads each zip
//! through the `unzip` tool, and prints what it finds. `hevc_decode` decodes each
//! stream with the `cpu-hevc` feature, and reads the zips through this module.
//!
//! `scripts/conformance/fetch_hevc.sh` fetches the zips. Set
//! `VIDR_HEVC_CONFORMANCE` to that directory, and `VIDR_HEVC_ONLY` to run the zips
//! whose name holds that text.

use std::path::{Path, PathBuf};
use std::process::Command;

/// The file name endings of a bitstream member, in lowercase.
pub(super) const BITSTREAMS: [&str; 6] = [".bit", ".bin", ".hevc", ".265", ".h265", ".hvc"];

/// The file name endings of a hash member, in lowercase.
const HASHES: [&str; 2] = [".md5", "_md5.txt"];

/// What a hash member covers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum HashKind {
    /// The decoded picture file.
    Yuv,
    /// The bitstream file.
    Bitstream,
    /// Its name states neither.
    Unnamed,
}

/// One zip's lookup.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct Layout {
    pub(super) bitstream: Option<String>,
    pub(super) hash: Option<(String, HashKind)>,
}

fn hash_kind(lower: &str) -> HashKind {
    let stem = HASHES
        .iter()
        .find_map(|ending| lower.strip_suffix(ending))
        .unwrap_or(lower);
    if stem.contains("yuv") {
        HashKind::Yuv
    } else if stem.ends_with("_bit")
        || stem.ends_with("_bin")
        || BITSTREAMS.iter().any(|ending| stem.ends_with(ending))
    {
        HashKind::Bitstream
    } else {
        HashKind::Unnamed
    }
}

/// True for the resource fork copies that macOS adds to a zip.
fn is_resource_fork(lower: &str) -> bool {
    lower.starts_with("__macosx/")
        || lower
            .rsplit('/')
            .next()
            .is_some_and(|name| name.starts_with("._"))
}

/// Picks the bitstream and the hash from a zip's member names. A hash of the
/// decoded pictures wins over any other.
pub(super) fn lookup<'a>(members: impl Iterator<Item = &'a str>) -> Layout {
    let mut layout = Layout {
        bitstream: None,
        hash: None,
    };
    for member in members {
        let lower = member.to_ascii_lowercase();
        if is_resource_fork(&lower) {
            continue;
        }
        if BITSTREAMS.iter().any(|ending| lower.ends_with(ending)) {
            layout.bitstream.get_or_insert_with(|| member.to_owned());
        }
        if HASHES.iter().any(|ending| lower.ends_with(ending)) {
            let kind = hash_kind(&lower);
            let better = match &layout.hash {
                None => true,
                Some((_, held)) => kind == HashKind::Yuv && *held != HashKind::Yuv,
            };
            if better {
                layout.hash = Some((member.to_owned(), kind));
            }
        }
    }
    layout
}

/// The first run of 32 hexadecimal digits in a hash file, in lowercase.
pub(super) fn md5_token(text: &str) -> Option<String> {
    text.split(|letter: char| !letter.is_ascii_hexdigit())
        .find(|word| word.len() == 32)
        .map(str::to_ascii_lowercase)
}

/// The zip's name without `.zip`.
pub(super) fn zip_name(zip: &Path) -> String {
    zip.file_stem()
        .map_or_else(String::new, |name| name.to_string_lossy().into_owned())
}

/// The zips in `directory` whose name holds `VIDR_HEVC_ONLY`, by name.
pub(super) fn zips(directory: &Path) -> Vec<PathBuf> {
    let only = std::env::var("VIDR_HEVC_ONLY").ok();
    let entries = std::fs::read_dir(directory)
        .unwrap_or_else(|error| panic!("reading {}: {error}", directory.display()));
    let mut found: Vec<PathBuf> = entries
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().is_some_and(|ending| ending == "zip"))
        .filter(|path| {
            only.as_deref()
                .is_none_or(|text| zip_name(path).contains(text))
        })
        .collect();
    found.sort();
    found
}

/// Runs `unzip FLAG ZIP [MEMBER]`, and returns its bytes. A failed call is an error
/// that names the zip.
pub(super) fn unzip_bytes(flag: &str, zip: &Path, member: Option<&str>) -> Result<Vec<u8>, String> {
    let output = Command::new("unzip")
        .arg(flag)
        .arg(zip)
        .args(member)
        .output()
        .map_err(|error| format!("unzip does not start: {error}. Install unzip."))?;
    if !output.status.success() {
        return Err(format!(
            "unzip {flag} {} {member:?} exits with {}",
            zip.display(),
            output.status
        ));
    }
    Ok(output.stdout)
}

/// [`unzip_bytes`] as text.
pub(super) fn unzip(flag: &str, zip: &Path, member: Option<&str>) -> Result<String, String> {
    let bytes = unzip_bytes(flag, zip, member)?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

#[test]
fn hevc_zips_hold_a_bitstream_and_a_hash() {
    let Some(directory) = std::env::var_os("VIDR_HEVC_CONFORMANCE") else {
        eprintln!(
            "skipping: VIDR_HEVC_CONFORMANCE is unset. Run \
             `scripts/conformance/fetch_hevc.sh DIR`, then set it to DIR."
        );
        return;
    };
    let directory = PathBuf::from(directory);
    let zips = zips(&directory);
    assert!(
        !zips.is_empty(),
        "{} holds no zip for VIDR_HEVC_ONLY. Run `scripts/conformance/fetch_hevc.sh {}`.",
        directory.display(),
        directory.display()
    );
    println!("| Zip | Bitstream | Hash file | Covers | MD5 |");
    println!("| --- | --- | --- | --- | --- |");
    let mut failed = Vec::new();
    for zip in &zips {
        let name = zip.display().to_string();
        let listing = match unzip("-Z1", zip, None) {
            Ok(listing) => listing,
            Err(reason) => {
                failed.push(reason);
                continue;
            }
        };
        let layout = lookup(listing.lines());
        let Some(bitstream) = &layout.bitstream else {
            failed.push(format!("{name}: no member ends in {BITSTREAMS:?}"));
            continue;
        };
        let Some((hash, kind)) = &layout.hash else {
            println!("| {name} | {bitstream} | none | none | none |");
            continue;
        };
        let text = match unzip("-p", zip, Some(hash)) {
            Ok(text) => text,
            Err(reason) => {
                failed.push(reason);
                continue;
            }
        };
        let md5 = md5_token(&text).unwrap_or_else(|| "none".to_owned());
        println!("| {name} | {bitstream} | {hash} | {kind:?} | {md5} |");
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

#[test]
fn a_decoded_picture_hash_wins_over_a_bitstream_hash() {
    let layout = lookup(
        [
            "AMP_A_Samsung_7/AMP_A_Samsung_7.BIT.md5",
            "AMP_A_Samsung_7/AMP_A_Samsung_7.bit",
            "AMP_A_Samsung_7/AMP_A_Samsung_7_yuv.md5",
            "AMP_A_Samsung_7/readme.doc",
        ]
        .into_iter(),
    );
    assert_eq!(
        layout,
        Layout {
            bitstream: Some("AMP_A_Samsung_7/AMP_A_Samsung_7.bit".to_owned()),
            hash: Some((
                "AMP_A_Samsung_7/AMP_A_Samsung_7_yuv.md5".to_owned(),
                HashKind::Yuv
            )),
        }
    );
}

#[test]
fn a_zip_with_only_a_plain_hash_keeps_it() {
    let layout = lookup(["X.bin", "X.md5"].into_iter());
    assert_eq!(layout.bitstream.as_deref(), Some("X.bin"));
    assert_eq!(layout.hash, Some(("X.md5".to_owned(), HashKind::Unnamed)));
    assert_eq!(hash_kind("x.bit.md5"), HashKind::Bitstream);
}

#[test]
fn every_hash_name_in_the_set_has_its_kind() {
    for (name, kind) in [
        ("x.yuv.md5", HashKind::Yuv),
        ("x_yuv.md5", HashKind::Yuv),
        ("x.bit.md5", HashKind::Bitstream),
        ("x_bit.md5", HashKind::Bitstream),
        ("x_bin.md5", HashKind::Bitstream),
        ("x.md5", HashKind::Unnamed),
        ("x_md5.txt", HashKind::Unnamed),
    ] {
        assert_eq!(hash_kind(name), kind, "`{name}`");
    }
}

#[test]
fn resource_forks_are_not_members() {
    let layout = lookup(
        [
            "__MACOSX/X/._X_yuv.md5",
            "__MACOSX/X/._X.bit",
            "X/._Y.bit",
            "X/X.bit",
            "X/X_yuv.md5",
        ]
        .into_iter(),
    );
    assert_eq!(layout.bitstream.as_deref(), Some("X/X.bit"));
    assert_eq!(layout.hash, Some(("X/X_yuv.md5".to_owned(), HashKind::Yuv)));
}

#[test]
fn a_text_hash_member_counts() {
    let layout = lookup(["X/X.bin", "X/X.txt", "X/X_md5.txt"].into_iter());
    assert_eq!(
        layout.hash,
        Some(("X/X_md5.txt".to_owned(), HashKind::Unnamed))
    );
}

#[test]
fn the_token_reads_each_hash_file_form() {
    let digest = "D41D8CD98F00B204E9800998ECF8427E";
    let lower = digest.to_ascii_lowercase();
    for text in [
        format!("{digest}  x.yuv\n"),
        format!("{digest} *x.yuv\r\n"),
        format!("MD5 (x.yuv) = {digest}\n"),
    ] {
        assert_eq!(md5_token(&text), Some(lower.clone()), "`{text}`");
    }
    assert_eq!(md5_token("no hash here\n"), None);
}
