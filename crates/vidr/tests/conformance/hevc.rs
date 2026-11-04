//! The layout of the JCT-VC HEVC_v1 zips: which member is the bitstream, and which
//! hash goes with it.
//!
//! No HEVC software decoder exists in vidr yet, so nothing here decodes. The test
//! needs no decoder, so it compiles with no features. It reads each zip through the
//! `unzip` tool, and prints what it finds. The decoder's unit adds the comparison.
//!
//! `scripts/conformance/fetch_hevc.sh` fetches the zips. Set
//! `VIDR_HEVC_CONFORMANCE` to that directory.

use std::path::{Path, PathBuf};
use std::process::Command;

/// The file name endings of a bitstream member, in lowercase.
const BITSTREAMS: [&str; 6] = [".bit", ".bin", ".hevc", ".265", ".h265", ".hvc"];

/// What a hash member covers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HashKind {
    /// The decoded picture file.
    Yuv,
    /// The bitstream file.
    Bitstream,
    /// Its name states neither.
    Unnamed,
}

/// One zip's lookup.
#[derive(Debug, PartialEq, Eq)]
struct Layout {
    bitstream: Option<String>,
    hash: Option<(String, HashKind)>,
}

fn hash_kind(lower: &str) -> HashKind {
    let stem = lower.trim_end_matches(".md5");
    if stem.contains("yuv") {
        HashKind::Yuv
    } else if BITSTREAMS.iter().any(|ending| stem.ends_with(ending)) {
        HashKind::Bitstream
    } else {
        HashKind::Unnamed
    }
}

/// Picks the bitstream and the hash from a zip's member names. A hash of the
/// decoded pictures wins over any other.
fn lookup<'a>(members: impl Iterator<Item = &'a str>) -> Layout {
    let mut layout = Layout {
        bitstream: None,
        hash: None,
    };
    for member in members {
        let lower = member.to_ascii_lowercase();
        if BITSTREAMS.iter().any(|ending| lower.ends_with(ending)) {
            layout.bitstream.get_or_insert_with(|| member.to_owned());
        }
        if lower.ends_with(".md5") {
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
fn md5_token(text: &str) -> Option<String> {
    text.split(|letter: char| !letter.is_ascii_hexdigit())
        .find(|word| word.len() == 32)
        .map(str::to_ascii_lowercase)
}

fn zips(directory: &Path) -> Vec<PathBuf> {
    let entries = std::fs::read_dir(directory)
        .unwrap_or_else(|error| panic!("reading {}: {error}", directory.display()));
    let mut found: Vec<PathBuf> = entries
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().is_some_and(|ending| ending == "zip"))
        .collect();
    found.sort();
    found
}

/// Runs `unzip FLAG ZIP [MEMBER]`, and returns its output. A failed call is an
/// error that names the zip.
fn unzip(flag: &str, zip: &Path, member: Option<&str>) -> Result<String, String> {
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
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
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
        "{} holds no zip. Run `scripts/conformance/fetch_hevc.sh {}`.",
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
