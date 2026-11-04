//! Compares the header parsers with ffmpeg's `trace_headers` log.
//!
//! `tests/fixtures/gen_headers.py` writes one `NAME.headers` sidecar per stream.
//! Each test feeds every header through its parser and rebuilds the syntax
//! elements the header holds under ffmpeg's names. The two lists must match.
//! A name in an `IGNORED` list is an element the parser does not keep.

use std::path::PathBuf;

/// One header with the values `trace_headers` logs for it.
struct Section {
    kind: String,
    raw: Vec<u8>,
    elements: Vec<(String, i64)>,
}

fn directory() -> PathBuf {
    std::env::var_os("VIDR_BITSTREAM_FIXTURES").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/bitstream"),
        PathBuf::from,
    )
}

fn decode_hex(text: &str) -> Vec<u8> {
    text.as_bytes()
        .chunks(2)
        .map(|pair| {
            let digits = std::str::from_utf8(pair).unwrap_or("");
            u8::from_str_radix(digits, 16).unwrap_or_else(|_| panic!("bad hex `{digits}`"))
        })
        .collect()
}

/// Loads a sidecar, or prints a notice and returns `None` when it is missing.
fn load(name: &str) -> Option<Vec<Section>> {
    let path = directory().join(format!("{name}.headers"));
    let Ok(text) = std::fs::read_to_string(&path) else {
        eprintln!(
            "skipping: header fixture {} is missing. Run \
             `python3 crates/vidr/tests/fixtures/gen_bitstream.py`, or set \
             VIDR_BITSTREAM_FIXTURES to its output directory.",
            path.display()
        );
        return None;
    };
    let mut sections = Vec::new();
    for line in text.lines() {
        if let Some(header) = line.strip_prefix("@ ") {
            let (kind, raw) = header
                .split_once(' ')
                .unwrap_or_else(|| panic!("{name}: bad section line `{line}`"));
            sections.push(Section {
                kind: kind.to_owned(),
                raw: decode_hex(raw),
                elements: Vec::new(),
            });
        } else {
            let (element, value) = line
                .split_once(' ')
                .unwrap_or_else(|| panic!("{name}: bad element line `{line}`"));
            let value = value
                .parse::<i64>()
                .unwrap_or_else(|_| panic!("{name}: bad value in `{line}`"));
            let Some(section) = sections.last_mut() else {
                panic!("{name}: element before the first section: `{line}`");
            };
            section.elements.push((element.to_owned(), value));
        }
    }
    Some(sections)
}

/// The elements a parser holds, under ffmpeg's names.
#[derive(Default)]
struct Elements(Vec<(String, i64)>);

impl Elements {
    fn put(&mut self, name: impl Into<String>, value: impl Into<i64>) {
        self.0.push((name.into(), value.into()));
    }
}

/// Compares `oracle`, without its `ignored` names, against `emitted`. The order
/// of different names does not matter. The order of a repeated name does.
fn compare(label: &str, oracle: &[(String, i64)], ignored: &[&[&str]], emitted: Elements) {
    let is_ignored = |name: &str| {
        let base = name.split('[').next().unwrap_or(name);
        ignored.iter().any(|list| list.contains(&base))
    };
    let mut expected: Vec<(String, i64)> = oracle
        .iter()
        .filter(|(name, _)| !is_ignored(name))
        .cloned()
        .collect();
    let mut actual = emitted.0;
    expected.sort_by(|a, b| a.0.cmp(&b.0));
    actual.sort_by(|a, b| a.0.cmp(&b.0));
    for (want, got) in expected.iter().zip(&actual) {
        assert_eq!(
            want, got,
            "{label}: ffmpeg on the left, the parser on the right"
        );
    }
    assert_eq!(
        expected.len(),
        actual.len(),
        "{label}: element counts differ. Extra in ffmpeg: {:?}. Extra in the parser: {:?}.",
        expected.iter().skip(actual.len()).collect::<Vec<_>>(),
        actual.iter().skip(expected.len()).collect::<Vec<_>>()
    );
}

#[cfg(feature = "hevc")]
#[path = "header_oracle_hevc.rs"]
mod hevc;
#[cfg(feature = "vp9")]
#[path = "header_oracle_vp9.rs"]
mod vp9;
