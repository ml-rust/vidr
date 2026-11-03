//! Generated media fixtures for the bitstream tests.
//!
//! `tests/fixtures/gen_bitstream.py` writes them. A test whose fixture is
//! missing prints a notice and passes without checks.

use std::path::PathBuf;

/// The streams of one fixture name.
pub(crate) struct Fixture {
    /// The `avcC`, `hvcC`, or `av1C` payload.
    pub(crate) config: Vec<u8>,
    /// Every MP4 sample, in decode order.
    pub(crate) samples: Vec<Vec<u8>>,
    /// ffmpeg's raw Annex-B or OBU conversion of the samples.
    pub(crate) raw: Vec<u8>,
}

fn directory() -> PathBuf {
    std::env::var_os("VIDR_BITSTREAM_FIXTURES").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/bitstream"),
        PathBuf::from,
    )
}

/// Splits the `NAME.samples` framing: a 4-byte big-endian size before each
/// sample. `tests/nvdec/` reads the same framing.
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

/// Loads fixture `name`, or prints a notice and returns `None` when a file
/// is missing.
pub(crate) fn load(name: &str) -> Option<Fixture> {
    let dir = directory();
    let read = |extension: &str| std::fs::read(dir.join(format!("{name}.{extension}"))).ok();
    match (read("config"), read("samples"), read("raw")) {
        (Some(config), Some(samples), Some(raw)) => Some(Fixture {
            config,
            samples: unframe(&samples),
            raw,
        }),
        _ => {
            eprintln!(
                "skipping: bitstream fixture `{name}` is missing in {}. \
                 Run `python3 crates/vidr/tests/fixtures/gen_bitstream.py`, \
                 or set VIDR_BITSTREAM_FIXTURES to its output directory.",
                dir.display()
            );
            None
        }
    }
}

/// Splits length-prefixed NAL units with a 4-byte size, for test oracles.
#[cfg(any(feature = "h264", feature = "hevc"))]
pub(crate) fn nal_units(sample: &[u8]) -> Vec<&[u8]> {
    let mut units = Vec::new();
    let mut rest = sample;
    while rest.len() >= 4 {
        let size = u32::from_be_bytes([rest[0], rest[1], rest[2], rest[3]]) as usize;
        units.push(&rest[4..4 + size]);
        rest = &rest[4 + size..];
    }
    units
}

/// Joins NAL units with 4-byte sizes, for test oracles.
#[cfg(any(feature = "h264", feature = "hevc"))]
pub(crate) fn sample_of(units: &[&[u8]]) -> Vec<u8> {
    let mut out = Vec::new();
    for unit in units {
        out.extend_from_slice(&(unit.len() as u32).to_be_bytes());
        out.extend_from_slice(unit);
    }
    out
}
