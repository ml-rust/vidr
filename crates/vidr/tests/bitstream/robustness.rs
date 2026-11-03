//! Truncated and corrupt inputs return errors and never panic.

use super::fixtures::{Fixture, load};

/// A deterministic xorshift generator for corruption offsets and values.
struct Noise(u64);

impl Noise {
    fn step(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// Every prefix of `bytes`, then copies with bytes overwritten at random.
fn variants(bytes: &[u8], noise: &mut Noise) -> Vec<Vec<u8>> {
    let mut out: Vec<Vec<u8>> = (0..bytes.len()).map(|end| bytes[..end].to_vec()).collect();
    if !bytes.is_empty() {
        for _ in 0..64 {
            let mut copy = bytes.to_vec();
            for _ in 0..1 + noise.step() % 4 {
                let at = (noise.step() % bytes.len() as u64) as usize;
                copy[at] = noise.step() as u8;
            }
            out.push(copy);
        }
    }
    out
}

/// Runs `parse` on every variant of the config, and `filter` on every
/// variant of every sample after a fresh `setup`.
fn exercise<C, F>(
    fixture: &Fixture,
    parse: impl Fn(&[u8]) -> Option<C>,
    setup: impl Fn(&C) -> F,
    filter: impl Fn(&mut F, &[u8]),
) {
    let mut noise = Noise(0x9e37_79b9_7f4a_7c15);
    for config in variants(&fixture.config, &mut noise) {
        let _ = parse(&config);
    }
    let Some(config) = parse(&fixture.config) else {
        panic!("the fixture config does not parse");
    };
    let mut state = setup(&config);
    for sample in &fixture.samples {
        for variant in variants(sample, &mut noise) {
            filter(&mut state, &variant);
        }
        filter(&mut state, sample);
    }
}

#[cfg(feature = "h264")]
#[test]
fn h264_filters_survive_corruption() {
    use vidr::bitstream::{AvcConfig, FromAnnexB, NalCodec, ToAnnexB};
    let Some(fixture) = load("h264") else { return };
    exercise(
        &fixture,
        |bytes| AvcConfig::parse(bytes).ok(),
        |config| {
            (
                ToAnnexB::h264(config),
                FromAnnexB::new(NalCodec::H264, config.length_size),
            )
        },
        |(to, from), sample| {
            if let Ok(unit) = to.filter(sample) {
                let _ = from.filter(&unit);
            }
            let _ = from.filter(sample);
        },
    );
}

#[cfg(feature = "hevc")]
#[test]
fn hevc_filters_survive_corruption() {
    use vidr::bitstream::{FromAnnexB, HevcConfig, NalCodec, ToAnnexB};
    let Some(fixture) = load("hevc") else { return };
    exercise(
        &fixture,
        |bytes| HevcConfig::parse(bytes).ok(),
        |config| {
            (
                ToAnnexB::hevc(config),
                FromAnnexB::new(NalCodec::Hevc, config.length_size),
            )
        },
        |(to, from), sample| {
            if let Ok(unit) = to.filter(sample) {
                let _ = from.filter(&unit);
            }
            let _ = from.filter(sample);
        },
    );
}

#[cfg(feature = "av1")]
#[test]
fn av1_filters_survive_corruption() {
    use vidr::bitstream::av1::{Av1Config, Av1FromEncoder, Av1ToDecoder};
    let Some(fixture) = load("av1") else { return };
    exercise(
        &fixture,
        |bytes| Av1Config::parse(bytes).ok(),
        |config| (Av1ToDecoder::new(config).ok(), Av1FromEncoder::new()),
        |(to, from), sample| {
            if let Some(Ok(unit)) = to.as_mut().map(|to| to.filter(sample)) {
                let _ = from.filter(&unit);
            }
            let _ = from.filter(sample);
        },
    );
}
