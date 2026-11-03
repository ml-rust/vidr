//! AV1 filters against ffmpeg's `obu` muxer and MP4 muxer.

use vidr::bitstream::av1::{Av1Config, Av1FromEncoder, Av1ToDecoder, ObuType, obu};

use super::fixtures::load;

fn to_decoder(config: &Av1Config, samples: &[Vec<u8>]) -> Vec<Vec<u8>> {
    let mut filter = Av1ToDecoder::new(config).unwrap();
    samples
        .iter()
        .map(|sample| filter.filter(sample).unwrap())
        .collect()
}

#[test]
fn to_decoder_matches_ffmpeg() {
    let Some(fixture) = load("av1") else { return };
    let config = Av1Config::parse(&fixture.config).unwrap();
    let units = to_decoder(&config, &fixture.samples);
    assert_eq!(
        units.concat(),
        fixture.raw,
        "output differs from ffmpeg's obu muxer"
    );
}

#[test]
fn first_sample_without_sequence_header_gets_the_av1c_one() {
    let Some(fixture) = load("av1") else { return };
    let config = Av1Config::parse(&fixture.config).unwrap();
    let mut samples = fixture.samples.clone();
    let first: Vec<u8> = obu::split(&samples[0])
        .map(Result::unwrap)
        .filter(|parsed| parsed.header.obu_type != ObuType::SEQUENCE_HEADER)
        .flat_map(|parsed| parsed.bytes.to_vec())
        .collect();
    assert!(
        first.len() < samples[0].len(),
        "the first sample carries a sequence header"
    );
    samples[0] = first;
    assert_eq!(
        to_decoder(&config, &samples),
        to_decoder(&config, &fixture.samples)
    );
}

#[test]
fn real_av1c_round_trips() {
    let Some(fixture) = load("av1") else { return };
    let config = Av1Config::parse(&fixture.config).unwrap();
    assert!(config.sequence_header().unwrap().is_some());
    assert_eq!(config.to_bytes().unwrap(), fixture.config);
}

#[test]
fn from_encoder_reproduces_samples_and_av1c() {
    let Some(fixture) = load("av1") else { return };
    let config = Av1Config::parse(&fixture.config).unwrap();
    let mut filter = Av1FromEncoder::new();
    for (index, unit) in to_decoder(&config, &fixture.samples).iter().enumerate() {
        assert_eq!(
            filter.filter(unit).unwrap(),
            fixture.samples[index],
            "sample {index}"
        );
    }
    assert_eq!(filter.config().unwrap().to_bytes().unwrap(), fixture.config);
}

#[test]
fn from_encoder_takes_the_obu_stream() {
    let Some(fixture) = load("av1") else { return };
    // Split ffmpeg's OBU stream at its temporal delimiters.
    let mut units: Vec<Vec<u8>> = Vec::new();
    for parsed in obu::split(&fixture.raw).map(Result::unwrap) {
        if parsed.header.obu_type == ObuType::TEMPORAL_DELIMITER {
            units.push(Vec::new());
        }
        units.last_mut().unwrap().extend_from_slice(parsed.bytes);
    }
    let mut filter = Av1FromEncoder::new();
    let samples: Vec<Vec<u8>> = units
        .iter()
        .map(|unit| filter.filter(unit).unwrap())
        .collect();
    assert_eq!(samples, fixture.samples);
}
