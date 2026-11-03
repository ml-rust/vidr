//! H.264 filters against ffmpeg's `h264_mp4toannexb` and MP4 muxer.

use vidr::bitstream::{
    AvcConfig, FromAnnexB, H264NalType, LengthSize, NalCodec, NalConfig, ToAnnexB,
};

use super::fixtures::{load, nal_units, sample_of};

fn to_annexb(config: &AvcConfig, samples: &[Vec<u8>]) -> Vec<Vec<u8>> {
    let mut filter = ToAnnexB::h264(config);
    samples
        .iter()
        .map(|sample| filter.filter(sample).unwrap())
        .collect()
}

fn check_matches_ffmpeg(name: &str) {
    let Some(fixture) = load(name) else { return };
    let config = AvcConfig::parse(&fixture.config).unwrap();
    let units = to_annexb(&config, &fixture.samples);
    assert_eq!(
        units.concat(),
        fixture.raw,
        "{name}: output differs from h264_mp4toannexb"
    );
}

#[test]
fn to_annexb_matches_ffmpeg() {
    check_matches_ffmpeg("h264");
}

#[test]
fn to_annexb_with_in_band_parameter_sets_matches_ffmpeg() {
    check_matches_ffmpeg("h264_inband");
}

#[test]
fn real_avcc_round_trips() {
    for name in ["h264", "h264_inband"] {
        let Some(fixture) = load(name) else { return };
        let config = AvcConfig::parse(&fixture.config).unwrap();
        assert_eq!(config.length_size, LengthSize::Four);
        assert_eq!(config.sps.len(), 1);
        assert_eq!(config.pps.len(), 1);
        assert!(config.high_profile.is_some(), "x264 writes High profile");
        assert_eq!(config.to_bytes().unwrap(), fixture.config, "{name}");
    }
}

#[test]
fn from_annexb_reproduces_samples_and_avcc() {
    let Some(fixture) = load("h264") else { return };
    let config = AvcConfig::parse(&fixture.config).unwrap();
    let mut filter = FromAnnexB::new(NalCodec::H264, config.length_size);
    for (index, unit) in to_annexb(&config, &fixture.samples).iter().enumerate() {
        assert_eq!(
            filter.filter(unit).unwrap(),
            fixture.samples[index],
            "sample {index}"
        );
    }
    let Some(NalConfig::Avc(built)) = filter.config() else {
        panic!("no avcC after the first access unit");
    };
    assert_eq!(built.to_bytes().unwrap(), fixture.config);
}

#[test]
fn from_annexb_removes_in_band_parameter_sets() {
    let Some(fixture) = load("h264_inband") else {
        return;
    };
    let config = AvcConfig::parse(&fixture.config).unwrap();
    let mut filter = FromAnnexB::new(NalCodec::H264, LengthSize::Four);
    for (index, unit) in to_annexb(&config, &fixture.samples).iter().enumerate() {
        let kept: Vec<&[u8]> = nal_units(&fixture.samples[index])
            .into_iter()
            .filter(|unit| {
                H264NalType::of(unit)
                    .and_then(H264NalType::parameter_set)
                    .is_none()
            })
            .collect();
        assert_eq!(
            filter.filter(unit).unwrap(),
            sample_of(&kept),
            "sample {index}"
        );
    }
    let Some(NalConfig::Avc(built)) = filter.config() else {
        panic!("no avcC after the first access unit");
    };
    assert_eq!(built.to_bytes().unwrap(), fixture.config);
}
