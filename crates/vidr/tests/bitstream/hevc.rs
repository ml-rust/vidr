//! HEVC filters against ffmpeg's `hevc_mp4toannexb` and MP4 muxer.

use vidr::bitstream::{
    FromAnnexB, HevcConfig, HevcNalType, LengthSize, NalCodec, NalConfig, ToAnnexB, hevc_sps_vui,
};

use super::fixtures::{load, nal_units, sample_of};

fn to_annexb(config: &HevcConfig, samples: &[Vec<u8>]) -> Vec<Vec<u8>> {
    let mut filter = ToAnnexB::hevc(config);
    samples
        .iter()
        .map(|sample| filter.filter(sample).unwrap())
        .collect()
}

fn config_annexb(config: &HevcConfig) -> Vec<u8> {
    let mut out = Vec::new();
    vidr::bitstream::annexb::join(config.nal_units(), &mut out).unwrap();
    out
}

fn has_irap(sample: &[u8]) -> bool {
    nal_units(sample)
        .iter()
        .any(|unit| HevcNalType::of(unit).is_some_and(HevcNalType::is_irap))
}

#[test]
fn to_annexb_matches_ffmpeg() {
    for name in ["hevc", "hevc_plain"] {
        let Some(fixture) = load(name) else { return };
        let config = HevcConfig::parse(&fixture.config).unwrap();
        let units = to_annexb(&config, &fixture.samples);
        assert_eq!(
            units.concat(),
            fixture.raw,
            "{name}: output differs from hevc_mp4toannexb"
        );
    }
}

/// ffmpeg puts the config before in-band VPS, SPS, and PPS. vidr does not.
/// With that one difference, the output matches byte for byte.
#[test]
fn to_annexb_with_in_band_parameter_sets_skips_ffmpeg_duplicate() {
    let Some(fixture) = load("hevc_inband") else {
        return;
    };
    let config = HevcConfig::parse(&fixture.config).unwrap();
    let prefix = config_annexb(&config);
    let units = to_annexb(&config, &fixture.samples);
    let mut expected = Vec::new();
    let mut irap_samples = 0;
    for (sample, unit) in fixture.samples.iter().zip(&units) {
        if has_irap(sample) {
            irap_samples += 1;
            expected.extend_from_slice(&prefix);
        }
        expected.extend_from_slice(unit);
    }
    assert!(irap_samples >= 2, "the fixture holds an IDR and a CRA");
    assert_eq!(expected, fixture.raw);
}

#[test]
fn real_hvcc_round_trips() {
    for name in ["hevc", "hevc_plain", "hevc_inband"] {
        let Some(fixture) = load(name) else { return };
        let config = HevcConfig::parse(&fixture.config).unwrap();
        assert_eq!(config.length_size, LengthSize::Four);
        assert_eq!(config.to_bytes().unwrap(), fixture.config, "{name}");
    }
}

#[test]
fn from_annexb_reproduces_samples_and_hvcc() {
    let Some(fixture) = load("hevc_plain") else {
        return;
    };
    let config = HevcConfig::parse(&fixture.config).unwrap();
    let mut filter = FromAnnexB::new(NalCodec::Hevc, config.length_size);
    for (index, unit) in to_annexb(&config, &fixture.samples).iter().enumerate() {
        assert_eq!(
            filter.filter(unit).unwrap(),
            fixture.samples[index],
            "sample {index}"
        );
    }
    let Some(NalConfig::Hevc(built)) = filter.config() else {
        panic!("no hvcC after the first access unit");
    };
    assert_eq!(built.to_bytes().unwrap(), fixture.config);
}

/// The in-band stream's `hev1` record has `array_completeness` 0 and a
/// prefix SEI array. The header fields still match the record vidr builds.
#[test]
fn from_annexb_removes_in_band_parameter_sets() {
    let Some(fixture) = load("hevc_inband") else {
        return;
    };
    let config = HevcConfig::parse(&fixture.config).unwrap();
    let mut filter = FromAnnexB::new(NalCodec::Hevc, LengthSize::Four);
    for (index, unit) in to_annexb(&config, &fixture.samples).iter().enumerate() {
        let kept: Vec<&[u8]> = nal_units(&fixture.samples[index])
            .into_iter()
            .filter(|unit| {
                HevcNalType::of(unit)
                    .and_then(HevcNalType::parameter_set)
                    .is_none()
            })
            .collect();
        assert_eq!(
            filter.filter(unit).unwrap(),
            sample_of(&kept),
            "sample {index}"
        );
    }
    let Some(NalConfig::Hevc(built)) = filter.config() else {
        panic!("no hvcC after the first access unit");
    };
    assert_eq!(built.to_bytes().unwrap()[..22], fixture.config[..22]);
    let source_units: Vec<&[u8]> = config
        .nal_units()
        .filter(|unit| {
            HevcNalType::of(unit)
                .and_then(HevcNalType::parameter_set)
                .is_some()
        })
        .collect();
    assert_eq!(built.nal_units().collect::<Vec<_>>(), source_units);
}

/// `aspect_ratio_idc` 14 from the table, then 255 with an explicit 7:5. The
/// ratio the SPS VUI carries equals what ffprobe reads from the raw stream.
#[test]
fn sps_sample_aspect_ratio_matches_ffprobe() {
    for (name, expected) in [("hevc_sar", (4, 3)), ("hevc_sar_ext", (7, 5))] {
        let Some(fixture) = load(name) else { return };
        let Some(probed) = fixture.sar else {
            panic!("{name}.sar is missing. Regenerate the fixtures.");
        };
        assert_eq!(probed, expected, "{name}: ffprobe ratio");
        let config = HevcConfig::parse(&fixture.config).unwrap();
        let sps = config
            .arrays
            .iter()
            .find(|array| array.nal_unit_type == HevcNalType::SPS)
            .unwrap();
        let signal = hevc_sps_vui(&sps.nal_units[0]).unwrap();
        assert_eq!(signal.sample_aspect_ratio, Some(probed), "{name}");
    }
}
