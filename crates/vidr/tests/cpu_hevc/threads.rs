//! Thread counts against each other on real streams: bytes, order, and pts, on clean
//! streams and on damaged ones.

use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use vidr::bitstream::{HevcConfig, ToAnnexB};
use vidr::decode::Decoder;
use vidr::decode::hevc::{HevcDecoder, HevcOptions};

use super::packets::nal_packets;
use crate::decode_fixtures::cpu::samples;
use crate::decode_fixtures::load;

/// The thread counts each test compares. The first runs on the calling thread.
const THREADS: [usize; 3] = [1, 2, 8];

/// pts and plane bytes of one frame.
type Frame = (i64, Vec<u8>, Vec<u8>);

/// The frames in output order, and the text of the first error.
#[derive(Debug, PartialEq, Eq)]
struct Decoded {
    frames: Vec<Frame>,
    error: Option<String>,
}

fn take(decoder: &mut HevcDecoder<CpuRuntime>, frames: &mut Vec<Frame>) {
    while let Some(frame) = decoder.receive() {
        frames.push((
            frame.pts(),
            frame.y().to_bytes().expect("the plane reads"),
            frame.uv().to_bytes().expect("the plane reads"),
        ));
    }
}

/// Sends every packet, and stops at the first error.
fn decode(
    config: &[u8],
    packets: &[(Vec<u8>, i64)],
    threads: usize,
    check_picture_hash: bool,
) -> Decoded {
    let options = HevcOptions {
        threads: Some(threads),
        check_picture_hash,
    };
    let mut decoder = HevcDecoder::<CpuRuntime>::with_options(&CpuDevice::new(), config, options)
        .expect("the decoder opens");
    let mut frames = Vec::new();
    for (packet, pts) in packets {
        let sent = decoder.send(packet, *pts);
        take(&mut decoder, &mut frames);
        if let Err(error) = sent {
            return Decoded {
                frames,
                error: Some(error.to_string()),
            };
        }
    }
    let flushed = decoder.flush();
    take(&mut decoder, &mut frames);
    Decoded {
        frames,
        error: flushed.err().map(|error| error.to_string()),
    }
}

fn assert_counts_agree(name: &str, config: &[u8], packets: &[(Vec<u8>, i64)]) {
    let reference = decode(config, packets, THREADS[0], false);
    assert_eq!(reference.error, None, "{name}: one thread");
    assert!(!reference.frames.is_empty(), "{name}: no frames");
    for threads in &THREADS[1..] {
        let decoded = decode(config, packets, *threads, false);
        assert_eq!(decoded.error, None, "{name}: {threads} threads");
        assert_eq!(
            decoded.frames.len(),
            reference.frames.len(),
            "{name}: {threads} threads: frames"
        );
        for (index, (got, expected)) in decoded.frames.iter().zip(&reference.frames).enumerate() {
            assert!(
                got == expected,
                "{name}: {threads} threads: frame {index} differs: pts {} against {}",
                got.0,
                expected.0
            );
        }
    }
}

/// The packets of `NAME.hevc`, or `None` after a notice.
fn raw_packets(name: &str) -> Option<Vec<(Vec<u8>, i64)>> {
    let dir = std::env::var_os("VIDR_HEVC_FIXTURES").map_or_else(
        || std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/hevc"),
        std::path::PathBuf::from,
    );
    let Ok(stream) = std::fs::read(dir.join(format!("{name}.hevc"))) else {
        eprintln!(
            "skipping: HEVC fixture `{name}` is missing in {}. Run \
             crates/vidr/tests/fixtures/gen_hevc.py, or set VIDR_HEVC_FIXTURES to its output \
             directory.",
            dir.display()
        );
        return None;
    };
    let packets = nal_packets(&stream).unwrap_or_else(|error| panic!("{name}: {error}"));
    Some(packets.collect())
}

/// Reordered, skipped, and dropped pictures, in decode order and in several sequences.
#[test]
fn raw_streams_agree_at_every_thread_count() {
    for name in [
        "bframes_8",
        "b_pyramid_10",
        "p_only_8",
        "rasl_cut_8",
        "end_of_sequence_8",
        "no_output_prior_8",
        "scene_cut_8",
    ] {
        if let Some(packets) = raw_packets(name) {
            assert_counts_agree(name, &[], &packets);
        }
    }
}

#[test]
fn container_samples_agree_at_every_thread_count() {
    for name in ["hevc_8bit", "hevc_10bit", "hevc_chroma_center"] {
        if let Some(stream) = load(name) {
            assert_counts_agree(name, &stream.config, &samples(&stream));
        }
    }
}

/// The filter passes over rows see every SAO and deblocking setting of these streams.
#[test]
fn annexb_samples_agree_at_every_thread_count() {
    let name = "hevc_8bit";
    let Some(stream) = load(name) else {
        return;
    };
    let config = HevcConfig::parse(&stream.config).expect("the config parses");
    let mut filter = ToAnnexB::hevc(&config);
    let packets: Vec<_> = samples(&stream)
        .into_iter()
        .map(|(sample, pts)| (filter.filter(&sample).expect("the sample filters"), pts))
        .collect();
    assert_counts_agree(name, &[], &packets);
}

/// The hash check runs in the worker, and passes at every count.
#[test]
fn the_hash_check_agrees_at_every_thread_count() {
    let Some(packets) = raw_packets("bframes_8") else {
        return;
    };
    let reference = decode(&[], &packets, 1, true);
    assert_eq!(reference.error, None);
    for threads in &THREADS[1..] {
        assert!(
            decode(&[], &packets, *threads, true) == reference,
            "{threads} threads"
        );
    }
}

/// On damaged data the first error and every frame before it stay the same. A failing
/// picture can feed no frame of the output, because its error comes first.
#[test]
fn damaged_streams_keep_their_first_error_and_frames() {
    let Some(packets) = raw_packets("bframes_8") else {
        return;
    };
    for seed in 0..6usize {
        let mut damaged = packets.clone();
        let index = (seed * 7 + 5) % damaged.len();
        let at = damaged[index].0.len() / 2 + seed;
        if let Some(byte) = damaged[index].0.get_mut(at) {
            *byte ^= 0x55;
        }
        let reference = decode(&[], &damaged, 1, true);
        for threads in &THREADS[1..] {
            let decoded = decode(&[], &damaged, *threads, true);
            assert_eq!(
                decoded.error, reference.error,
                "seed {seed}: {threads} threads"
            );
            assert!(
                decoded.frames == reference.frames,
                "seed {seed}: {threads} threads: {} frames against {}",
                decoded.frames.len(),
                reference.frames.len()
            );
        }
    }
}
