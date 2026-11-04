//! The worker path against the calling-thread path on hand-built streams: frames, pts,
//! error text, the cap on pictures in flight, and drop in every state.

use numr::runtime::cpu::{CpuDevice, CpuRuntime};

use super::super::super::cpu::hevc::stream_tests::{
    annexb_of, empty_slice_picture, half_of_two_ctb_picture, intra_pictures, second_ctb_slice,
    two_ctb_parameter_sets, two_ctb_picture,
};
use super::*;
use crate::decode::Decoder;
use crate::decode::hevc::HevcOptions;

type TestDecoder = HevcDecoder<CpuRuntime>;

/// A decoder on `workers` threads. One worker runs the pool form, which no public count asks
/// for.
fn pool_decoder(workers: usize) -> TestDecoder {
    let engine = ThreadedEngine::start(workers, DecodeOptions::default()).unwrap();
    TestDecoder::open(&CpuDevice::new(), &[], Engine::Threaded(Box::new(engine))).unwrap()
}

/// pts and bytes of each frame.
type Frames = Vec<(i64, Vec<u8>, Vec<u8>)>;

fn take_frames(decoder: &mut TestDecoder, frames: &mut Frames) {
    while let Some(frame) = decoder.receive() {
        frames.push((
            frame.pts(),
            frame.y().to_bytes().unwrap(),
            frame.uv().to_bytes().unwrap(),
        ));
    }
}

/// Sends one packet per unit, with the pts its index gives, then flushes. Returns the frames
/// and the first error.
fn run(decoder: &mut TestDecoder, units: &[Vec<u8>]) -> (Frames, Option<DecodeError>) {
    let mut frames = Vec::new();
    for (index, unit) in units.iter().enumerate() {
        let packet = annexb_of(std::slice::from_ref(unit));
        let sent = decoder.send(&packet, index as i64);
        take_frames(decoder, &mut frames);
        if let Err(error) = sent {
            return (frames, Some(error));
        }
    }
    let flushed = decoder.flush();
    take_frames(decoder, &mut frames);
    (frames, flushed.err())
}

fn serial_run(units: &[Vec<u8>]) -> (Frames, Option<DecodeError>) {
    let mut decoder = TestDecoder::with_threads(&CpuDevice::new(), &[], 1).unwrap();
    run(&mut decoder, units)
}

/// Every pool size gives the frames and the error of one thread.
fn assert_equal_to_serial(units: &[Vec<u8>]) -> Option<DecodeError> {
    let (frames, error) = serial_run(units);
    for workers in [1, 2, 8] {
        let (pool_frames, pool_error) = run(&mut pool_decoder(workers), units);
        assert_eq!(pool_frames, frames, "{workers} workers");
        assert_eq!(pool_error, error, "{workers} workers");
    }
    error
}

fn two_ctb_stream(tail: &[Vec<u8>]) -> Vec<Vec<u8>> {
    let mut units = two_ctb_parameter_sets();
    units.extend(tail.iter().cloned());
    units
}

#[test]
fn clean_pictures_equal_one_thread() {
    assert_eq!(assert_equal_to_serial(&intra_pictures(40)), None);
}

#[test]
fn pictures_of_two_slice_segments_equal_one_thread() {
    let first = half_of_two_ctb_picture();
    let units = two_ctb_stream(&[first.clone(), second_ctb_slice(), first, second_ctb_slice()]);
    assert_eq!(assert_equal_to_serial(&units), None);
}

/// A parse error of a worker: the same text, after the same frames.
#[test]
fn a_parse_error_equals_one_thread() {
    let picture = two_ctb_picture();
    let units = two_ctb_stream(&[
        picture.clone(),
        picture.clone(),
        empty_slice_picture(),
        picture.clone(),
        picture,
    ]);
    assert!(assert_equal_to_serial(&units).is_some());
}

/// A picture that ends incomplete, inside the stream.
#[test]
fn an_incomplete_picture_equals_one_thread() {
    let picture = two_ctb_picture();
    let units = two_ctb_stream(&[picture.clone(), half_of_two_ctb_picture(), picture]);
    assert!(assert_equal_to_serial(&units).is_some());
}

/// An error at the end of the stream, where waiting pictures still leave on one thread.
#[test]
fn an_incomplete_last_picture_equals_one_thread() {
    let picture = two_ctb_picture();
    let units = two_ctb_stream(&[picture.clone(), picture, half_of_two_ctb_picture()]);
    let error = assert_equal_to_serial(&units).unwrap();
    assert!(
        error.to_string().contains("the end of the stream"),
        "{error}"
    );
}

/// A repeated address ends the picture on the calling thread, with the parse's text.
#[test]
fn a_repeated_slice_address_equals_one_thread() {
    let first = half_of_two_ctb_picture();
    let units = two_ctb_stream(&[
        first,
        second_ctb_slice(),
        second_ctb_slice(),
        two_ctb_picture(),
    ]);
    assert!(assert_equal_to_serial(&units).is_some());
}

#[test]
fn the_first_error_stays_on_the_pool() {
    let units = intra_pictures(4);
    let mut decoder = pool_decoder(2);
    let error = decoder
        .send(&annexb_of(std::slice::from_ref(&empty_slice_picture())), 0)
        .unwrap_err();
    assert_eq!(decoder.send(&annexb_of(&units), 1), Err(error.clone()));
    assert_eq!(decoder.flush(), Err(error.clone()));
    assert_eq!(decoder.flush(), Err(error));
}

/// The send budget charges each output to the packet that outputs it, as on one thread.
#[test]
fn a_packet_past_its_luma_limit_is_a_named_error_on_the_pool() {
    let mut decoder = pool_decoder(4);
    decoder.send_luma_limit = 2 * 16 * 16;
    let stream = annexb_of(&intra_pictures(4));
    let error = decoder.send(&stream, 9).unwrap_err();
    let DecodeError::Software { call, input, .. } = &error else {
        panic!("a packet over its limit is a software error: {error}");
    };
    assert_eq!(*call, "HevcDecoder::send");
    assert_eq!(input, "the packet at pts 9");
    assert_eq!(decoder.queued(), 2);
}

/// Budget is per packet: a backlog of earlier packets costs the current one nothing.
#[test]
fn many_small_packets_stay_under_their_budget_on_the_pool() {
    let mut decoder = pool_decoder(2);
    decoder.send_luma_limit = 16 * 16;
    let (frames, error) = run(&mut decoder, &intra_pictures(32));
    assert_eq!(error, None);
    assert_eq!(frames.len(), 32);
}

/// A corrupt stream cannot grow the pictures in flight, on any pool size.
#[test]
fn pictures_in_flight_stay_at_their_cap() {
    for workers in [1, 2, 4] {
        let mut decoder = pool_decoder(workers);
        let (frames, error) = run(&mut decoder, &intra_pictures(64));
        assert_eq!(error, None);
        assert_eq!(frames.len(), 64);
        let Engine::Threaded(engine) = &decoder.engine else {
            panic!("the decoder runs on the pool");
        };
        // One unit can add one picture past a check of the cap.
        assert!(
            engine.peak_in_flight <= engine.cap() + 1,
            "{} pictures in flight on {workers} workers",
            engine.peak_in_flight
        );
    }
}

/// One packet of many pictures also holds to the cap, with no receive in between.
#[test]
fn one_packet_of_many_pictures_stays_at_the_cap() {
    let mut decoder = pool_decoder(2);
    decoder.send(&annexb_of(&intra_pictures(64)), 1).unwrap();
    let Engine::Threaded(engine) = &decoder.engine else {
        panic!("the decoder runs on the pool");
    };
    assert!(engine.peak_in_flight <= engine.cap() + 1);
    assert_eq!(decoder.flush(), Ok(()));
    assert_eq!(decoder.queued(), 64);
}

/// A worker's panic is a named error at its picture's place, and the frames before it stay.
#[test]
fn a_worker_panic_is_a_named_error() {
    let mut decoder = pool_decoder(2);
    let mut units = intra_pictures(3);
    let bad = units.len() - 1;
    let mut frames = Vec::new();
    for (index, unit) in units.drain(..).enumerate() {
        if index == bad {
            let Engine::Threaded(engine) = &mut decoder.engine else {
                panic!("the decoder runs on the pool");
            };
            engine.stream.stages_mut().0.panic_next = true;
        }
        decoder
            .send(&annexb_of(std::slice::from_ref(&unit)), index as i64)
            .unwrap();
        take_frames(&mut decoder, &mut frames);
    }
    let error = decoder.flush().unwrap_err();
    take_frames(&mut decoder, &mut frames);
    let DecodeError::Software { reason, .. } = &error else {
        panic!("a panic is a software error: {error}");
    };
    assert!(reason.contains("parse stage"), "{reason}");
    assert_eq!(frames.len(), 2);
    assert_eq!(decoder.flush(), Err(error));
}

#[test]
fn drop_before_any_send_joins() {
    drop(pool_decoder(4));
}

/// A slice that no boundary ends, then the pictures it leaves.
#[test]
fn drop_mid_picture_joins() {
    let mut decoder = pool_decoder(4);
    let units = intra_pictures(1);
    for unit in &units {
        decoder
            .send(&annexb_of(std::slice::from_ref(unit)), 0)
            .unwrap();
    }
    drop(decoder);
}

#[test]
fn drop_with_pictures_in_flight_joins() {
    let mut decoder = pool_decoder(1);
    decoder.send(&annexb_of(&intra_pictures(32)), 0).unwrap();
    drop(decoder);
}

#[test]
fn drop_after_an_error_joins() {
    let mut decoder = pool_decoder(2);
    let _ = decoder.send(&annexb_of(std::slice::from_ref(&empty_slice_picture())), 0);
    drop(decoder);
}

#[test]
fn drop_after_a_flush_joins() {
    let mut decoder = pool_decoder(2);
    decoder.send(&annexb_of(&intra_pictures(4)), 0).unwrap();
    decoder.flush().unwrap();
    drop(decoder);
}

/// Every public form reaches the same frames.
#[test]
fn each_constructor_decodes() {
    let device = CpuDevice::new();
    let stream = annexb_of(&intra_pictures(4));
    let opened = [
        TestDecoder::new(&device, &[]),
        TestDecoder::with_threads(&device, &[], 3),
        TestDecoder::with_picture_hash_check(&device, &[]),
        TestDecoder::with_options(
            &device,
            &[],
            HevcOptions {
                threads: Some(2),
                check_picture_hash: true,
            },
        ),
    ];
    for decoder in opened {
        let mut decoder = decoder.unwrap();
        decoder.send(&stream, 1).unwrap();
        decoder.flush().unwrap();
        assert_eq!(decoder.queued(), 4);
    }
}

#[test]
fn receive_finds_frames_after_a_send_returns() {
    let mut decoder = pool_decoder(2);
    decoder.send(&annexb_of(&intra_pictures(8)), 1).unwrap();
    let mut frames = Vec::new();
    decoder.flush().unwrap();
    take_frames(&mut decoder, &mut frames);
    assert_eq!(frames.len(), 8);
    assert!(decoder.receive().is_none());
}
