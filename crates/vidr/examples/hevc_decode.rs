//! Decodes an Annex-B HEVC file with no hash check and no output, for instruction profiles.
//!
//! `cargo run --release -p vidr --features cpu-hevc --example hevc_decode -- FILE [THREADS]`
//!
//! `THREADS` is the worker count, and defaults to the machine's CPUs. 1 runs on the calling
//! thread.

use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use vidr::bitstream::annexb::{self, StartCode};
use vidr::decode::Decoder;
use vidr::decode::hevc::{HevcDecoder, HevcOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("expected an Annex-B HEVC file. Run: hevc_decode FILE [THREADS]")?;
    let threads = std::env::args()
        .nth(2)
        .map(|text| text.parse::<usize>())
        .transpose()?;
    let stream = std::fs::read(&path)?;
    let options = HevcOptions {
        threads,
        ..HevcOptions::default()
    };
    let mut decoder = HevcDecoder::<CpuRuntime>::with_options(&CpuDevice::new(), &[], options)?;
    let mut frames = 0u64;
    let mut packet = Vec::new();
    for (pts, nal) in annexb::split(&stream)?.enumerate() {
        packet.clear();
        packet.extend_from_slice(StartCode::Four.bytes());
        packet.extend_from_slice(nal);
        decoder.send(&packet, pts as i64)?;
        while decoder.receive().is_some() {
            frames += 1;
        }
    }
    decoder.flush()?;
    while decoder.receive().is_some() {
        frames += 1;
    }
    println!("{frames} frames from {path}");
    Ok(())
}
