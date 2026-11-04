//! Decodes an Annex-B HEVC file with no hash check and no output, for instruction profiles.
//!
//! `cargo run --release -p vidr --features cpu-hevc --example hevc_decode -- FILE`

use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use vidr::bitstream::annexb::{self, StartCode};
use vidr::decode::Decoder;
use vidr::decode::hevc::HevcDecoder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("expected an Annex-B HEVC file. Run: hevc_decode FILE")?;
    let stream = std::fs::read(&path)?;
    let mut decoder = HevcDecoder::<CpuRuntime>::new(&CpuDevice::new(), &[])?;
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
