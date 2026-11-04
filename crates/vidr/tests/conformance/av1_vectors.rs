//! libaom's AV1 test vectors on disk, and the hash check, for each decoder's sweep. containr reads no IVF, and a test
//! holds its own small reader, as a container reader belongs in containr. A `.mkv` vector goes
//! through containr, which needs the `mkv` feature.

use std::path::Path;

use vidr::geometry::Dimensions;

/// `DKIF`.
const IVF_SIGNATURE: &[u8] = b"DKIF";
/// The bytes before each frame: its size, then its pts.
const IVF_FRAME_HEADER: usize = 12;

/// What one vector did.
pub(super) enum Outcome {
    Pass { frames: usize },
    Refused { reason: String },
    Fail { reason: String },
}

/// Compares frames against the hashes in order.
pub(super) struct HashCheck {
    expected: Vec<String>,
    next: usize,
}

impl HashCheck {
    pub(super) const fn new(expected: Vec<String>) -> Self {
        Self { expected, next: 0 }
    }

    /// Takes the hash of the next frame, which `display` sizes.
    pub(super) fn compare(&mut self, hash: &str, display: Dimensions) -> Result<(), String> {
        let Some(expected) = self.expected.get(self.next) else {
            return Err(format!(
                "the decoder gives more frames than the {} hashes",
                self.expected.len()
            ));
        };
        if hash != expected {
            return Err(format!(
                "frame {} ({}x{}): md5 {hash}, libaom has {expected}",
                self.next,
                display.width(),
                display.height()
            ));
        }
        self.next += 1;
        Ok(())
    }

    /// The outcome once the stream ends: a pass when each hash found its frame.
    pub(super) fn finish(&self) -> Result<Outcome, String> {
        if self.next != self.expected.len() {
            return Err(format!(
                "the decoder gives {} frames, and libaom has {}",
                self.next,
                self.expected.len()
            ));
        }
        Ok(Outcome::Pass { frames: self.next })
    }
}

/// Whether the vector's name states a layout every decoder here refuses.
pub(super) fn refusal_expected(name: &str) -> bool {
    name.contains("monochrome")
}

/// The frames of an IVF file, each with its pts.
fn ivf_packets(data: &[u8], name: &str) -> Result<Vec<(Vec<u8>, i64)>, String> {
    if data.get(..4) != Some(IVF_SIGNATURE) {
        return Err(format!(
            "{name} does not start with DKIF. Fetch the vector again."
        ));
    }
    let header = data
        .get(6..8)
        .map(|bytes| usize::from(u16::from_le_bytes([bytes[0], bytes[1]])))
        .ok_or_else(|| format!("{name} ends inside its IVF header. Fetch the vector again."))?;
    if data.len() < header {
        return Err(format!("{name} ends inside its IVF header"));
    }
    let mut packets = Vec::new();
    let mut position = header;
    while position < data.len() {
        let frame = data
            .get(position..position + IVF_FRAME_HEADER)
            .ok_or_else(|| format!("{name} ends inside a frame header at byte {position}"))?;
        let size = u32::from_le_bytes([frame[0], frame[1], frame[2], frame[3]]) as usize;
        let mut pts = [0; 8];
        pts.copy_from_slice(&frame[4..12]);
        let start = position + IVF_FRAME_HEADER;
        let payload = data
            .get(start..start + size)
            .ok_or_else(|| format!("{name} ends inside the frame at byte {start}"))?;
        packets.push((payload.to_vec(), i64::from_le_bytes(pts)));
        position = start + size;
    }
    Ok(packets)
}

/// Every packet of a Matroska vector, with its pts.
#[cfg(feature = "mkv")]
fn mkv_packets(path: &Path) -> Result<Vec<(Vec<u8>, i64)>, String> {
    use containr::TrackKind;
    use containr::mkv::Demuxer;

    let mut demuxer = Demuxer::open_path(path).map_err(|error| format!("demux: {error}"))?;
    let track = demuxer
        .tracks()
        .iter()
        .find(|track| track.kind == TrackKind::Video)
        .map(|track| track.id)
        .ok_or_else(|| "the file has no video track".to_owned())?;
    let packets = demuxer
        .track_packets(track)
        .map_err(|error| format!("demux: {error}"))?;
    let mut read = Vec::new();
    for packet in packets {
        let packet = packet.map_err(|error| format!("demux: {error}"))?;
        let mut data = Vec::new();
        demuxer
            .read_packet_into(&packet, &mut data)
            .map_err(|error| format!("demux at packet {}: {error}", packet.index))?;
        read.push((data, packet.pts));
    }
    Ok(read)
}

#[cfg(not(feature = "mkv"))]
fn mkv_packets(path: &Path) -> Result<Vec<(Vec<u8>, i64)>, String> {
    Err(format!(
        "{} is Matroska, and this build has no `mkv` feature. Build with --features mkv.",
        path.display()
    ))
}

/// Every packet of a vector, with its pts.
pub(super) fn read_packets(path: &Path) -> Result<Vec<(Vec<u8>, i64)>, String> {
    if path.extension().is_some_and(|extension| extension == "mkv") {
        return mkv_packets(path);
    }
    let name = path.display().to_string();
    let data = std::fs::read(path).map_err(|error| format!("reading {name}: {error}"))?;
    ivf_packets(&data, &name)
}

#[test]
fn an_ivf_file_gives_each_frame_with_its_pts() {
    let mut file = b"DKIF".to_vec();
    file.extend([0, 0, 32, 0]);
    file.resize(32, 0);
    for (payload, pts) in [(&[1_u8, 2, 3][..], 5_u64), (&[4][..], 9)] {
        file.extend((payload.len() as u32).to_le_bytes());
        file.extend(pts.to_le_bytes());
        file.extend(payload);
    }
    let packets = ivf_packets(&file, "test").expect("parses");
    assert_eq!(packets, [(vec![1, 2, 3], 5), (vec![4], 9)]);
}

#[test]
fn a_short_ivf_file_is_an_error_at_every_byte() {
    let mut file = b"DKIF".to_vec();
    file.extend([0, 0, 32, 0]);
    file.resize(32, 0);
    file.extend(3_u32.to_le_bytes());
    file.extend(0_u64.to_le_bytes());
    file.extend([1, 2, 3]);
    for end in 0..file.len() {
        if end == 32 {
            continue;
        }
        assert!(ivf_packets(&file[..end], "test").is_err(), "end {end}");
    }
}
