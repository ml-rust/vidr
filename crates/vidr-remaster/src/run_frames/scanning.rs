//! A pre-scan: one whole decode on the chosen decoder, with each frame dropped at
//! once. See [`ScanState`]. The reuse probe scans a candidate the same way.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use vidr::decode::{ColorFacts, DecodeCodec, DecodedFrame, Decoder as _};
use vidr::media::{ContainerColour, DecodeConfig, Input, InputSource, read_stream_facts};
use vidr::pipeline::{PacketSource as _, PumpControl};
use vidr::{DecodeError, Rotation, StreamFacts};

use crate::error::RemasterError;
use crate::execution::{
    DecoderBackend, ExecutionError, RuntimeKind, Selector, SourceCodec, SourceFacts,
};
use crate::frame_path::HostCopies;
use crate::request::{Request, Validated};
use crate::reuse::{Reuse, VideoFacts, VideoProbe, find_upscaled_for};
use crate::run::Progress;
use crate::scan::{
    ScanError, ScanFacts, ScanFrame, ScanRefusal, ScanState, candidate_facts, decoded_stream,
};

use super::backends::{BoxDecoder, decode_error};
use super::pass::stop_check;
use super::runtime::{Opened, RunRuntime};

/// Frames between `Scanned` events.
const SCAN_NOTICE_INTERVAL: u64 = 10_000;

/// The decoder choice's facts from a track's decoder config.
///
/// # Errors
///
/// [`ExecutionError::UnknownCodec`] for a codec this release does not name.
pub(super) fn source_facts(decode: &DecodeConfig) -> Result<SourceFacts, RemasterError> {
    let codec = match decode.codec {
        DecodeCodec::H264 => SourceCodec::H264,
        DecodeCodec::Hevc => SourceCodec::Hevc,
        DecodeCodec::Av1 => SourceCodec::Av1,
        DecodeCodec::Vp9 => SourceCodec::Vp9,
        other => {
            return Err(ExecutionError::UnknownCodec {
                name: other.to_string(),
            }
            .into());
        }
    };
    Ok(SourceFacts {
        codec,
        h264_profile: h264_profile(codec, decode),
    })
}

/// `profile_idc` of an H.264 track. Only OpenH264 reads it, and its features hold
/// the parser. A record that does not parse gets its error when the decoder opens.
#[cfg(any(feature = "openh264", feature = "openh264-cisco"))]
fn h264_profile(codec: SourceCodec, decode: &DecodeConfig) -> Option<u8> {
    if codec != SourceCodec::H264 {
        return None;
    }
    vidr::bitstream::AvcConfig::parse(&decode.config)
        .ok()
        .map(|config| config.profile_indication)
}

#[cfg(not(any(feature = "openh264", feature = "openh264-cisco")))]
fn h264_profile(_codec: SourceCodec, _decode: &DecodeConfig) -> Option<u8> {
    None
}

/// A video's packets and what its decoder takes.
pub(super) struct OpenedFrames {
    pub(super) source: InputSource,
    pub(super) decode: DecodeConfig,
    pub(super) container: Option<ContainerColour>,
}

pub(super) fn open_frames(path: &Path) -> Result<OpenedFrames, RemasterError> {
    let input = Input::open(path)?;
    let track = input.video_track()?.clone();
    let decode = input.decode_config(&track)?;
    let container = input.colour(&track);
    Ok(OpenedFrames {
        source: InputSource::new(input)?,
        decode,
        container,
    })
}

/// What a scan establishes, and the decoder that gave it.
pub(super) struct ChosenScan {
    pub(super) scan: ScanFacts,
    pub(super) backend: DecoderBackend,
    /// Why NVDEC gave way to a CPU decoder, when it did.
    pub(super) fell_through: Option<String>,
}

/// Whether the error states what the GPU cannot do. Other errors state what the
/// stream holds, and a CPU decoder gives nothing better for those.
const fn refused_by_hardware(error: &DecodeError) -> bool {
    matches!(
        error,
        DecodeError::Unsupported { .. } | DecodeError::CodedSize { .. }
    )
}

/// Runs each scan of one run, and counts their host copies.
pub(super) struct Scanner<'a, R: Runtime> {
    opened: &'a Opened<R>,
    control: &'a PumpControl,
    selector: &'a Selector,
    copies: HostCopies,
}

impl<'a, R: RunRuntime> Scanner<'a, R> {
    pub(super) const fn new(
        opened: &'a Opened<R>,
        control: &'a PumpControl,
        selector: &'a Selector,
    ) -> Self {
        Self {
            opened,
            control,
            selector,
            copies: HostCopies {
                frames: 0,
                bytes: 0,
            },
        }
    }

    /// The host copies of every scan so far.
    pub(super) const fn copies(&self) -> HostCopies {
        self.copies
    }

    /// Scans `path` on `first`. Under `--decoder auto`, a video NVDEC refuses goes to
    /// the CPU decoder of its codec.
    pub(super) fn scan<P: FnMut(Progress)>(
        &mut self,
        first: DecoderBackend,
        path: &Path,
        progress: &mut P,
    ) -> Result<ChosenScan, RemasterError> {
        match self.scan_file(first, path, progress) {
            Err(RemasterError::Decode { error, .. })
                if first == DecoderBackend::Nvdec
                    && self.selector.decoder_is_auto()
                    && refused_by_hardware(&error) =>
            {
                let facts = source_facts(&open_frames(path)?.decode)?;
                let (backend, reason) = self
                    .selector
                    .cpu_decoder_after(&facts, &error.to_string())?;
                Ok(ChosenScan {
                    scan: self.scan_file(backend, path, progress)?,
                    backend,
                    fell_through: Some(reason),
                })
            }
            other => other.map(|scan| ChosenScan {
                scan,
                backend: first,
                fell_through: None,
            }),
        }
    }

    fn scan_file<P: FnMut(Progress)>(
        &mut self,
        backend: DecoderBackend,
        path: &Path,
        progress: &mut P,
    ) -> Result<ScanFacts, RemasterError> {
        let opened = open_frames(path)?;
        let mut source = opened.source;
        let mut decoder = R::open_decoder(self.opened, backend, &opened.decode, path)?;
        let mut state = ScanState::new(opened.container);
        let mut data = Vec::new();
        let mut packets = 0_u64;
        let failed = |error| decode_error(path, error);
        while let Some(packet) = source.next(&mut data)? {
            stop_check(self.control)?;
            packets += 1;
            decoder.send(&data, packet.pts).map_err(failed)?;
            self.observe_ready(&mut decoder, &mut state, path, progress)?;
        }
        decoder.flush().map_err(failed)?;
        self.observe_ready(&mut decoder, &mut state, path, progress)?;
        state
            .finish(packets)
            .map_err(|refusal| scan_error(path, refusal))
    }

    fn observe_ready<P: FnMut(Progress)>(
        &mut self,
        decoder: &mut BoxDecoder<R>,
        state: &mut ScanState,
        path: &Path,
        progress: &mut P,
    ) -> Result<(), RemasterError> {
        let copies_through_host = decoder.route().copies_through_host();
        while let Some(frame) = decoder.receive() {
            if copies_through_host {
                self.copies.frames += 1;
                self.copies.bytes += plane_bytes(frame.y()) + plane_bytes(frame.uv());
            }
            let frames = state
                .observe(scan_frame(&frame))
                .map_err(|refusal| scan_error(path, refusal))?;
            if frames.is_multiple_of(SCAN_NOTICE_INTERVAL) {
                progress(Progress::Scanned { frames });
            }
        }
        Ok(())
    }
}

fn scan_error(path: &Path, refusal: ScanRefusal) -> RemasterError {
    RemasterError::Scan(ScanError {
        path: path.to_path_buf(),
        refusal,
    })
}

fn plane_bytes<R: Runtime<DType = DType>>(plane: &Tensor<R>) -> u64 {
    plane.numel() as u64 * plane.dtype().size_in_bytes() as u64
}

fn scan_frame<R: Runtime>(frame: &DecodedFrame<R>) -> ScanFrame {
    let color: ColorFacts = *frame.color();
    ScanFrame {
        pts: frame.pts(),
        display: frame.display(),
        format: frame.format(),
        color,
        progressive: frame.progressive(),
    }
}

/// The video that supplies the frames: the source, or an enhanced version of it.
pub(super) struct FrameSource {
    pub(super) path: PathBuf,
    pub(super) stream: StreamFacts,
    pub(super) scan: ScanFacts,
    /// The decoder that reads it.
    pub(super) decoder: DecoderBackend,
    /// Why this decoder reads a reused video.
    pub(super) decoder_reason: String,
}

/// [`VideoProbe`] over `read_stream_facts` and a full scan of the candidate.
///
/// The pump decodes the selected candidate, so it must pass what the source
/// passes. The probe keeps each result, so the selected one needs no second scan.
struct ScanProbe<'s, 'a, R: Runtime, P> {
    scanner: &'s mut Scanner<'a, R>,
    runtime: RuntimeKind,
    progress: &'s mut P,
    rotation: Rotation,
    scanned: HashMap<PathBuf, FrameSource>,
}

impl<R: RunRuntime, P: FnMut(Progress)> ScanProbe<'_, '_, R, P> {
    fn read(&mut self, path: &Path) -> Result<FrameSource, RemasterError> {
        let input = Input::open(path)?;
        let stated = read_stream_facts(&input)?;
        let track = input.video_track()?.clone();
        let facts = source_facts(&input.decode_config(&track)?)?;
        drop(input);
        let (first, reason) = self.scanner.selector.decoder_for(self.runtime, &facts)?;
        let chosen = self.scanner.scan(first, path, self.progress)?;
        Ok(FrameSource {
            path: path.to_path_buf(),
            stream: decoded_stream(stated, &chosen.scan),
            scan: chosen.scan,
            decoder: chosen.backend,
            decoder_reason: chosen.fell_through.unwrap_or(reason),
        })
    }
}

impl<R: RunRuntime, P: FnMut(Progress)> VideoProbe for ScanProbe<'_, '_, R, P> {
    type Error = String;

    fn probe(&mut self, path: &Path) -> Result<VideoFacts, String> {
        let frames = self.read(path).map_err(|error| error.to_string())?;
        let facts = candidate_facts(&frames.stream, self.rotation)?;
        self.scanned.insert(path.to_path_buf(), frames);
        Ok(facts)
    }
}

/// What [`select_existing`] compares with a run.
pub(super) struct ReuseQuery<'q> {
    pub(super) request: &'q Request,
    pub(super) validated: &'q Validated,
    pub(super) source_facts: &'q VideoFacts,
    pub(super) rotation: Rotation,
    pub(super) runtime: RuntimeKind,
    /// The name of this run's encoder. A recorded candidate must name it too.
    pub(super) encoder: &'q str,
}

/// Selects an enhanced video, unless the request runs the model regardless.
pub(super) fn select_existing<R: RunRuntime, P: FnMut(Progress)>(
    query: &ReuseQuery<'_>,
    scanner: &mut Scanner<'_, R>,
    control: &PumpControl,
    progress: &mut P,
) -> Result<Option<FrameSource>, RemasterError> {
    let request = query.request;
    if request.force_ai {
        return Ok(None);
    }
    let mut probe = ScanProbe {
        scanner,
        runtime: query.runtime,
        progress: &mut *progress,
        rotation: query.rotation,
        scanned: HashMap::new(),
    };
    let found = find_upscaled_for(
        &mut probe,
        &query.validated.source,
        request.model,
        query.source_facts,
        &query.validated.destination,
        request.from_upscaled.as_deref(),
        query.encoder,
    );
    let mut scanned = probe.scanned;
    // A stop makes a probe fail, and that failure is no statement on the candidate.
    stop_check(control)?;
    let Reuse { selected, skipped } = found?;
    for candidate in skipped {
        progress(Progress::Skipped(candidate));
    }
    Ok(selected.and_then(|path| scanned.remove(&path)))
}
