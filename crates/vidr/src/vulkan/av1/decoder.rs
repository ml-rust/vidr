//! One AV1 stream on the Vulkan core: sequences, sessions, frames, and frames shown again.

use std::collections::VecDeque;
use std::ptr;
use std::sync::Arc;

use ash::vk;
use numr::runtime::wgpu::WgpuRuntime;

use super::refs::{FrameRefs, NamedFrame, RefPool, SLOTS, lost_reference};
use super::std_params::StdSequence;
use super::std_picture::{
    slot_facts, std_cdef, std_film_grain, std_global_motion, std_loop_filter, std_loop_restoration,
    std_picture_info, std_quantization, std_reference_info, std_segmentation, std_tile_info,
};
use crate::bitstream::av1::frame_header::{
    ALL_FRAMES, FrameHeader, KEY_FRAME, NUM_REF_FRAMES, REFS_PER_FRAME,
};
use crate::bitstream::av1::seq_header::{SequenceHeader, bit_depth};
use crate::bitstream::av1::temporal_unit::{Av1Parser, ParsedFrame, looks_like_annex_b};
use crate::bitstream::av1::{Av1Config, Av1SequenceFacts};
use crate::decode::av1_frame::{check_facts, check_layout, color_facts};
use crate::decode::{ColorFacts, DecodeCodec, DecodedFrame, PixelFormat};
use crate::error::DecodeError;
use crate::vulkan::caps::{DecodeCaps, DpbPreference, check_stream, dpb_mode};
use crate::vulkan::copy::whole_frame;
use crate::vulkan::device::VulkanDevice;
use crate::vulkan::engine::{CopySource, DecodeEngine, DecodeRequest, EngineSpec, SlotRef};
use crate::vulkan::error::request_error;
use crate::vulkan::session::{SessionParameters, VideoProfile, query_av1_caps};

/// `VK_MAX_VIDEO_AV1_REFERENCES_PER_FRAME_KHR`.
const ACTIVE_REFERENCES: u32 = REFS_PER_FRAME as u32;
const CODEC: DecodeCodec = DecodeCodec::Av1;

/// The session in use.
struct Active {
    /// What the driver lists for this profile.
    caps: DecodeCaps,
    sequence: Arc<SequenceHeader>,
    format: PixelFormat,
    color: ColorFacts,
    engine: DecodeEngine,
    /// The sequence header and what its pointers name. NVIDIA's driver reads the colour config
    /// after `vkCreateVideoSessionParameters` returns, and it turns off chroma filters on a stale
    /// value. This drops after `engine`, so it outlives the session parameters.
    _std_sequence: StdSequence,
}

/// AV1 on Vulkan Video. See [`crate::vulkan::VulkanDecoder`].
pub(in crate::vulkan) struct Av1Decoder {
    device: Arc<VulkanDevice>,
    preference: DpbPreference,
    parser: Av1Parser,
    pool: RefPool,
    /// Each name, as a bit, that `ref_order_hint` states as lost until a frame refreshes it.
    lost_names: u8,
    active: Option<Active>,
    /// The grained planes of each slot's frame, for a frame shown again. A slot holds samples
    /// with no grain, so a copy of it cannot repeat them.
    grained: Vec<Option<DecodedFrame<WgpuRuntime>>>,
    frames: VecDeque<DecodedFrame<WgpuRuntime>>,
}

/// The refusal of a stream in Annex B framing.
fn annex_b_refusal(packet_bytes: usize) -> DecodeError {
    DecodeError::VulkanRefused {
        codec: CODEC,
        what: format!("a {packet_bytes}-byte packet in Annex B framing"),
        reason: "vidr reads the low-overhead format. Run `ffmpeg -i INPUT -c copy -f obu OUTPUT` \
                 to convert the stream"
            .to_owned(),
    }
}

/// The refusal of a frame that reads a lost name.
fn lost_refusal(name: u8, pts: i64) -> DecodeError {
    DecodeError::VulkanRefused {
        codec: CODEC,
        what: format!("reference {name} at pts {pts}, which ref_order_hint states as lost"),
        reason: "Vulkan Video activates a DPB slot only through a decode, so no slot can stand in \
                 for it. Decode this stream with another decoder, or start at a key frame"
            .to_owned(),
    }
}

/// Refuses a sequence vidr has no path for.
fn check_sequence(sequence: &SequenceHeader) -> Result<(), DecodeError> {
    check_facts(&Av1SequenceFacts::from(sequence))?;
    if sequence.seq_profile != 0 {
        return Err(DecodeError::VulkanRefused {
            codec: CODEC,
            what: format!("seq_profile {}", sequence.seq_profile),
            reason: "this path decodes Main only. Use another decoder".to_owned(),
        });
    }
    Ok(())
}

/// The session size: what the sequence header states as its largest frame.
fn session_size(sequence: &SequenceHeader) -> (u32, u32) {
    (
        u32::from(sequence.max_frame_width_minus_1) + 1,
        u32::from(sequence.max_frame_height_minus_1) + 1,
    )
}

impl Av1Decoder {
    /// A decoder with no session yet. `config` is an `av1C` payload, or empty.
    pub(in crate::vulkan) fn new(
        device: Arc<VulkanDevice>,
        config: &[u8],
        preference: DpbPreference,
    ) -> Result<Self, DecodeError> {
        let mut parser = Av1Parser::default();
        if !config.is_empty() {
            let parsed = Av1Config::parse(config)?;
            check_layout(
                bit_depth(parsed.high_bitdepth, parsed.twelve_bit),
                parsed.monochrome,
                parsed.chroma_subsampling_x,
                parsed.chroma_subsampling_y,
            )?;
            if let Some(header) = parsed.sequence_header()? {
                parser.set_sequence(header.payload)?;
            }
        }
        Ok(Self {
            device,
            preference,
            parser,
            pool: RefPool::default(),
            lost_names: 0,
            active: None,
            grained: vec![None; SLOTS as usize],
            frames: VecDeque::new(),
        })
    }

    /// Decodes each coded frame of a packet, and queues the last shown frame of each temporal
    /// unit: the highest spatial layer of the operating point.
    pub(in crate::vulkan) fn send(&mut self, packet: &[u8], pts: i64) -> Result<(), DecodeError> {
        if looks_like_annex_b(packet) {
            return Err(annex_b_refusal(packet.len()));
        }
        let parsed = self.parser.parse_packet(packet)?;
        let mut shown = None;
        let mut unit = 0;
        for frame in &parsed {
            if frame.unit != unit {
                self.frames.extend(shown.take());
                unit = frame.unit;
            }
            let output = if frame.header.show_existing_frame {
                Some(self.show_existing(&frame.header, pts)?)
            } else {
                self.decode_frame(frame, pts)?
            };
            if output.is_some() {
                shown = output;
            }
        }
        self.frames.extend(shown);
        Ok(())
    }

    /// The next frame in decode order. AV1 output has no reorder.
    pub(in crate::vulkan) fn receive(&mut self) -> Option<DecodedFrame<WgpuRuntime>> {
        self.frames.pop_front()
    }

    /// Repeats a stored frame: its grained planes, or a new copy of its slot. The slot stays a
    /// reference. A key frame also refreshes every name: section 7.21.
    fn show_existing(
        &mut self,
        header: &FrameHeader,
        pts: i64,
    ) -> Result<DecodedFrame<WgpuRuntime>, DecodeError> {
        let name = header.frame_to_show_map_idx;
        let (Some(named), Some(active)) = (self.pool.get(name), &mut self.active) else {
            return Err(request_error(
                pts,
                format!("show_existing_frame names reference {name}, which holds no frame"),
            ));
        };
        let stored = self
            .grained
            .get(named.slot as usize)
            .and_then(Option::as_ref);
        let frame = match stored {
            Some(frame) => frame.clone().with_pts(pts),
            None => {
                let source = active.engine.read_slot(named.slot, pts)?;
                whole_frame(
                    &self.device,
                    &source,
                    active.format,
                    named.coded,
                    named.color,
                    pts,
                )?
            }
        };
        if header.refresh_frame_flags == ALL_FRAMES {
            self.refresh_names(ALL_FRAMES, named);
        }
        Ok(frame)
    }

    /// Stores `named` under the names `flags` sets, and frees each slot no name holds after it.
    fn refresh_names(&mut self, flags: u8, named: NamedFrame) {
        self.lost_names &= !flags;
        for freed in self.pool.refresh(flags, named) {
            self.free_slot(freed);
        }
    }

    fn free_slot(&mut self, slot: u32) {
        if let Some(active) = &mut self.active {
            active.engine.release_slot(slot);
        }
        if let Some(entry) = self.grained.get_mut(slot as usize) {
            *entry = None;
        }
    }

    /// Empties each name `lost` sets. The parser leaves the name named, and its old slot holds a
    /// frame the stream no longer means, so no frame decodes against it.
    fn drop_lost_names(&mut self, lost: u8) {
        self.lost_names |= lost;
        for name in (0..NUM_REF_FRAMES as u8).filter(|name| (lost >> name) & 1 == 1) {
            if let Some(freed) = self.pool.clear(name) {
                self.free_slot(freed);
            }
        }
    }

    /// Opens a session for `sequence`, with its parameters.
    fn open_session(
        &mut self,
        sequence: &Arc<SequenceHeader>,
        format: PixelFormat,
    ) -> Result<(), DecodeError> {
        // The old session goes first, so its memory is free for the new one.
        self.active = None;
        self.pool = RefPool::default();
        self.lost_names = 0;
        self.grained.fill(None);
        let grain = sequence.film_grain_params_present;
        let profile = VideoProfile::av1(grain, format);
        let caps = query_av1_caps(&self.device, &profile)?;
        let size = session_size(sequence);
        check_stream(&caps, CODEC, size, SLOTS, ACTIVE_REFERENCES)?;
        let mode = dpb_mode(CODEC, caps.decode_flags, self.preference)?;
        let mut engine = DecodeEngine::new(
            &self.device,
            profile,
            &EngineSpec {
                caps: &caps,
                mode,
                coded: size,
                slots: SLOTS,
                active_references: ACTIVE_REFERENCES,
                readable_slots: true,
                output_image: grain,
            },
        )?;
        let std_sequence = StdSequence::new(sequence);
        let create = vk::VideoDecodeAV1SessionParametersCreateInfoKHR {
            p_std_sequence_header: std_sequence.as_ptr(),
            ..Default::default()
        };
        // SAFETY: `create` is the AV1 structure of this AV1 session. It and all it points to live
        // through the call. `std_sequence` then moves into `Active`, which keeps each heap address
        // and outlives the parameters.
        let parameters = unsafe {
            SessionParameters::new(
                &self.device,
                engine.session(),
                ptr::from_ref(&create).cast(),
            )
        }?;
        engine.set_parameters(parameters);
        self.active = Some(Active {
            caps,
            sequence: Arc::clone(sequence),
            format,
            color: color_facts(&Av1SequenceFacts::from(&**sequence)),
            engine,
            _std_sequence: std_sequence,
        });
        Ok(())
    }

    /// Puts a session in place that takes this frame. Section 7.5 starts a new coded video
    /// sequence at a key frame. So a new sequence header opens a new session there. The first
    /// frame of a stream can also be an intra-only frame, as libaom takes one.
    fn prepare_session(&mut self, frame: &ParsedFrame<'_>, pts: i64) -> Result<(), DecodeError> {
        let header = &frame.header;
        let current = self
            .active
            .as_ref()
            .filter(|active| Arc::ptr_eq(&active.sequence, &frame.sequence));
        if current.is_none() {
            let starts = if self.active.is_some() {
                header.frame_type == KEY_FRAME
            } else {
                header.frame_is_intra
            };
            if !starts {
                return Err(request_error(
                    pts,
                    "the frame reads references of an earlier sequence header or of no frame. \
                     Start the stream at a key frame"
                        .to_owned(),
                ));
            }
            check_sequence(&frame.sequence)?;
            let depth = frame.sequence.color.bit_depth;
            let Some(format) = PixelFormat::for_bit_depth(depth) else {
                return Err(request_error(
                    pts,
                    format!("BitDepth {depth} has no format"),
                ));
            };
            self.open_session(&frame.sequence, format)?;
        }
        let Some(active) = &self.active else {
            return Err(request_error(pts, "no session is open".to_owned()));
        };
        let coded = header.upscaled_size();
        check_stream(&active.caps, CODEC, coded, SLOTS, ACTIVE_REFERENCES)?;
        let session = active.engine.max_coded();
        if coded.0 > session.0 || coded.1 > session.1 {
            return Err(DecodeError::VulkanRefused {
                codec: CODEC,
                what: format!("a frame of {}x{}", coded.0, coded.1),
                reason: format!(
                    "the sequence header states at most {}x{}. The stream is damaged",
                    session.0, session.1
                ),
            });
        }
        Ok(())
    }

    /// The slots `header` reads, and the free slot it takes.
    fn frame_slots(&self, header: &FrameHeader, pts: i64) -> Result<(FrameRefs, u32), DecodeError> {
        if let Some(name) = lost_reference(header, self.lost_names) {
            return Err(lost_refusal(name, pts));
        }
        let refs = self.pool.frame_refs(header).map_err(|empty| {
            request_error(
                pts,
                format!(
                    "the frame reads reference {}, which holds no frame. Start the stream at a \
                     key frame",
                    empty.name
                ),
            )
        })?;
        self.pool.check_scaling(header).map_err(|refused| {
            let (reference, size) = (refused.reference, refused.frame);
            DecodeError::VulkanRefused {
                codec: CODEC,
                what: format!(
                    "a frame of {}x{} on reference {}",
                    size.0, size.1, refused.name
                ),
                reason: format!(
                    "the reference holds {}x{}, and AV1 allows at most 2 times the frame and at \
                     least a sixteenth of it. The stream is damaged",
                    reference.0, reference.1
                ),
            }
        })?;
        let Some(slot) = self.pool.free_slot() else {
            return Err(request_error(
                pts,
                "every DPB slot holds a frame".to_owned(),
            ));
        };
        Ok((refs, slot))
    }

    /// Decodes one frame into a slot. Returns the frame when it shows.
    fn decode_frame(
        &mut self,
        frame: &ParsedFrame<'_>,
        pts: i64,
    ) -> Result<Option<DecodedFrame<WgpuRuntime>>, DecodeError> {
        self.prepare_session(frame, pts)?;
        let header = &frame.header;
        self.drop_lost_names(header.lost_names);
        let (refs, slot) = self.frame_slots(header, pts)?;
        let Some(active) = &mut self.active else {
            return Err(request_error(pts, "no session is open".to_owned()));
        };
        let reference = header.refresh_frame_flags != 0;
        let grain = header.film_grain.apply_grain;
        let keep_grained = grain && header.showable_frame && reference;
        let output = header.show_frame || keep_grained;
        let source = submit_frame(active, frame, &refs, slot, output, pts)?;
        let coded = header.upscaled_size();
        let color = active.color;
        let decoded = if output {
            Some(whole_frame(
                &self.device,
                &source,
                active.format,
                coded,
                color,
                pts,
            )?)
        } else {
            None
        };
        if let Some(entry) = self.grained.get_mut(slot as usize) {
            *entry = decoded.as_ref().filter(|_| keep_grained).cloned();
        }
        let named = NamedFrame {
            slot,
            coded,
            color,
            facts: slot_facts(header),
        };
        self.refresh_names(header.refresh_frame_flags, named);
        Ok(decoded.filter(|_| header.show_frame))
    }
}

/// Runs one frame's decode. Every structure the driver reads lives in this function, through the
/// decode call.
fn submit_frame(
    active: &mut Active,
    frame: &ParsedFrame<'_>,
    refs: &FrameRefs,
    slot: u32,
    output: bool,
    pts: i64,
) -> Result<CopySource, DecodeError> {
    let header = &frame.header;
    let tile_info = std_tile_info(&header.tile_info);
    let quantization = std_quantization(&header.quantization);
    let segmentation = std_segmentation(&header.segmentation);
    let loop_filter = std_loop_filter(&header.loop_filter);
    let cdef = std_cdef(&header.cdef);
    let loop_restoration = std_loop_restoration(&header.loop_restoration);
    let global_motion = std_global_motion(header);
    let film_grain = std_film_grain(&header.film_grain);
    let mut std_picture = std_picture_info(header);
    std_picture.pTileInfo = ptr::from_ref(&tile_info);
    std_picture.pQuantization = ptr::from_ref(&quantization);
    std_picture.pSegmentation = ptr::from_ref(&segmentation);
    std_picture.pLoopFilter = ptr::from_ref(&loop_filter);
    std_picture.pCDEF = ptr::from_ref(&cdef);
    std_picture.pLoopRestoration = ptr::from_ref(&loop_restoration);
    std_picture.pGlobalMotion = ptr::from_ref(&global_motion);
    if frame.sequence.film_grain_params_present {
        std_picture.pFilmGrain = ptr::from_ref(&film_grain);
    }
    let offsets: Vec<u32> = frame.tiles.iter().map(|tile| tile.offset).collect();
    let sizes: Vec<u32> = frame.tiles.iter().map(|tile| tile.size).collect();
    let byte_ranges: Vec<(u32, u32)> = frame
        .tiles
        .iter()
        .map(|tile| (tile.offset, tile.size))
        .collect();
    let picture_info = vk::VideoDecodeAV1PictureInfoKHR {
        p_std_picture_info: ptr::from_ref(&std_picture),
        reference_name_slot_indices: refs.name_slots,
        frame_header_offset: 0,
        tile_count: offsets.len() as u32,
        p_tile_offsets: offsets.as_ptr(),
        p_tile_sizes: sizes.as_ptr(),
        ..Default::default()
    };
    // Each list stays whole before the next one points into it.
    let named = || refs.distinct[..refs.count].iter().flatten();
    let std_references: Vec<_> = named()
        .map(|entry| std_reference_info(&entry.facts))
        .collect();
    let dpb_infos: Vec<_> = std_references
        .iter()
        .map(|info| vk::VideoDecodeAV1DpbSlotInfoKHR {
            p_std_reference_info: ptr::from_ref(info),
            ..Default::default()
        })
        .collect();
    let references: Vec<SlotRef> = named()
        .zip(&dpb_infos)
        .map(|(entry, info)| SlotRef {
            slot: entry.slot,
            codec_info: ptr::from_ref(info).cast(),
            coded: entry.coded,
        })
        .collect();
    let setup_std = std_reference_info(&slot_facts(header));
    let setup_dpb = vk::VideoDecodeAV1DpbSlotInfoKHR {
        p_std_reference_info: ptr::from_ref(&setup_std),
        ..Default::default()
    };
    let request = DecodeRequest {
        bitstream: &frame.parts,
        picture_info: ptr::from_ref(&picture_info).cast(),
        setup: SlotRef {
            slot,
            codec_info: ptr::from_ref(&setup_dpb).cast(),
            coded: header.upscaled_size(),
        },
        references: &references,
        reference: header.refresh_frame_flags != 0,
        output,
        separate_output: header.film_grain.apply_grain,
        byte_ranges: &byte_ranges,
        pts,
    };
    // SAFETY: each pointer in `request` names an AV1 structure above, for this AV1 session, and
    // each one lives through the call. `frame` holds the tile arrays.
    unsafe { active.engine.decode(&request) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn annex_b_is_refused_by_name_with_the_next_action() {
        let text = annex_b_refusal(40).to_string();
        assert!(text.contains("40-byte"), "{text}");
        assert!(text.contains("Annex B"), "{text}");
        assert!(text.contains("-f obu"), "{text}");
    }

    #[test]
    fn a_lost_name_is_refused_by_name_with_the_next_action() {
        let error = lost_refusal(3, 90);
        assert!(matches!(error, DecodeError::VulkanRefused { .. }));
        let text = error.to_string();
        assert!(text.contains("reference 3"), "{text}");
        assert!(text.contains("pts 90"), "{text}");
        assert!(text.contains("another decoder"), "{text}");
    }

    #[test]
    fn a_session_takes_the_size_the_sequence_header_states() {
        let sequence = SequenceHeader {
            max_frame_width_minus_1: 319,
            max_frame_height_minus_1: 179,
            ..SequenceHeader::default()
        };
        assert_eq!(session_size(&sequence), (320, 180));
    }

    #[test]
    fn a_layout_with_no_frame_or_another_profile_is_refused_by_name() {
        let mono = SequenceHeader {
            color: crate::bitstream::av1::seq_header::ColorConfig {
                bit_depth: 8,
                mono_chrome: true,
                subsampling_x: true,
                subsampling_y: true,
                ..Default::default()
            },
            ..SequenceHeader::default()
        };
        assert!(matches!(
            check_sequence(&mono),
            Err(DecodeError::UnsupportedLayout { .. })
        ));
        let twelve = SequenceHeader {
            seq_profile: 2,
            color: crate::bitstream::av1::seq_header::ColorConfig {
                bit_depth: 12,
                subsampling_x: true,
                subsampling_y: true,
                ..Default::default()
            },
            ..SequenceHeader::default()
        };
        assert!(matches!(
            check_sequence(&twelve),
            Err(DecodeError::UnsupportedLayout { .. })
        ));
        let plain = SequenceHeader {
            color: crate::bitstream::av1::seq_header::ColorConfig {
                bit_depth: 10,
                subsampling_x: true,
                subsampling_y: true,
                ..Default::default()
            },
            ..SequenceHeader::default()
        };
        assert_eq!(check_sequence(&plain), Ok(()));
    }
}
