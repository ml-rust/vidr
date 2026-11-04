//! One VP9 stream on the Vulkan core: superframes, sessions, frames, and frames shown again.

use std::collections::VecDeque;
use std::ptr;
use std::sync::Arc;

use numr::runtime::wgpu::WgpuRuntime;

use super::refs::{FrameRefs, NamedFrame, RefPool, SLOTS};
use super::std_video::{
    check_tile_bytes, offsets, std_color, std_loop_filter, std_picture_info, std_segmentation,
};
use crate::bitstream::{
    KEY_FRAME, RefFrameTable, UncompressedHeader, Vp9HeaderParser, split_superframe,
};
use crate::decode::vp9_frame::{check_layout, vp9_color_facts};
use crate::decode::{DecodeCodec, DecodedFrame, PixelFormat};
use crate::error::DecodeError;
use crate::vulkan::caps::{DecodeCaps, DpbPreference, check_stream, dpb_mode};
use crate::vulkan::copy::whole_frame as to_frame;
use crate::vulkan::device::VulkanDevice;
use crate::vulkan::engine::{CopySource, DecodeEngine, DecodeRequest, EngineSpec, SlotRef};
use crate::vulkan::error::{DECODER, request_error};
use crate::vulkan::gap::{self, VideoDecodeVp9PictureInfo};
use crate::vulkan::session::{VideoProfile, query_vp9_caps};

/// `VK_MAX_VIDEO_VP9_REFERENCES_PER_FRAME_KHR`.
const ACTIVE_REFERENCES: u32 = gap::MAX_VP9_REFERENCES as u32;

/// The session in use.
struct Active {
    /// What the driver lists for this profile.
    caps: DecodeCaps,
    profile: u8,
    format: PixelFormat,
    engine: DecodeEngine,
}

/// VP9 on Vulkan Video. See [`crate::vulkan::VulkanDecoder`].
pub(in crate::vulkan) struct Vp9Decoder {
    device: Arc<VulkanDevice>,
    preference: DpbPreference,
    /// The smallest session size a caller asks for.
    max_coded: (u32, u32),
    parser: Vp9HeaderParser,
    refs: RefFrameTable,
    pool: RefPool,
    active: Option<Active>,
    frames: VecDeque<DecodedFrame<WgpuRuntime>>,
}

/// The refusal of a frame that outgrows its session at a frame that keeps references.
fn growth_refusal(coded: (u32, u32), session: (u32, u32)) -> DecodeError {
    DecodeError::VulkanRefused {
        codec: DecodeCodec::Vp9,
        what: format!(
            "a frame of {}x{} that keeps its references",
            coded.0, coded.1
        ),
        reason: format!(
            "the session holds at most {}x{}, and a new session has none of the references. \
             Open with VulkanDecoder::with_max_coded_size at the stream's largest frame",
            session.0, session.1
        ),
    }
}

impl Vp9Decoder {
    /// A decoder with no session yet. The first frame opens one. `max_coded` sizes that
    /// session, so a later frame up to this size keeps its references.
    pub(in crate::vulkan) fn new(
        device: Arc<VulkanDevice>,
        preference: DpbPreference,
        max_coded: (u32, u32),
    ) -> Self {
        Self {
            device,
            preference,
            max_coded,
            parser: Vp9HeaderParser::new(),
            refs: RefFrameTable::default(),
            pool: RefPool::default(),
            active: None,
            frames: VecDeque::new(),
        }
    }

    /// Decodes each coded frame of a chunk, and queues the last shown one, as libvpx does.
    pub(in crate::vulkan) fn send(&mut self, packet: &[u8], pts: i64) -> Result<(), DecodeError> {
        let mut shown = None;
        for frame in split_superframe(packet)? {
            let header = self.parser.parse(frame, &self.refs)?;
            check_layout(DECODER, &header)?;
            let output = if header.show_existing_frame {
                Some(self.show_existing(&header, pts)?)
            } else {
                self.decode_frame(frame, &header, pts)?
            };
            if output.is_some() {
                shown = output;
            }
        }
        self.frames.extend(shown);
        Ok(())
    }

    /// Copies a stored slot again. The slot stays a reference.
    fn show_existing(
        &mut self,
        header: &UncompressedHeader,
        pts: i64,
    ) -> Result<DecodedFrame<WgpuRuntime>, DecodeError> {
        let name = header.frame_to_show_map_idx;
        let (Some(named), Some(active)) = (self.pool.get(name), &mut self.active) else {
            return Err(request_error(
                pts,
                format!("show_existing_frame names reference {name}, which holds no frame"),
            ));
        };
        let source = active.engine.read_slot(named.slot, pts)?;
        to_frame(
            &self.device,
            &source,
            active.format,
            named.coded,
            named.color,
            pts,
        )
    }

    /// Puts a session in place that takes this frame. A key frame replaces every reference,
    /// so it can open a new session. Any other frame keeps references, which a new session
    /// lacks, so it refuses a change by name.
    fn prepare_session(
        &mut self,
        header: &UncompressedHeader,
        format: PixelFormat,
        pts: i64,
    ) -> Result<(), DecodeError> {
        let coded = (header.frame_width, header.frame_height);
        let key = header.frame_type == KEY_FRAME;
        if let Some(active) = &self.active {
            // A session sized past this frame holds no check of the driver's minimum.
            check_stream(
                &active.caps,
                DecodeCodec::Vp9,
                coded,
                SLOTS,
                ACTIVE_REFERENCES,
            )?;
            let session = active.engine.max_coded();
            let fits = coded.0 <= session.0 && coded.1 <= session.1;
            let same = active.profile == header.profile && active.format == format;
            if fits && same {
                return Ok(());
            }
            if !key {
                if !same {
                    return Err(DecodeError::VulkanRefused {
                        codec: DecodeCodec::Vp9,
                        what: format!(
                            "profile {} at {} bits at a frame that keeps its references",
                            header.profile,
                            format.bit_depth()
                        ),
                        reason: "a session holds one profile and depth. Re-encode the stream to \
                                 change them at a key frame"
                            .to_owned(),
                    });
                }
                return Err(growth_refusal(coded, session));
            }
        } else if !header.frame_is_intra {
            return Err(request_error(
                pts,
                "the first frame is an inter frame with no reference. Start the stream at a key \
                 frame"
                    .to_owned(),
            ));
        }
        // The old session goes first, so its memory is free for the new one.
        self.active = None;
        self.pool = RefPool::default();
        let profile = VideoProfile::vp9(u32::from(header.profile), format);
        let caps = query_vp9_caps(&self.device, &profile)?;
        let size = (
            coded.0.max(self.max_coded.0.min(caps.max_coded.0)),
            coded.1.max(self.max_coded.1.min(caps.max_coded.1)),
        );
        check_stream(&caps, DecodeCodec::Vp9, coded, SLOTS, ACTIVE_REFERENCES)?;
        check_stream(&caps, DecodeCodec::Vp9, size, SLOTS, ACTIVE_REFERENCES)?;
        let mode = dpb_mode(DecodeCodec::Vp9, caps.decode_flags, self.preference)?;
        let engine = DecodeEngine::new(
            &self.device,
            profile,
            &EngineSpec {
                caps: &caps,
                mode,
                coded: size,
                slots: SLOTS,
                active_references: ACTIVE_REFERENCES,
                readable_slots: true,
                output_image: false,
            },
        )?;
        self.active = Some(Active {
            profile: header.profile,
            format,
            caps,
            engine,
        });
        Ok(())
    }

    /// Decodes one frame into a slot. Returns the frame when it shows.
    fn decode_frame(
        &mut self,
        frame: &[u8],
        header: &UncompressedHeader,
        pts: i64,
    ) -> Result<Option<DecodedFrame<WgpuRuntime>>, DecodeError> {
        check_tile_bytes(header, frame.len(), pts)?;
        let format =
            PixelFormat::for_bit_depth(header.color_config.bit_depth).ok_or_else(|| {
                request_error(
                    pts,
                    format!("BitDepth {} has no format", header.color_config.bit_depth),
                )
            })?;
        self.prepare_session(header, format, pts)?;
        let (refs, slot) = self.frame_slots(header, pts)?;
        let Some(active) = &mut self.active else {
            return Err(request_error(pts, "no session is open".to_owned()));
        };
        let source = submit_frame(active, frame, header, &refs, slot, pts)?;

        let coded = (header.frame_width, header.frame_height);
        let facts = vp9_color_facts(&header.color_config);
        let shown = if header.show_frame {
            Some(to_frame(
                &self.device,
                &source,
                active.format,
                coded,
                facts,
                pts,
            )?)
        } else {
            None
        };
        // 8.10: the reference update of a frame that finished decoding.
        self.refs.update(header);
        let named = NamedFrame {
            slot,
            coded,
            color: facts,
        };
        for freed in self.pool.refresh(header.refresh_frame_flags, named) {
            active.engine.release_slot(freed);
        }
        Ok(shown)
    }

    /// The slots `header` reads, and the free slot it takes.
    fn frame_slots(
        &self,
        header: &UncompressedHeader,
        pts: i64,
    ) -> Result<(FrameRefs, u32), DecodeError> {
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
        let Some(slot) = self.pool.free_slot() else {
            return Err(request_error(
                pts,
                "every DPB slot holds a frame".to_owned(),
            ));
        };
        Ok((refs, slot))
    }

    /// The next frame in decode order. VP9 output has no reorder.
    pub(in crate::vulkan) fn receive(&mut self) -> Option<DecodedFrame<WgpuRuntime>> {
        self.frames.pop_front()
    }
}

/// Runs one frame's decode. Every structure the driver reads lives in this function, through
/// the decode call.
fn submit_frame(
    active: &mut Active,
    frame: &[u8],
    header: &UncompressedHeader,
    refs: &FrameRefs,
    slot: u32,
    pts: i64,
) -> Result<CopySource, DecodeError> {
    let color = std_color(&header.color_config);
    let loop_filter = std_loop_filter(&header.loop_filter);
    let segmentation = std_segmentation(&header.segmentation);
    let mut std_picture = std_picture_info(header);
    std_picture.p_color_config = ptr::from_ref(&color);
    std_picture.p_loop_filter = ptr::from_ref(&loop_filter);
    if header.segmentation.segmentation_enabled {
        std_picture.p_segmentation = ptr::from_ref(&segmentation);
    }
    let (compressed_header_offset, tiles_offset) = offsets(header);
    let picture_info = VideoDecodeVp9PictureInfo {
        s_type: gap::TYPE_VP9_PICTURE_INFO,
        p_next: ptr::null(),
        p_std_picture_info: ptr::from_ref(&std_picture),
        reference_name_slot_indices: refs.name_slots,
        uncompressed_header_offset: 0,
        compressed_header_offset,
        tiles_offset,
    };
    let references = refs.distinct.map(|named| SlotRef {
        slot: named.map_or(0, |named| named.slot),
        codec_info: ptr::null(),
        coded: named.map_or((0, 0), |named| named.coded),
    });
    let request = DecodeRequest {
        bitstream: &[frame],
        picture_info: ptr::from_ref(&picture_info).cast(),
        setup: SlotRef {
            slot,
            codec_info: ptr::null(),
            coded: (header.frame_width, header.frame_height),
        },
        references: &references[..refs.count],
        reference: header.refresh_frame_flags != 0,
        output: header.show_frame,
        separate_output: false,
        byte_ranges: &[],
        pts,
    };
    // SAFETY: `picture_info` is the VP9 structure of this VP9 session. It and all it
    // points to live through the call, and VP9 has no slot structure.
    unsafe { active.engine.decode(&request) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_growing_frame_names_both_sizes_and_the_next_action() {
        let text = growth_refusal((640, 360), (320, 180)).to_string();
        assert!(text.contains("640x360"), "{text}");
        assert!(text.contains("320x180"), "{text}");
        assert!(text.contains("with_max_coded_size"), "{text}");
    }
}
