//! One H.264 stream on the Vulkan core: parameter sets, sessions, pictures, and output.

use std::collections::{HashMap, VecDeque};
use std::ptr;
use std::sync::Arc;

use ash::vk;
use h264_reader::Context;
use h264_reader::nal::pps::PicParameterSet;
use h264_reader::nal::sps::{SeqParamSetId, SeqParameterSet};
use numr::runtime::wgpu::WgpuRuntime;

use super::dpb::{CurrentPicture, Dpb, Marking, marked_frame};
use super::output::OutputQueue;
use super::params::{
    DECODER, SequenceFacts, check_level, check_pps, check_sps, parse_pps, parse_sps, sequence_facts,
};
use super::poc::PocState;
use super::slice::{Picture, parse_slice};
use super::std_video::{PictureFacts, StdPps, StdSps, std_picture_info, std_reference_info};
use crate::bitstream::{AvcConfig, H264NalType, ToAnnexB, annexb, h264_sps_vui};
use crate::decode::{ColorFacts, DecodeCodec, DecodedFrame, PixelFormat, vui_color};
use crate::error::DecodeError;
use crate::geometry::Dimensions;
use crate::vulkan::caps::{DecodeCaps, DpbPreference, check_stream, choose_dpb_mode};
use crate::vulkan::copy::copy_planes;
use crate::vulkan::device::VulkanDevice;
use crate::vulkan::engine::{DecodeEngine, DecodeRequest, EngineSpec, SlotRef};
use crate::vulkan::session::{SessionParameters, VideoProfile, query_h264_caps};

/// What a session was made for. The driver fixes the profile, size, and slots at session
/// creation, and each activation checks the level against the driver.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SessionKey {
    profile_idc: u8,
    level_idc: u8,
    coded: (u32, u32),
    slots: u32,
}

impl SessionKey {
    const fn of(facts: &SequenceFacts) -> Self {
        Self {
            profile_idc: facts.profile_idc,
            level_idc: facts.level_idc,
            coded: facts.coded,
            slots: facts.slots(),
        }
    }
}

/// The sequence in decode.
struct Active {
    sps_id: u8,
    sps_bytes: Vec<u8>,
    facts: SequenceFacts,
    key: SessionKey,
    engine: DecodeEngine,
    dpb: Dpb,
    poc: PocState,
    display: Dimensions,
    color: ColorFacts,
}

/// H.264 on Vulkan Video. See [`crate::vulkan::VulkanDecoder`].
pub(in crate::vulkan) struct H264Decoder {
    device: Arc<VulkanDevice>,
    preference: DpbPreference,
    filter: Option<ToAnnexB>,
    scratch: Vec<u8>,
    context: Context,
    sps_bytes: HashMap<u8, Vec<u8>>,
    pps_bytes: HashMap<u8, Vec<u8>>,
    parameters_stale: bool,
    active: Option<Active>,
    output: OutputQueue<DecodedFrame<WgpuRuntime>>,
    frames: VecDeque<DecodedFrame<WgpuRuntime>>,
    next_id: u64,
}

fn references_error(pts: i64, reason: String) -> DecodeError {
    DecodeError::H264References {
        decoder: DECODER,
        pts,
        reason,
    }
}

/// Empties the output queue before an IDR picture or a new sequence, as C.4.4 states.
fn empty_output(
    output: &mut OutputQueue<DecodedFrame<WgpuRuntime>>,
    frames: &mut VecDeque<DecodedFrame<WgpuRuntime>>,
    without_output: bool,
) {
    if without_output {
        output.discard();
    } else {
        frames.extend(output.drain());
    }
}

/// Refuses a picture that cannot follow what the DPB holds.
fn check_picture_start(
    dpb: &Dpb,
    idr: bool,
    intra: bool,
    frame_num: u32,
    pts: i64,
) -> Result<(), DecodeError> {
    if !dpb.started() && !idr && !intra {
        return Err(references_error(
            pts,
            "the first picture is neither IDR nor intra".to_owned(),
        ));
    }
    if !idr && dpb.has_gap(frame_num) {
        return Err(DecodeError::H264Tool {
            decoder: DECODER,
            field: "a gap in frame_num at frame_num",
            value: u64::from(frame_num),
            supported: "consecutive frame numbers",
        });
    }
    Ok(())
}

/// Runs one picture's decode and copy. Every structure the driver reads lives in this
/// function, through the decode call.
fn decode_frame(
    device: &VulkanDevice,
    active: &mut Active,
    picture: &Picture<'_>,
    current: CurrentPicture,
) -> Result<DecodedFrame<WgpuRuntime>, DecodeError> {
    let facts = &picture.facts;
    let pts = picture.pts;
    let reference = facts.marking != Marking::NonReference;
    let std_picture = std_picture_info(&PictureFacts {
        sps_id: facts.sps_id,
        pps_id: facts.pps_id,
        frame_num: facts.frame_num as u16,
        idr_pic_id: facts.idr_pic_id as u16,
        poc: current.poc,
        idr: facts.idr,
        intra: picture.intra,
        reference,
    });
    let picture_info = vk::VideoDecodeH264PictureInfoKHR::default()
        .std_picture_info(&std_picture)
        .slice_offsets(&picture.slice_offsets);

    // An IDR picture refers to nothing, so its decode binds no reference slot.
    let held = if facts.idr {
        &[][..]
    } else {
        active.dpb.references()
    };
    let reference_infos: Vec<_> = held.iter().map(std_reference_info).collect();
    let dpb_infos: Vec<_> = reference_infos
        .iter()
        .map(|info| vk::VideoDecodeH264DpbSlotInfoKHR::default().std_reference_info(info))
        .collect();
    let references: Vec<_> = held
        .iter()
        .zip(&dpb_infos)
        .map(|(frame, info)| SlotRef {
            slot: frame.slot,
            codec_info: ptr::from_ref(info).cast(),
        })
        .collect();
    // The setup slot holds the picture as its marking leaves it. The picture info above
    // keeps the header's values, which the picture's own decode reads.
    let setup_info = std_reference_info(&marked_frame(current, &facts.marking));
    let setup_dpb = vk::VideoDecodeH264DpbSlotInfoKHR::default().std_reference_info(&setup_info);
    let request = DecodeRequest {
        bitstream: &picture.parts,
        picture_info: ptr::from_ref(&picture_info).cast(),
        setup: SlotRef {
            slot: current.slot,
            codec_info: ptr::from_ref(&setup_dpb).cast(),
        },
        references: &references,
        reference,
        pts,
    };
    // SAFETY: each pointer in `request` names a H.264 structure above, for this H.264
    // session, and each one lives through the call.
    let source = unsafe { active.engine.decode(&request) }?;
    let (y, uv) = copy_planes(device, &source, active.facts.area)?;
    Ok(DecodedFrame::from_packed_planes(
        y,
        uv,
        PixelFormat::Nv12,
        pts,
        active.display,
        active.color,
    ))
}

impl H264Decoder {
    /// Opens a decoder. With an `avcC` in `config`, each parameter set in it goes in, and
    /// the driver answers for its first SPS here, so a refusal comes before any packet.
    pub(in crate::vulkan) fn new(
        device: Arc<VulkanDevice>,
        config: &[u8],
        preference: DpbPreference,
    ) -> Result<Self, DecodeError> {
        let mut decoder = Self {
            device,
            preference,
            filter: None,
            scratch: Vec::new(),
            context: Context::new(),
            sps_bytes: HashMap::new(),
            pps_bytes: HashMap::new(),
            parameters_stale: false,
            active: None,
            output: OutputQueue::new(0, 1),
            frames: VecDeque::new(),
            next_id: 0,
        };
        if !config.is_empty() {
            let parsed = AvcConfig::parse(config)?;
            for sps in &parsed.sps {
                decoder.store_sps(sps)?;
            }
            for pps in &parsed.pps {
                decoder.store_pps(pps)?;
            }
            if let Some(first) = parsed.sps.first() {
                let sps = parse_sps(first)?;
                decoder.activate(sps.seq_parameter_set_id.id())?;
            }
            decoder.filter = Some(ToAnnexB::h264(&parsed));
        }
        Ok(decoder)
    }

    fn store_sps(&mut self, nal: &[u8]) -> Result<(), DecodeError> {
        let sps = parse_sps(nal)?;
        check_sps(&sps)?;
        let id = sps.seq_parameter_set_id.id();
        if self.sps_bytes.get(&id).is_some_and(|bytes| bytes == nal) {
            return Ok(());
        }
        self.context.put_seq_param_set(sps);
        self.sps_bytes.insert(id, nal.to_vec());
        self.parameters_stale = true;
        Ok(())
    }

    fn store_pps(&mut self, nal: &[u8]) -> Result<(), DecodeError> {
        let pps = parse_pps(&self.context, nal)?;
        check_pps(&pps)?;
        let id = pps.pic_parameter_set_id.id();
        if self.pps_bytes.get(&id).is_some_and(|bytes| bytes == nal) {
            return Ok(());
        }
        self.context.put_pic_param_set(pps);
        self.pps_bytes.insert(id, nal.to_vec());
        self.parameters_stale = true;
        Ok(())
    }

    fn sps(&self, id: u8) -> Result<&SeqParameterSet, DecodeError> {
        SeqParamSetId::from_u32(u32::from(id))
            .ok()
            .and_then(|id| self.context.sps_by_id(id))
            .ok_or_else(|| DecodeError::H264Header {
                decoder: DECODER,
                unit: "slice",
                reason: format!("SPS {id} is absent"),
            })
    }

    /// Makes SPS `id` the active sequence, with an empty DPB. The session stays when the
    /// new sequence fits it.
    fn activate(&mut self, id: u8) -> Result<(), DecodeError> {
        let sps = self.sps(id)?;
        let facts = sequence_facts(sps)?;
        let key = SessionKey::of(&facts);
        let nal = self.sps_bytes.get(&id).cloned().unwrap_or_default();
        let display = Dimensions::new(facts.area.width, facts.area.height).map_err(|_| {
            DecodeError::DisplayArea {
                rect: (
                    facts.area.left as i32,
                    facts.area.top as i32,
                    (facts.area.left + facts.area.width) as i32,
                    (facts.area.top + facts.area.height) as i32,
                ),
                coded: facts.coded,
            }
        })?;
        let color = vui_color(&h264_sps_vui(&nal)?);
        // The driver answers for every sequence, also for one that keeps its session.
        let profile = VideoProfile::h264(facts.profile_idc);
        let (caps, max_level) = query_h264_caps(&self.device, &profile)?;
        check_level(facts.level_idc, max_level)?;
        check_stream(
            &caps,
            DecodeCodec::H264,
            facts.coded,
            facts.slots(),
            facts.max_references,
        )?;
        let dpb = Dpb::new(facts.slots(), facts.max_references, facts.max_frame_num);
        self.output
            .set_limits(facts.reorder_frames, facts.buffer_frames);

        let previous = self.active.take();
        let engine = match previous {
            Some(active) if active.key == key => active.engine,
            other => {
                // The old session goes first, so its memory is free for the new one.
                drop(other);
                self.open_engine(&facts, profile, &caps)?
            }
        };
        self.parameters_stale = true;
        self.active = Some(Active {
            sps_id: id,
            sps_bytes: nal,
            facts,
            key,
            engine,
            dpb,
            poc: PocState::default(),
            display,
            color,
        });
        Ok(())
    }

    fn open_engine(
        &self,
        facts: &SequenceFacts,
        profile: VideoProfile,
        caps: &DecodeCaps,
    ) -> Result<DecodeEngine, DecodeError> {
        let mode = choose_dpb_mode(caps.decode_flags, self.preference).map_err(|reason| {
            DecodeError::VulkanRefused {
                codec: DecodeCodec::H264,
                what: "a DPB mode".to_owned(),
                reason,
            }
        })?;
        DecodeEngine::new(
            &self.device,
            profile,
            &EngineSpec {
                caps,
                mode,
                coded: facts.coded,
                slots: facts.slots(),
                active_references: facts.max_references,
            },
        )
    }

    /// Takes one Annex-B access unit, or a sample through the filter.
    pub(in crate::vulkan) fn send(&mut self, packet: &[u8], pts: i64) -> Result<(), DecodeError> {
        let mut scratch = std::mem::take(&mut self.scratch);
        scratch.clear();
        let result = match &mut self.filter {
            Some(filter) => filter
                .filter_into(packet, &mut scratch)
                .map_err(DecodeError::from)
                .and_then(|()| self.read_unit(&scratch, pts)),
            None => self.read_unit(packet, pts),
        };
        self.scratch = scratch;
        result
    }

    fn read_unit(&mut self, unit: &[u8], pts: i64) -> Result<(), DecodeError> {
        let mut current = None;
        for nal in annexb::split(unit)? {
            let Some(kind) = H264NalType::of(nal) else {
                continue;
            };
            match kind {
                H264NalType::SPS => {
                    self.finish_picture(&mut current)?;
                    self.store_sps(nal)?;
                }
                H264NalType::PPS => {
                    self.finish_picture(&mut current)?;
                    self.store_pps(nal)?;
                }
                H264NalType::AUD | H264NalType::END_OF_SEQUENCE | H264NalType::END_OF_STREAM => {
                    self.finish_picture(&mut current)?;
                }
                H264NalType::SLICE | H264NalType::IDR => {
                    self.read_slice(&mut current, nal, pts)?;
                }
                other if other.is_vcl() => {
                    return Err(DecodeError::H264Tool {
                        decoder: DECODER,
                        field: "nal_unit_type",
                        value: u64::from(other.value()),
                        supported: "1 and 5, because Vulkan Video has no data partitioning",
                    });
                }
                _ => {}
            }
        }
        // One packet holds one access unit, so its end ends the picture.
        self.finish_picture(&mut current)
    }

    fn read_slice<'u>(
        &mut self,
        current: &mut Option<Picture<'u>>,
        nal: &'u [u8],
        pts: i64,
    ) -> Result<(), DecodeError> {
        let facts = parse_slice(&self.context, nal)?;
        // A redundant coded picture repeats what the primary one holds.
        if facts.redundant_pic_cnt > 0 {
            return Ok(());
        }
        let continues = current
            .as_ref()
            .is_some_and(|picture| !facts.starts_picture_after(&picture.facts));
        let intra = facts.intra;
        if !continues {
            self.finish_picture(current)?;
            *current = Some(Picture::new(facts, pts));
        }
        if let Some(picture) = current {
            picture.push_slice(intra, nal);
        }
        Ok(())
    }

    /// Decodes the gathered picture, when one is waiting.
    fn finish_picture(&mut self, current: &mut Option<Picture<'_>>) -> Result<(), DecodeError> {
        let Some(picture) = current.take() else {
            return Ok(());
        };
        self.prepare_sequence(&picture)?;
        self.decode_picture(&picture)
    }

    /// Puts the picture's SPS in place and refreshes the session parameters.
    fn prepare_sequence(&mut self, picture: &Picture<'_>) -> Result<(), DecodeError> {
        let id = picture.facts.sps_id;
        let wanted = self.sps_bytes.get(&id).map_or(&[][..], Vec::as_slice);
        let unchanged = self
            .active
            .as_ref()
            .is_some_and(|active| active.sps_id == id && active.sps_bytes == wanted);
        if !unchanged {
            let started = self
                .active
                .as_ref()
                .is_some_and(|active| active.dpb.started());
            if started && !picture.facts.idr {
                return Err(references_error(
                    picture.pts,
                    format!("SPS {id} takes effect at a non-IDR picture"),
                ));
            }
            // The old sequence ends here, so each frame of it leaves or goes first.
            // Only the stream's flag drops pictures. No change of sequence infers it.
            let without_output = picture.facts.idr && picture.facts.no_output_of_prior_pics;
            empty_output(&mut self.output, &mut self.frames, without_output);
            self.activate(id)?;
        }
        if self.parameters_stale {
            self.refresh_parameters(id)?;
        }
        Ok(())
    }

    fn refresh_parameters(&mut self, id: u8) -> Result<(), DecodeError> {
        let sps = self.sps(id)?;
        let std_sps = StdSps::new(sps);
        let owners: Vec<StdPps> = self
            .context
            .pps()
            .filter(|pps: &&PicParameterSet| pps.seq_parameter_set_id.id() == id)
            .map(|pps| StdPps::new(pps, sps))
            .collect();
        let sps_list = [std_sps.as_std()];
        let pps_list: Vec<_> = owners.iter().map(StdPps::as_std).collect();
        let add = vk::VideoDecodeH264SessionParametersAddInfoKHR::default()
            .std_sp_ss(&sps_list)
            .std_pp_ss(&pps_list);
        let create = vk::VideoDecodeH264SessionParametersCreateInfoKHR::default()
            .max_std_sps_count(1)
            .max_std_pps_count(pps_list.len().max(1) as u32)
            .parameters_add_info(&add);
        let Some(active) = &mut self.active else {
            return Ok(());
        };
        // SAFETY: `create` chains the H.264 structures this session's codec takes, and
        // each owner above lives through the call.
        let parameters = unsafe {
            SessionParameters::new(
                &self.device,
                active.engine.session(),
                ptr::from_ref(&create).cast(),
            )
        }?;
        active.engine.set_parameters(parameters);
        self.parameters_stale = false;
        Ok(())
    }

    fn decode_picture(&mut self, picture: &Picture<'_>) -> Result<(), DecodeError> {
        let facts = &picture.facts;
        let pts = picture.pts;
        let Some(active) = &mut self.active else {
            return Err(references_error(pts, "no SPS is active".to_owned()));
        };
        check_picture_start(&active.dpb, facts.idr, picture.intra, facts.frame_num, pts)?;

        let poc_slice = facts.poc_slice();
        let order = active
            .poc
            .compute(&active.facts.poc, active.facts.max_frame_num, &poc_slice);
        if facts.idr {
            // An IDR picture follows every earlier picture in output order.
            empty_output(
                &mut self.output,
                &mut self.frames,
                facts.no_output_of_prior_pics,
            );
        }

        let Some(slot) = active.dpb.free_slot() else {
            return Err(references_error(
                pts,
                "every DPB slot holds a reference".to_owned(),
            ));
        };
        let id = self.next_id;
        let current = CurrentPicture {
            slot,
            id,
            frame_num: facts.frame_num,
            poc: order,
        };
        let frame = decode_frame(&self.device, active, picture, current)?;
        self.next_id += 1;

        let outcome = active
            .dpb
            .mark(current, &facts.marking)
            .map_err(|reason| references_error(pts, reason))?;
        for freed in &outcome.freed {
            active.engine.release_slot(*freed);
        }
        let mmco5_top = outcome.mmco5_poc.map(|poc| poc[0]);
        active.poc.finish(&poc_slice, mmco5_top);
        if outcome.mmco5_poc.is_some() {
            // Operation 5 resets the order count, so every earlier picture leaves first.
            self.frames.extend(self.output.drain());
        }
        let order = outcome.mmco5_poc.unwrap_or(order);
        self.output.push(id, order[0].min(order[1]), frame);
        let dpb = &active.dpb;
        while let Some(ready) = self
            .output
            .pop_ready(dpb.references().len(), |id| dpb.holds(id))
        {
            self.frames.push_back(ready);
        }
        Ok(())
    }

    /// Ends the stream: every waiting frame leaves.
    pub(in crate::vulkan) fn flush(&mut self) {
        self.frames.extend(self.output.drain());
    }

    /// The next frame in display order.
    pub(in crate::vulkan) fn receive(&mut self) -> Option<DecodedFrame<WgpuRuntime>> {
        self.frames.pop_front()
    }
}

#[cfg(test)]
mod tests {
    use super::super::params::parse_sps;
    use super::super::test_nal::{TestSps, sps_nal};
    use super::*;

    fn facts() -> SequenceFacts {
        let sps = parse_sps(&sps_nal(&TestSps::default())).expect("parses");
        sequence_facts(&sps).expect("facts")
    }

    #[test]
    fn a_session_key_changes_with_what_a_session_was_made_for() {
        let facts = facts();
        let key = SessionKey::of(&facts);
        assert_eq!(key.level_idc, 30);
        assert_eq!(key, SessionKey::of(&facts));
        assert_ne!(
            key,
            SessionKey::of(&SequenceFacts {
                level_idc: 31,
                ..facts.clone()
            })
        );
        assert_ne!(
            key,
            SessionKey::of(&SequenceFacts {
                profile_idc: 77,
                ..facts.clone()
            })
        );
        assert_ne!(
            key,
            SessionKey::of(&SequenceFacts {
                coded: (640, 192),
                ..facts.clone()
            })
        );
        assert_ne!(
            key,
            SessionKey::of(&SequenceFacts {
                max_references: 5,
                ..facts
            })
        );
    }

    #[test]
    fn a_picture_start_is_checked_against_the_dpb() {
        let mut dpb = Dpb::new(3, 2, 16);
        assert!(matches!(
            check_picture_start(&dpb, false, false, 0, 7),
            Err(DecodeError::H264References { pts: 7, .. })
        ));
        assert_eq!(check_picture_start(&dpb, false, true, 0, 7), Ok(()));
        assert_eq!(check_picture_start(&dpb, true, true, 0, 7), Ok(()));
        let first = CurrentPicture {
            slot: 0,
            id: 0,
            frame_num: 0,
            poc: [0, 0],
        };
        dpb.mark(first, &Marking::Idr { long_term: false })
            .expect("fits");
        assert_eq!(check_picture_start(&dpb, false, false, 1, 7), Ok(()));
        assert!(matches!(
            check_picture_start(&dpb, false, false, 2, 7),
            Err(DecodeError::H264Tool { value: 2, .. })
        ));
        assert_eq!(check_picture_start(&dpb, true, true, 0, 7), Ok(()));
    }
}
