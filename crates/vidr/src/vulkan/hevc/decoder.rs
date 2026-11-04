//! One HEVC stream on the Vulkan core: parameter sets, sessions, pictures, and output.

use std::collections::{HashMap, VecDeque};
use std::hash::Hash;
use std::ptr;
use std::sync::Arc;

use ash::vk;
use numr::runtime::wgpu::WgpuRuntime;

use super::picture::Picture;
use super::sequence::{DECODER, SequenceFacts, sequence_facts};
use super::slots::{ActiveReference, ListError, SlotLists, SlotPicture, SlotPool, slot_lists};
use super::std_params::{StdParameterSets, StdPps, StdSps, StdVps};
use super::std_picture::{std_picture_info, std_reference_info};
use crate::bitstream::hevc_pps::PicParameterSet;
use crate::bitstream::hevc_slice::{SliceSegmentHeader, slice_pic_parameter_set_id};
use crate::bitstream::hevc_sps_full::SeqParameterSet;
use crate::bitstream::hevc_vps::VideoParameterSet;
use crate::bitstream::{HevcConfig, HevcNalType, ToAnnexB, VuiSignal, annexb};
use crate::decode::hevc_order::buffer::RefBuffer;
use crate::decode::hevc_order::poc::{PictureFacts, PictureStart, SequenceState, is_bla, is_cra};
use crate::decode::hevc_order::rps::RpsPocs;
use crate::decode::{ColorFacts, DecodeCodec, DecodedFrame, vui_color};
use crate::error::DecodeError;
use crate::geometry::Dimensions;
use crate::vulkan::caps::{DecodeCaps, DpbPreference, check_stream, dpb_mode};
use crate::vulkan::copy::copy_planes;
use crate::vulkan::device::VulkanDevice;
use crate::vulkan::engine::{DecodeEngine, DecodeRequest, EngineSpec, SlotRef};
use crate::vulkan::error::request_error;
use crate::vulkan::session::{SessionParameters, VideoProfile, query_hevc_caps};

/// One parameter set: its parsed form, and its bytes for a change test.
struct Stored<T> {
    bytes: Vec<u8>,
    parsed: T,
}

/// The sequence in decode.
struct Active {
    sps_id: u32,
    sps_bytes: Vec<u8>,
    facts: SequenceFacts,
    engine: DecodeEngine,
    /// What the session parameters' pointers name. NVIDIA's driver reads through them after
    /// `vkCreateVideoSessionParametersKHR` returns. This drops after `engine`, so it outlives the
    /// parameters, and it stays when a new sequence keeps the session.
    parameter_sets: Option<StdParameterSets>,
    pool: SlotPool,
    display: Dimensions,
    color: ColorFacts,
}

/// HEVC on Vulkan Video. See [`crate::vulkan::VulkanDecoder`].
pub(in crate::vulkan) struct HevcDecoder {
    device: Arc<VulkanDevice>,
    preference: DpbPreference,
    filter: Option<ToAnnexB>,
    scratch: Vec<u8>,
    vps: HashMap<u8, Stored<VideoParameterSet>>,
    sps: HashMap<u32, Stored<SeqParameterSet>>,
    pps: HashMap<u32, Stored<PicParameterSet>>,
    parameters_stale: bool,
    sequence: SequenceState,
    buffer: RefBuffer<SlotPicture>,
    active: Option<Active>,
    /// The copied frame of each picture that still waits for output.
    waiting: HashMap<u64, DecodedFrame<WgpuRuntime>>,
    frames: VecDeque<DecodedFrame<WgpuRuntime>>,
    next_id: u64,
}

/// Stores `parsed` under `id` when its bytes differ from what `id` holds. The result states
/// whether the set changed.
fn store_set<K: Eq + Hash, T>(
    sets: &mut HashMap<K, Stored<T>>,
    id: K,
    nal: &[u8],
    parsed: T,
) -> bool {
    if sets.get(&id).is_some_and(|stored| stored.bytes == nal) {
        return false;
    }
    let bytes = nal.to_vec();
    sets.insert(id, Stored { bytes, parsed });
    true
}

/// The refusal of a "Curr" list with no slot. A clear or a copy cannot activate a slot.
fn list_refusal(error: ListError, pts: i64) -> DecodeError {
    match error {
        ListError::Missing { poc, long_term } => {
            let kind = if long_term { "long-term" } else { "short-term" };
            DecodeError::VulkanRefused {
                codec: DecodeCodec::Hevc,
                what: format!("a {kind} reference to POC {poc} at pts {pts}, which the DPB lacks"),
                reason: "Vulkan Video activates a DPB slot only through a decode, so no slot can \
                         stand in for it. Decode this stream with another decoder, or start at \
                         an earlier IRAP picture"
                    .to_owned(),
            }
        }
        ListError::TooLong { entries } => DecodeError::HevcTool {
            decoder: DECODER,
            field: "the entries of a RefPicSet list",
            value: entries as u64,
            supported: "at most 8",
        },
    }
}

/// Refuses a reference picture set past what the buffer holds. The engine holds the "Curr"
/// entries to the session's limit.
fn check_rps_size(rps: &RpsPocs, facts: &SequenceFacts, pts: i64) -> Result<(), DecodeError> {
    if rps.entry_count() > facts.limits.max_dec_buffering {
        return Err(request_error(
            pts,
            format!(
                "the reference picture set holds {} pictures, past \
                 sps_max_dec_pic_buffering_minus1 + 1 of {}",
                rps.entry_count(),
                facts.limits.max_dec_buffering
            ),
        ));
    }
    Ok(())
}

/// Runs one picture's decode, and its copy when `output` holds. Every structure the driver
/// reads lives in this function, through the decode call.
fn decode_frame(
    device: &VulkanDevice,
    active: &mut Active,
    picture: &Picture<'_>,
    lists: &SlotLists,
    current: ActiveReference,
    ids: (u8, u8),
    output: bool,
) -> Result<Option<DecodedFrame<WgpuRuntime>>, DecodeError> {
    let pts = picture.pts;
    let std_picture = std_picture_info(&picture.header, ids.0, ids.1, current.poc, lists);
    let picture_info = vk::VideoDecodeH265PictureInfoKHR::default()
        .std_picture_info(&std_picture)
        .slice_segment_offsets(&picture.offsets);

    let coded = active.facts.coded;
    let reference_infos: Vec<_> = lists.references.iter().map(std_reference_info).collect();
    let dpb_infos: Vec<_> = reference_infos
        .iter()
        .map(|info| vk::VideoDecodeH265DpbSlotInfoKHR::default().std_reference_info(info))
        .collect();
    let references: Vec<_> = lists
        .references
        .iter()
        .zip(&dpb_infos)
        .map(|(reference, info)| SlotRef {
            slot: reference.slot,
            codec_info: ptr::from_ref(info).cast(),
            coded,
        })
        .collect();
    let setup_info = std_reference_info(&current);
    let setup_dpb = vk::VideoDecodeH265DpbSlotInfoKHR::default().std_reference_info(&setup_info);
    let request = DecodeRequest {
        bitstream: &picture.parts,
        picture_info: ptr::from_ref(&picture_info).cast(),
        setup: SlotRef {
            slot: current.slot,
            codec_info: ptr::from_ref(&setup_dpb).cast(),
            coded,
        },
        references: &references,
        reference: true,
        output,
        separate_output: false,
        byte_ranges: &[],
        pts,
    };
    // SAFETY: each pointer in `request` names a H.265 structure above, for this H.265 session,
    // and each one lives through the call.
    let source = unsafe { active.engine.decode(&request) }?;
    if !output {
        return Ok(None);
    }
    let format = active.facts.format;
    let (y, uv) = copy_planes(device, &source, active.facts.area, format)?;
    Ok(Some(DecodedFrame::from_packed_planes(
        y,
        uv,
        format,
        pts,
        active.display,
        active.color,
    )))
}

impl HevcDecoder {
    /// Opens a decoder. With an `hvcC` in `config`, each parameter set in it goes in. The
    /// driver answers for its first SPS here, so a refusal comes before any packet.
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
            vps: HashMap::new(),
            sps: HashMap::new(),
            pps: HashMap::new(),
            parameters_stale: false,
            sequence: SequenceState::new(),
            buffer: RefBuffer::new(),
            active: None,
            waiting: HashMap::new(),
            frames: VecDeque::new(),
            next_id: 0,
        };
        if !config.is_empty() {
            let parsed = HevcConfig::parse(config)?;
            for nal in parsed.nal_units() {
                decoder.read_parameter_set(nal)?;
            }
            let first = parsed
                .nal_units()
                .find(|nal| {
                    nal.first()
                        .is_some_and(|&byte| HevcNalType::from_header(byte) == HevcNalType::SPS)
                })
                .map(SeqParameterSet::parse)
                .transpose()?;
            if let Some(sps) = first {
                decoder.activate(sps.sps_seq_parameter_set_id)?;
            }
            decoder.filter = Some(ToAnnexB::hevc(&parsed));
        }
        Ok(decoder)
    }

    /// Stores one VPS, SPS, or PPS. Other units pass. A new set makes the session parameters
    /// stale, and the next picture refreshes them.
    fn read_parameter_set(&mut self, nal: &[u8]) -> Result<(), DecodeError> {
        let Some(&first) = nal.first() else {
            return Ok(());
        };
        let kind = HevcNalType::from_header(first);
        if kind == HevcNalType::VPS {
            let parsed = VideoParameterSet::parse(nal)?;
            let id = parsed.vps_video_parameter_set_id;
            self.parameters_stale |= store_set(&mut self.vps, id, nal, parsed);
        } else if kind == HevcNalType::SPS {
            let parsed = SeqParameterSet::parse(nal)?;
            sequence_facts(&parsed)?;
            let id = parsed.sps_seq_parameter_set_id;
            self.parameters_stale |= store_set(&mut self.sps, id, nal, parsed);
        } else if kind == HevcNalType::PPS {
            let parsed = PicParameterSet::parse(nal)?;
            let id = parsed.pps_pic_parameter_set_id;
            self.parameters_stale |= store_set(&mut self.pps, id, nal, parsed);
        }
        Ok(())
    }

    /// Makes SPS `id` the active sequence. The caller emptied the buffer. The session stays
    /// when the new sequence fits it.
    fn activate(&mut self, id: u32) -> Result<(), DecodeError> {
        let Some(stored) = self.sps.get(&id) else {
            return Err(DecodeError::VulkanRefused {
                codec: DecodeCodec::Hevc,
                what: format!("SPS {id}"),
                reason: "the stream names it and holds no such SPS. Send the parameter sets \
                         before the slice"
                    .to_owned(),
            });
        };
        let facts = sequence_facts(&stored.parsed)?;
        let display = facts.area.display(facts.coded)?;
        let signal = stored
            .parsed
            .vui
            .map_or_else(VuiSignal::default, |vui| vui.signal);
        let color = vui_color(&signal);
        let sps_bytes = stored.bytes.clone();

        let profile = VideoProfile::hevc(facts.std_profile_idc, facts.format);
        let caps = self.driver_caps(&facts, &profile)?;
        let (engine, parameter_sets) = self.session_for(&facts, profile, &caps)?;
        self.parameters_stale = true;
        self.active = Some(Active {
            sps_id: id,
            sps_bytes,
            facts,
            engine,
            parameter_sets,
            pool: SlotPool::new(facts.slots),
            display,
            color,
        });
        Ok(())
    }

    /// What the driver lists for `profile`, or the refusal of a sequence past it.
    fn driver_caps(
        &self,
        facts: &SequenceFacts,
        profile: &VideoProfile,
    ) -> Result<DecodeCaps, DecodeError> {
        let (caps, max_level) = query_hevc_caps(&self.device, profile)?;
        if facts.std_level > max_level {
            return Err(DecodeError::VulkanRefused {
                codec: DecodeCodec::Hevc,
                what: format!("StdVideoH265LevelIdc {}", facts.std_level),
                reason: format!("the driver lists at most {max_level}"),
            });
        }
        check_stream(
            &caps,
            DecodeCodec::Hevc,
            facts.coded,
            facts.slots,
            facts.active_references,
        )?;
        Ok(caps)
    }

    /// Keeps the session of the active sequence when it fits `facts`, or opens a new one. A kept
    /// session keeps its parameters, so what they point to stays too.
    fn session_for(
        &mut self,
        facts: &SequenceFacts,
        profile: VideoProfile,
        caps: &DecodeCaps,
    ) -> Result<(DecodeEngine, Option<StdParameterSets>), DecodeError> {
        match self.active.take() {
            Some(active) if same_session(&active.facts, facts) => {
                Ok((active.engine, active.parameter_sets))
            }
            other => {
                // The old session goes first, so its memory is free for the new one.
                drop(other);
                let mode = dpb_mode(DecodeCodec::Hevc, caps.decode_flags, self.preference)?;
                let engine = DecodeEngine::new(
                    &self.device,
                    profile,
                    &EngineSpec {
                        caps,
                        mode,
                        coded: facts.coded,
                        slots: facts.slots,
                        active_references: facts.active_references,
                        readable_slots: false,
                        output_image: false,
                    },
                )?;
                Ok((engine, None))
            }
        }
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
        let mut independent = None;
        for nal in annexb::split(unit)? {
            let (Some(&first), Some(&second)) = (nal.first(), nal.get(1)) else {
                continue;
            };
            // nuh_layer_id: the low bit of the first byte and the top five of the second.
            if first & 1 != 0 || second >> 3 != 0 {
                continue;
            }
            let kind = HevcNalType::from_header(first);
            if kind == HevcNalType::VPS || kind == HevcNalType::SPS || kind == HevcNalType::PPS {
                self.finish_picture(&mut current)?;
                self.read_parameter_set(nal)?;
            } else if kind == HevcNalType::AUD || kind == HevcNalType::SEI_PREFIX {
                self.finish_picture(&mut current)?;
            } else if kind == HevcNalType::END_OF_SEQUENCE || kind == HevcNalType::END_OF_BITSTREAM
            {
                self.finish_picture(&mut current)?;
                // 8.1.3: the next picture restarts POC and NoRaslOutputFlag.
                self.sequence.end_of_sequence();
            } else if matches!(kind.value(), 0..=9 | 16..=21) {
                self.read_slice(&mut current, &mut independent, nal, pts)?;
            }
        }
        // One packet holds one access unit, so its end ends the picture.
        self.finish_picture(&mut current)
    }

    fn read_slice<'u>(
        &mut self,
        current: &mut Option<Picture<'u>>,
        independent: &mut Option<SliceSegmentHeader>,
        nal: &'u [u8],
        pts: i64,
    ) -> Result<(), DecodeError> {
        let pps_id = slice_pic_parameter_set_id(nal)?;
        let pps = self.pps.get(&pps_id).ok_or_else(|| {
            request_error(pts, format!("a slice names PPS {pps_id}, which is absent"))
        })?;
        let sps_id = pps.parsed.pps_seq_parameter_set_id;
        let sps = self.sps.get(&sps_id).ok_or_else(|| {
            request_error(
                pts,
                format!("PPS {pps_id} names SPS {sps_id}, which is absent"),
            )
        })?;
        let header =
            SliceSegmentHeader::parse(nal, &pps.parsed, &sps.parsed, independent.as_ref())?;
        if !header.dependent_slice_segment_flag {
            *independent = Some(header.clone());
        }
        if header.first_slice_segment_in_pic_flag {
            self.finish_picture(current)?;
            *current = Some(Picture::new(header, pts));
        }
        match current {
            Some(picture) => {
                picture.push_segment(nal);
                Ok(())
            }
            None => Err(request_error(
                pts,
                "a slice segment arrives with no first slice segment in its packet. Send one \
                 access unit per packet"
                    .to_owned(),
            )),
        }
    }

    /// Decodes the gathered picture, when one is waiting.
    fn finish_picture(&mut self, current: &mut Option<Picture<'_>>) -> Result<(), DecodeError> {
        match current.take() {
            Some(picture) => self.decode_picture(&picture),
            None => Ok(()),
        }
    }

    fn decode_picture(&mut self, picture: &Picture<'_>) -> Result<(), DecodeError> {
        let header = &picture.header;
        let pts = picture.pts;
        let pps_id = header.slice_pic_parameter_set_id;
        let absent = || request_error(pts, format!("PPS {pps_id} or its SPS is absent"));
        let sps_id = self
            .pps
            .get(&pps_id)
            .map(|pps| pps.parsed.pps_seq_parameter_set_id)
            .ok_or_else(absent)?;
        let sps = self.sps.get(&sps_id).ok_or_else(absent)?;
        let vps_id = sps.parsed.sps_video_parameter_set_id;
        let max_poc_lsb = 1u32 << (sps.parsed.log2_max_pic_order_cnt_lsb_minus4 + 4);
        let start = self.sequence.begin_picture(PictureFacts {
            nal_unit_type: header.nal_unit_type,
            temporal_id: header.temporal_id,
            poc_lsb: header.slice_pic_order_cnt_lsb,
            max_poc_lsb,
        });
        let PictureStart::Decode { poc, irap_no_rasl } = start else {
            return Ok(());
        };
        let unchanged = self
            .active
            .as_ref()
            .is_some_and(|active| active.sps_id == sps_id && active.sps_bytes == sps.bytes);
        if !unchanged && !irap_no_rasl {
            return Err(request_error(
                pts,
                format!(
                    "SPS {sps_id} takes effect at nal_unit_type {}, and a new sequence starts at \
                     an IRAP picture with NoRaslOutputFlag 1",
                    header.nal_unit_type
                ),
            ));
        }
        let facts = sequence_facts(&sps.parsed)?;
        let rps = RpsPocs::derive(header, &sps.parsed, poc);
        check_rps_size(&rps, &facts, pts)?;
        let sets = self.buffer.apply_rps(&rps, max_poc_lsb, irap_no_rasl);
        // C.5.2.2: a CRA picture sets NoOutputOfPriorPicsFlag to 1.
        let no_output = is_cra(header.nal_unit_type) || header.no_output_of_prior_pics_flag;
        self.buffer
            .before_picture(irap_no_rasl, no_output, &facts.limits);
        self.collect();
        if !unchanged {
            self.activate(sps_id)?;
        }
        if self.parameters_stale {
            self.refresh_parameters(sps_id)?;
        }
        if is_bla(header.nal_unit_type) || (is_cra(header.nal_unit_type) && irap_no_rasl) {
            self.generate_missing(&sets.missing_foll);
        }
        let lists = slot_lists(&sets).map_err(|error| list_refusal(error, pts))?;
        self.run_picture(picture, &lists, poc, (vps_id, sps_id as u8))
    }

    /// Stores each picture of 8.3.3, which takes no slot.
    fn generate_missing(&mut self, missing_foll: &[(i32, bool)]) {
        for &(missing, long_term) in missing_foll {
            let generated = SlotPicture {
                poc: missing,
                slot: None,
                id: u64::MAX,
            };
            self.buffer.insert_generated(generated, long_term);
        }
    }

    /// Decodes one picture into a free slot, and stores it in the buffer.
    fn run_picture(
        &mut self,
        picture: &Picture<'_>,
        lists: &SlotLists,
        poc: i32,
        ids: (u8, u8),
    ) -> Result<(), DecodeError> {
        let pts = picture.pts;
        let Some(active) = &mut self.active else {
            return Err(request_error(pts, "no SPS is active".to_owned()));
        };
        let Some(slot) = active.pool.take() else {
            return Err(request_error(
                pts,
                format!(
                    "every one of {} DPB slots holds a picture, so the stream passes its \
                     sps_max_dec_pic_buffering_minus1",
                    active.facts.slots
                ),
            ));
        };
        let current = ActiveReference {
            slot,
            poc,
            long_term: false,
        };
        let output = picture.header.pic_output_flag;
        let frame = decode_frame(&self.device, active, picture, lists, current, ids, output)?;
        let limits = active.facts.limits;
        let id = self.next_id;
        self.next_id += 1;
        if let Some(frame) = frame {
            self.waiting.insert(id, frame);
        }
        let stored = SlotPicture {
            poc,
            slot: Some(slot),
            id,
        };
        self.buffer.insert_picture(stored, output, &limits);
        self.collect();
        Ok(())
    }

    /// Moves each output picture's frame out, and frees what each removed picture held. The
    /// output comes first, because a bump can output a picture and remove it.
    fn collect(&mut self) {
        while let Some(picture) = self.buffer.pop_output() {
            if let Some(frame) = self.waiting.remove(&picture.id) {
                self.frames.push_back(frame);
            }
        }
        for picture in self.buffer.take_removed() {
            self.waiting.remove(&picture.id);
            if let (Some(slot), Some(active)) = (picture.slot, &mut self.active) {
                active.pool.release(slot);
                active.engine.release_slot(slot);
            }
        }
    }

    fn refresh_parameters(&mut self, sps_id: u32) -> Result<(), DecodeError> {
        let Some(sps) = self.sps.get(&sps_id) else {
            return Ok(());
        };
        let sps = &sps.parsed;
        let Some(vps) = self.vps.get(&sps.sps_video_parameter_set_id) else {
            return Err(DecodeError::VulkanRefused {
                codec: DecodeCodec::Hevc,
                what: format!("VPS {}", sps.sps_video_parameter_set_id),
                reason: format!(
                    "SPS {sps_id} names it and the stream holds no such VPS. Vulkan Video reads \
                     it from the session parameters"
                ),
            });
        };
        let mut pps_owners = Vec::new();
        for stored in self.pps.values() {
            if stored.parsed.pps_seq_parameter_set_id == sps_id {
                stored.parsed.check_with_sps(sps)?;
                pps_owners.push(StdPps::new(&stored.parsed, sps));
            }
        }
        let sets = StdParameterSets::new(StdVps::new(&vps.parsed), StdSps::new(sps), pps_owners);
        let add = vk::VideoDecodeH265SessionParametersAddInfoKHR::default()
            .std_vp_ss(sets.vps_list())
            .std_sp_ss(sets.sps_list())
            .std_pp_ss(sets.pps_list());
        let create = vk::VideoDecodeH265SessionParametersCreateInfoKHR::default()
            .max_std_vps_count(1)
            .max_std_sps_count(1)
            .max_std_pps_count(sets.pps_list().len().max(1) as u32)
            .parameters_add_info(&add);
        let Some(active) = &mut self.active else {
            return Ok(());
        };
        // SAFETY: `create` chains the H.265 structures this session's codec takes. `sets` lives
        // through the call, then moves into `active`, and a move keeps each heap address.
        let parameters = unsafe {
            SessionParameters::new(
                &self.device,
                active.engine.session(),
                ptr::from_ref(&create).cast(),
            )
        }?;
        active.engine.set_parameters(parameters);
        // The old sets go only now: `set_parameters` dropped the object that points to them.
        active.parameter_sets = Some(sets);
        self.parameters_stale = false;
        Ok(())
    }

    /// Ends the stream: every waiting frame leaves.
    pub(in crate::vulkan) fn flush(&mut self) {
        self.buffer.flush();
        self.collect();
    }

    /// The next frame in output order.
    pub(in crate::vulkan) fn receive(&mut self) -> Option<DecodedFrame<WgpuRuntime>> {
        self.frames.pop_front()
    }
}

/// Whether a session made for `old` also takes `new`. The driver fixes the profile, size,
/// and slots at session creation.
fn same_session(old: &SequenceFacts, new: &SequenceFacts) -> bool {
    old.std_profile_idc == new.std_profile_idc
        && old.format == new.format
        && old.coded == new.coded
        && old.slots == new.slots
        && old.active_references == new.active_references
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::hevc_sps_full::tests as sps_tests;

    #[test]
    fn a_session_changes_with_what_it_was_made_for() {
        let facts = sequence_facts(&sps_tests::valid(&sps_tests::Spec::default())).expect("facts");
        assert!(same_session(&facts, &facts));
        let level = SequenceFacts {
            std_level: facts.std_level + 1,
            ..facts
        };
        assert!(same_session(&facts, &level), "a level needs no new session");
        let size = SequenceFacts {
            coded: (facts.coded.0 + 8, facts.coded.1),
            ..facts
        };
        assert!(!same_session(&facts, &size));
        let slots = SequenceFacts {
            slots: facts.slots + 1,
            ..facts
        };
        assert!(!same_session(&facts, &slots));
    }

    #[test]
    fn a_missing_reference_is_refused_by_name_with_the_next_action() {
        let text = list_refusal(
            ListError::Missing {
                poc: 12,
                long_term: true,
            },
            90,
        )
        .to_string();
        assert!(text.contains("long-term reference to POC 12"), "{text}");
        assert!(text.contains("pts 90"), "{text}");
        assert!(text.contains("another decoder"), "{text}");
    }
}
