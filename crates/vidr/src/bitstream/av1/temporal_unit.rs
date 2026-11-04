//! A packet of low-overhead OBUs into frames: sequence headers, frame headers, tile groups, and
//! the operating point (AV1 specification 5.3.1, 5.9.1, and 7.5). It holds no decoding state past
//! what the header syntax reads.

use std::sync::Arc;

use super::frame_header::{FrameHeader, RefTable, parse_frame_header};
use super::obu::{self, Obu, ObuHeader, ObuType};
use super::seq_header::{SequenceHeader, parse_sequence_header};
use super::tile_group::{TileGroupFacts, TileRange, parse_tile_group};
use crate::error::BitstreamError;

const STRUCTURE: &str = "AV1 temporal unit";

/// One frame of a packet.
#[derive(Clone, Debug)]
pub(crate) struct ParsedFrame<'a> {
    pub(crate) sequence: Arc<SequenceHeader>,
    pub(crate) header: FrameHeader,
    /// Which temporal unit of the packet holds the frame.
    pub(crate) unit: u32,
    /// The frame's OBUs in stream order, whole. A frame shown again holds its one header.
    pub(crate) parts: Vec<&'a [u8]>,
    /// Each tile, by offset into the joined `parts`.
    pub(crate) tiles: Vec<TileRange>,
    /// Each tile group, in stream order.
    #[cfg_attr(
        not(test),
        allow(dead_code, reason = "the header oracle alone reads it")
    )]
    pub(crate) groups: Vec<TileGroupFacts>,
}

/// A frame whose tile groups still come.
struct Pending<'a> {
    header: FrameHeader,
    /// The header OBU's payload, whose header bits a copy must equal.
    header_payload: &'a [u8],
    parts: Vec<&'a [u8]>,
    bytes: u32,
    tiles: Vec<TileRange>,
    groups: Vec<TileGroupFacts>,
}

fn invalid(field: &'static str, value: u64, allowed: &'static str) -> BitstreamError {
    BitstreamError::InvalidField {
        structure: STRUCTURE,
        field,
        value,
        allowed,
    }
}

/// Whether `packet` reads as one Annex B temporal unit (AV1 specification Annex B). The test reads
/// a `temporal_unit_size` that fills the packet. A `frame_unit_size` and an `obu_length` that fit
/// follow it, then a temporal delimiter OBU. A low-overhead packet starts with its OBU header, so
/// it passes only with these exact sizes.
pub(crate) fn looks_like_annex_b(packet: &[u8]) -> bool {
    let Ok((unit, first)) = obu::read_leb128(packet, 0, "temporal_unit_size") else {
        return false;
    };
    if unit != (packet.len() - first) as u64 {
        return false;
    }
    let Ok((frame, second)) = obu::read_leb128(packet, first, "frame_unit_size") else {
        return false;
    };
    if frame == 0 || frame > unit.saturating_sub(second as u64) {
        return false;
    }
    let Ok((length, third)) = obu::read_leb128(packet, first + second, "obu_length") else {
        return false;
    };
    if !(1..=3).contains(&length) || length > frame.saturating_sub(third as u64) {
        return false;
    }
    packet.get(first + second + third).is_some_and(|&byte| {
        byte & 0x80 == 0 && (byte >> 3) & 0x0f == ObuType::TEMPORAL_DELIMITER.value()
    })
}

/// The stream state that crosses packets.
#[derive(Clone, Debug, Default)]
pub(crate) struct Av1Parser {
    sequence: Option<Arc<SequenceHeader>>,
    refs: RefTable,
}

impl Av1Parser {
    /// Takes a sequence header OBU payload. A header of another coded video sequence replaces the
    /// current one. It also clears the reference names, as section 7.5 requires.
    ///
    /// # Errors
    ///
    /// As [`parse_sequence_header`].
    pub(crate) fn set_sequence(&mut self, payload: &[u8]) -> Result<(), BitstreamError> {
        let parsed = parse_sequence_header(payload)?;
        let same = self
            .sequence
            .as_deref()
            .is_some_and(|current| current.same_sequence(&parsed));
        if !same {
            self.sequence = Some(Arc::new(parsed));
            self.refs = RefTable::default();
        }
        Ok(())
    }

    /// True when `payload` states the current sequence again.
    fn repeats_sequence(&self, payload: &[u8]) -> Result<bool, BitstreamError> {
        let parsed = parse_sequence_header(payload)?;
        Ok(self
            .sequence
            .as_deref()
            .is_some_and(|current| current.same_sequence(&parsed)))
    }

    /// The current sequence header.
    #[cfg(test)]
    pub(crate) const fn sequence(&self) -> Option<&Arc<SequenceHeader>> {
        self.sequence.as_ref()
    }

    /// The saved state of each reference name.
    #[cfg(test)]
    pub(crate) const fn refs(&self) -> &RefTable {
        &self.refs
    }

    /// Parses a packet of low-overhead OBUs into its frames, in decode order, and runs the
    /// reference update of each one.
    ///
    /// # Errors
    ///
    /// [`BitstreamError`] for each of these:
    ///
    /// - an OBU that does not parse
    /// - a frame header with no sequence header
    /// - a tile group with no frame header
    /// - a new sequence header inside an open frame
    /// - a frame header OBU inside an open frame, or a redundant one that is no exact copy
    /// - a frame that does not finish in the packet
    /// - a tile list OBU, which large-scale tile coding alone uses
    pub(crate) fn parse_packet<'a>(
        &mut self,
        packet: &'a [u8],
    ) -> Result<Vec<ParsedFrame<'a>>, BitstreamError> {
        let mut frames = Vec::new();
        let mut pending: Option<Pending<'a>> = None;
        let mut unit = 0;
        let mut seen_delimiter = false;
        for parsed in obu::split(packet) {
            let parsed = parsed?;
            let kind = parsed.header.obu_type;
            if self.dropped(&parsed.header) {
                continue;
            }
            match kind {
                ObuType::SEQUENCE_HEADER => {
                    // A repeat changes nothing, so only a new sequence stops an open frame.
                    if pending.is_some() && !self.repeats_sequence(parsed.payload)? {
                        return Err(invalid(
                            "obu_type",
                            kind.value().into(),
                            "a new sequence header only before a frame header or after a frame \
                             ends",
                        ));
                    }
                    self.set_sequence(parsed.payload)?;
                }
                ObuType::TEMPORAL_DELIMITER => {
                    if let Some(open) = &pending {
                        return Err(unfinished(open));
                    }
                    if seen_delimiter {
                        unit += 1;
                    }
                    seen_delimiter = true;
                }
                ObuType::FRAME_HEADER | ObuType::REDUNDANT_FRAME_HEADER => {
                    if let Some(open) = &mut pending {
                        // Section 6.8.1: a frame header OBU needs SeenFrameHeader 0.
                        if kind == ObuType::FRAME_HEADER {
                            return Err(invalid(
                                "obu_type",
                                kind.value().into(),
                                "a frame header only after the last frame finished. A copy of it \
                                 takes OBU_REDUNDANT_FRAME_HEADER",
                            ));
                        }
                        check_copy(open, parsed.payload)?;
                        open.bytes = add_part(&mut open.parts, open.bytes, parsed.bytes)?;
                    } else if kind == ObuType::REDUNDANT_FRAME_HEADER {
                        return Err(invalid(
                            "obu_type",
                            kind.value().into(),
                            "a redundant frame header only after its frame header",
                        ));
                    } else {
                        pending = self.begin_frame(&parsed, unit, &mut frames)?;
                    }
                }
                ObuType::FRAME => {
                    if pending.is_some() {
                        return Err(invalid(
                            "obu_type",
                            kind.value().into(),
                            "a frame OBU only after the last frame finished",
                        ));
                    }
                    let open = self
                        .begin_frame(&parsed, unit, &mut frames)?
                        .ok_or_else(|| {
                            invalid(
                                "show_existing_frame",
                                1,
                                "0 in a frame OBU, as section 6.10.1 states",
                            )
                        })?;
                    let header_bytes = open.header.header_bytes();
                    let body =
                        parsed
                            .payload
                            .get(header_bytes..)
                            .ok_or(BitstreamError::Truncated {
                                structure: STRUCTURE,
                                field: "tile_group_obu",
                                offset: header_bytes,
                                size: parsed.payload.len(),
                            })?;
                    pending = self.add_tile_group(open, &parsed, body, unit, &mut frames)?;
                }
                ObuType::TILE_GROUP => {
                    let Some(open) = pending.take() else {
                        return Err(invalid(
                            "obu_type",
                            kind.value().into(),
                            "a tile group only after its frame header",
                        ));
                    };
                    pending =
                        self.add_tile_group(open, &parsed, parsed.payload, unit, &mut frames)?;
                }
                ObuType::TILE_LIST => {
                    return Err(invalid(
                        "obu_type",
                        kind.value().into(),
                        "no tile list. Large-scale tile coding has no decode here",
                    ));
                }
                // Metadata, padding, and reserved types hold nothing the frames read.
                _ => {}
            }
        }
        if let Some(open) = &pending {
            return Err(unfinished(open));
        }
        Ok(frames)
    }

    /// Whether the selected operating point drops this OBU.
    fn dropped(&self, header: &ObuHeader) -> bool {
        let kind = header.obu_type;
        if kind == ObuType::SEQUENCE_HEADER || kind == ObuType::TEMPORAL_DELIMITER {
            return false;
        }
        let idc = self
            .sequence
            .as_deref()
            .map_or(0, SequenceHeader::operating_point_idc);
        let Some(layer) = header.extension else {
            return false;
        };
        if idc == 0 {
            return false;
        }
        let in_temporal = (idc >> layer.temporal_id) & 1 == 1;
        let in_spatial = (idc >> (u16::from(layer.spatial_id) + 8)) & 1 == 1;
        !(in_temporal && in_spatial)
    }

    /// Parses a new frame header. A frame shown again finishes here, so the result is `None`.
    fn begin_frame<'a>(
        &mut self,
        parsed: &Obu<'a>,
        unit: u32,
        frames: &mut Vec<ParsedFrame<'a>>,
    ) -> Result<Option<Pending<'a>>, BitstreamError> {
        let Some(sequence) = self.sequence.clone() else {
            return Err(invalid(
                "sequence_header_obu",
                0,
                "present before the first frame header",
            ));
        };
        let header = parse_frame_header(
            &sequence,
            parsed.header.extension,
            parsed.payload,
            &mut self.refs,
        )?;
        if header.show_existing_frame {
            self.refs.update(&header);
            frames.push(ParsedFrame {
                sequence,
                header,
                unit,
                parts: vec![parsed.bytes],
                tiles: Vec::new(),
                groups: Vec::new(),
            });
            return Ok(None);
        }
        let mut parts = Vec::with_capacity(1);
        let bytes = add_part(&mut parts, 0, parsed.bytes)?;
        Ok(Some(Pending {
            header,
            header_payload: parsed.payload,
            parts,
            bytes,
            tiles: Vec::new(),
            groups: Vec::new(),
        }))
    }

    /// Adds one tile group to `open`. A frame with every tile goes to `frames`, and otherwise the
    /// result holds it. An OBU_FRAME's bytes are already in `open.parts`.
    fn add_tile_group<'a>(
        &mut self,
        mut open: Pending<'a>,
        parsed: &Obu<'a>,
        body: &'a [u8],
        unit: u32,
        frames: &mut Vec<ParsedFrame<'a>>,
    ) -> Result<Option<Pending<'a>>, BitstreamError> {
        let whole_frame = parsed.header.obu_type == ObuType::FRAME;
        let start = if whole_frame { 0 } else { open.bytes };
        // `body` ends where the OBU ends, so its offset comes from their lengths.
        let base = start.saturating_add((parsed.bytes.len() - body.len()) as u32);
        let next_tile = open.groups.last().map_or(0, |group| group.tg_end + 1);
        let facts = parse_tile_group(
            body,
            base,
            &open.header.tile_info,
            next_tile,
            &mut open.tiles,
        )?;
        open.groups.push(facts);
        if !whole_frame {
            open.bytes = add_part(&mut open.parts, open.bytes, parsed.bytes)?;
        }
        if facts.tg_end + 1 < open.header.tile_info.num_tiles() {
            return Ok(Some(open));
        }
        let sequence = self.sequence.clone().ok_or_else(|| {
            invalid(
                "sequence_header_obu",
                0,
                "present before the first frame header",
            )
        })?;
        self.refs.update(&open.header);
        frames.push(ParsedFrame {
            sequence,
            header: open.header,
            unit,
            parts: open.parts,
            tiles: open.tiles,
            groups: open.groups,
        });
        Ok(None)
    }
}

fn add_part<'a>(
    parts: &mut Vec<&'a [u8]>,
    bytes: u32,
    part: &'a [u8],
) -> Result<u32, BitstreamError> {
    let total = u32::try_from(part.len())
        .ok()
        .and_then(|len| bytes.checked_add(len))
        .ok_or_else(|| invalid("obu_size", part.len() as u64, "a frame of less than 4 GiB"))?;
    parts.push(part);
    Ok(total)
}

/// `frame_header_copy()`: a copy holds the header's bits, then `trailing_bits()`: one 1 bit and 0
/// bits to the end of the OBU, as sections 5.9.1 and 5.3.4 state. The original's bits past its
/// header go unread. An OBU_FRAME holds `byte_alignment()` and tiles there.
fn is_header_copy(original: &[u8], copy: &[u8], header_bits: u64) -> bool {
    let whole = (header_bits / 8) as usize;
    let rest = (header_bits % 8) as u32;
    let (Some(original_head), Some(copy_head), Some(&copy_last)) =
        (original.get(..whole), copy.get(..whole), copy.get(whole))
    else {
        return false;
    };
    let header_mask = !(0xff_u8 >> rest);
    let original_last = original.get(whole).copied().unwrap_or(0);
    original_head == copy_head
        && original_last & header_mask == copy_last & header_mask
        && copy_last & !header_mask == 0x80 >> rest
        && copy.iter().skip(whole + 1).all(|&byte| byte == 0)
}

fn check_copy(open: &Pending<'_>, payload: &[u8]) -> Result<(), BitstreamError> {
    if is_header_copy(open.header_payload, payload, open.header.header_bits) {
        return Ok(());
    }
    Err(invalid(
        "frame_header_copy",
        payload.len() as u64,
        "the length of a payload with the frame header's exact bits, then trailing bits",
    ))
}

fn unfinished(open: &Pending<'_>) -> BitstreamError {
    BitstreamError::Truncated {
        structure: STRUCTURE,
        field: "tile_group_obu",
        offset: open.bytes as usize,
        size: open.bytes as usize,
    }
}

#[cfg(test)]
#[path = "temporal_unit_tests.rs"]
mod tests;
