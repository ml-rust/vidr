//! An Annex-B stream as `HevcDecoder` packets.

use vidr::BitstreamError;
use vidr::bitstream::{HevcNalType, StartCode, annexb};

/// True for a slice segment of the base layer that starts a picture.
fn starts_picture(nal: &[u8]) -> bool {
    let [first, second, third, ..] = nal else {
        return false;
    };
    let base_layer = first & 1 == 0 && second >> 3 == 0;
    let slice = matches!(HevcNalType::from_header(*first).value(), 0..=9 | 16..=21);
    base_layer && slice && third & 0x80 != 0
}

/// One packet per NAL unit, with its start code. The pts counts the pictures that
/// start up to and including that unit, in decode order.
pub(crate) fn nal_packets(
    stream: &[u8],
) -> Result<impl Iterator<Item = (Vec<u8>, i64)> + '_, BitstreamError> {
    let mut pictures = 0;
    let units = annexb::split(stream)?;
    Ok(units.map(move |nal| {
        pictures += i64::from(starts_picture(nal));
        let mut packet = Vec::with_capacity(nal.len() + 4);
        packet.extend_from_slice(StartCode::Four.bytes());
        packet.extend_from_slice(nal);
        (packet, pictures)
    }))
}
