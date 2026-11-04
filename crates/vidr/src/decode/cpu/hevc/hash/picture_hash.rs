//! The hash methods of D.3.20 and the SEI message that carries their values.

use crate::decode::cpu::dsp::sample::Sample;
use crate::decode::cpu::hevc::error::HevcError;
use crate::decode::cpu::hevc::picture::{Picture, Plane};

use super::md5::Md5;

/// `payloadType` of the decoded picture hash SEI message (D.2.1).
const PAYLOAD_DECODED_PICTURE_HASH: u32 = 132;
const HASH_MD5: u8 = 0;
const HASH_CRC: u8 = 1;
const HASH_CHECKSUM: u8 = 2;
/// Bytes after the NAL unit header of an SEI NAL unit, before the first message.
const NAL_HEADER_BYTES: usize = 2;

/// The values in one decoded picture hash SEI message, by `cIdx`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PictureHash {
    /// `picture_md5`.
    Md5([[u8; 16]; 3]),
    /// `picture_crc`.
    Crc([u16; 3]),
    /// `picture_checksum`.
    Checksum([u32; 3]),
}

/// Finds the decoded picture hash in an SEI NAL unit without emulation prevention
/// bytes. A reserved `hash_type` returns `None`, as D.3.20 tells decoders to ignore it.
///
/// # Errors
///
/// [`HevcError::InvalidSyntax`] for a message that runs past the NAL unit, or a hash
/// payload too short for three planes.
pub(crate) fn find_picture_hash(rbsp: &[u8]) -> Result<Option<PictureHash>, HevcError> {
    let mut rest = rbsp.get(NAL_HEADER_BYTES..).unwrap_or_default();
    // The last byte holds rbsp_trailing_bits, so a message needs two more.
    while rest.len() > 2 {
        let (payload_type, after_type) = read_extended_value(rest)?;
        let (payload_size, after_size) = read_extended_value(after_type)?;
        let size = payload_size as usize;
        let Some(payload) = after_size.get(..size) else {
            return Err(HevcError::InvalidSyntax {
                element: "payloadSize",
                value: i64::from(payload_size),
                allowed: "a size that fits the SEI NAL unit",
            });
        };
        if payload_type == PAYLOAD_DECODED_PICTURE_HASH {
            return parse_hash_payload(payload);
        }
        rest = after_size.get(size..).unwrap_or_default();
    }
    Ok(None)
}

/// `payloadType` or `payloadSize`: bytes of 255 count up, and the first other byte
/// ends the value (7.3.5).
fn read_extended_value(bytes: &[u8]) -> Result<(u32, &[u8]), HevcError> {
    let mut value = 0u32;
    for (index, &byte) in bytes.iter().enumerate() {
        value = value.saturating_add(u32::from(byte));
        if byte != 0xff {
            return Ok((value, bytes.get(index + 1..).unwrap_or_default()));
        }
    }
    Err(HevcError::InvalidSyntax {
        element: "sei_message",
        value: i64::from(value),
        allowed: "a message that ends inside the SEI NAL unit",
    })
}

/// `decoded_picture_hash()` (D.2.20) with `chroma_format_idc` 1.
fn parse_hash_payload(payload: &[u8]) -> Result<Option<PictureHash>, HevcError> {
    let Some((&hash_type, values)) = payload.split_first() else {
        return Err(short_payload(0));
    };
    let hash = match hash_type {
        HASH_MD5 => {
            let (chunks, _) = values.as_chunks::<16>();
            let [a, b, c, ..] = chunks else {
                return Err(short_payload(payload.len()));
            };
            PictureHash::Md5([*a, *b, *c])
        }
        HASH_CRC => {
            let (chunks, _) = values.as_chunks::<2>();
            let [a, b, c, ..] = chunks else {
                return Err(short_payload(payload.len()));
            };
            PictureHash::Crc([*a, *b, *c].map(u16::from_be_bytes))
        }
        HASH_CHECKSUM => {
            let (chunks, _) = values.as_chunks::<4>();
            let [a, b, c, ..] = chunks else {
                return Err(short_payload(payload.len()));
            };
            PictureHash::Checksum([*a, *b, *c].map(u32::from_be_bytes))
        }
        _ => return Ok(None),
    };
    Ok(Some(hash))
}

const fn short_payload(size: usize) -> HevcError {
    HevcError::InvalidSyntax {
        element: "payloadSize of the decoded picture hash",
        value: size as i64,
        allowed: "a size that holds hash_type and the values of three planes",
    }
}

/// Compares `picture` with the values of `hash`, plane by plane.
///
/// # Errors
///
/// - [`HevcError::PictureHashMismatch`] names the method and the first plane that
///   differs.
/// - [`HevcError::Allocation`] when the working buffer does not fit in memory.
pub(crate) fn check_picture_hash<S: Sample>(
    picture: &Picture<S>,
    hash: &PictureHash,
) -> Result<(), HevcError> {
    let mut scratch = Vec::new();
    for c_idx in 0..3 {
        let plane = picture.plane(c_idx);
        let (method, expected, computed) = match hash {
            PictureHash::Md5(values) => (
                "MD5",
                u128::from_be_bytes(values[c_idx]),
                u128::from_be_bytes(plane_md5(plane, &mut scratch)?),
            ),
            PictureHash::Crc(values) => (
                "CRC",
                u128::from(values[c_idx]),
                u128::from(plane_crc(plane, &mut scratch)?),
            ),
            PictureHash::Checksum(values) => (
                "checksum",
                u128::from(values[c_idx]),
                u128::from(plane_checksum(plane)),
            ),
        };
        if expected != computed {
            return Err(HevcError::PictureHashMismatch {
                method,
                c_idx,
                expected,
                computed,
            });
        }
    }
    Ok(())
}

/// `pictureData` of one row (D-22) goes through `bytes`.
fn for_each_row<S: Sample>(
    plane: &Plane<S>,
    scratch: &mut Vec<u8>,
    mut bytes: impl FnMut(&[u8]),
) -> Result<(), HevcError> {
    for y in 0..plane.height() {
        let row = plane.row(y).unwrap_or_default();
        let data = S::host_bytes(row, scratch).map_err(|_| HevcError::Allocation {
            what: "picture hash bytes",
            count: row.len() * S::BYTES,
        })?;
        bytes(data);
    }
    Ok(())
}

/// (D-23).
fn plane_md5<S: Sample>(plane: &Plane<S>, scratch: &mut Vec<u8>) -> Result<[u8; 16], HevcError> {
    let mut md5 = Md5::new();
    for_each_row(plane, scratch, |bytes| md5.update(bytes))?;
    Ok(md5.finish())
}

/// One byte through the loop of (D-24).
fn crc_byte(mut crc: u16, byte: u8) -> u16 {
    for bit in (0..8).rev() {
        let msb = crc >> 15;
        let value = u16::from((byte >> bit) & 1);
        crc = ((crc << 1) | value) ^ (msb * 0x1021);
    }
    crc
}

/// (D-24).
fn plane_crc<S: Sample>(plane: &Plane<S>, scratch: &mut Vec<u8>) -> Result<u16, HevcError> {
    let mut crc = 0xffffu16;
    for_each_row(plane, scratch, |bytes| {
        for &byte in bytes {
            crc = crc_byte(crc, byte);
        }
    })?;
    // The two zero bytes that (D-24) appends.
    Ok(crc_byte(crc_byte(crc, 0), 0))
}

/// (D-25).
fn plane_checksum<S: Sample>(plane: &Plane<S>) -> u32 {
    let mut sum = 0u32;
    for y in 0..plane.height() {
        let row = plane.row(y).unwrap_or_default();
        for (x, sample) in row.iter().enumerate() {
            let mask = ((x & 0xff) ^ (y & 0xff) ^ (x >> 8) ^ (y >> 8)) as u32;
            let value = sample.to_i32() as u32;
            sum = sum.wrapping_add((value & 0xff) ^ mask);
            if S::BYTES > 1 {
                sum = sum.wrapping_add((value >> 8) ^ mask);
            }
        }
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sei(payload_type: u8, payload: &[u8]) -> Vec<u8> {
        let mut bytes = vec![0x50, 0x01, payload_type, payload.len() as u8];
        bytes.extend_from_slice(payload);
        bytes.push(0x80);
        bytes
    }

    #[test]
    fn each_hash_type_parses() {
        let mut md5 = vec![0u8];
        md5.extend(0..48);
        let expected =
            |plane: usize| -> [u8; 16] { std::array::from_fn(|index| (plane * 16 + index) as u8) };
        assert_eq!(
            find_picture_hash(&sei(132, &md5)),
            Ok(Some(PictureHash::Md5([
                expected(0),
                expected(1),
                expected(2)
            ])))
        );
        assert_eq!(
            find_picture_hash(&sei(132, &[1, 0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc])),
            Ok(Some(PictureHash::Crc([0x1234, 0x5678, 0x9abc])))
        );
        assert_eq!(
            find_picture_hash(&sei(132, &[2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])),
            Ok(Some(PictureHash::Checksum([
                0x0102_0304,
                0x0506_0708,
                0x090a_0b0c
            ])))
        );
    }

    /// D.3.20: a decoder ignores a reserved `hash_type`.
    #[test]
    fn a_reserved_type_is_ignored() {
        assert_eq!(find_picture_hash(&sei(132, &[3, 0, 0, 0])), Ok(None));
    }

    #[test]
    fn other_messages_are_skipped() {
        let mut bytes = vec![0x50, 0x01, 5, 2, 9, 9, 132, 7];
        bytes.extend_from_slice(&[1, 0, 1, 0, 2, 0, 3]);
        bytes.push(0x80);
        assert_eq!(
            find_picture_hash(&bytes),
            Ok(Some(PictureHash::Crc([1, 2, 3])))
        );
        assert_eq!(find_picture_hash(&sei(5, &[1, 2])), Ok(None));
        assert_eq!(find_picture_hash(&[0x50, 0x01, 0x80]), Ok(None));
    }

    #[test]
    fn truncated_messages_are_refused() {
        assert!(matches!(
            find_picture_hash(&[0x50, 0x01, 132, 200, 1, 2]),
            Err(HevcError::InvalidSyntax {
                element: "payloadSize",
                value: 200,
                ..
            })
        ));
        assert!(matches!(
            find_picture_hash(&sei(132, &[1, 0, 0])),
            Err(HevcError::InvalidSyntax { .. })
        ));
        assert!(matches!(
            find_picture_hash(&[0x50, 0x01, 0xff, 0xff, 0xff]),
            Err(HevcError::InvalidSyntax {
                element: "sei_message",
                ..
            })
        ));
    }

    /// The byte form of the loop against (D-24) as printed.
    #[test]
    fn crc_follows_d_24() {
        let mut crc = 0xffffu16;
        for _ in 0..3 {
            crc = crc_byte(crc, 0);
        }
        assert_eq!(crc, reference_crc(&[0, 0, 0]));
        assert_eq!(
            crc_byte(crc_byte(0xffff, 0x31), 0),
            reference_crc(&[0x31, 0])
        );
    }

    /// (D-24) written one bit at a time as the specification prints it.
    fn reference_crc(data: &[u8]) -> u16 {
        let mut crc: u32 = 0xffff;
        for bit_idx in 0..data.len() * 8 {
            let byte = u32::from(data[bit_idx >> 3]);
            let msb = (crc >> 15) & 1;
            let value = (byte >> (7 - (bit_idx & 7))) & 1;
            crc = (((crc << 1) + value) & 0xffff) ^ (msb * 0x1021);
        }
        crc as u16
    }

    fn small_picture() -> Picture<u8> {
        let mut picture = Picture::<u8>::new(8, 8).unwrap();
        for c_idx in 0..3 {
            let plane = picture.plane_mut(c_idx);
            for y in 0..plane.height() {
                for (x, sample) in plane.row_mut(y).unwrap().iter_mut().enumerate() {
                    *sample = (c_idx * 50 + y * 8 + x) as u8;
                }
            }
        }
        picture
    }

    /// (D-25) by hand: x and y below 256 give a mask of x ^ y.
    #[test]
    fn checksum_follows_d_25() {
        let picture = small_picture();
        let mut expected = 0u32;
        for y in 0..8u32 {
            for x in 0..8u32 {
                expected += (y * 8 + x) ^ (x ^ y);
            }
        }
        assert_eq!(plane_checksum(picture.luma()), expected);
    }

    /// 10-bit samples add the high byte under the same mask.
    #[test]
    fn checksum_adds_the_high_byte_above_8_bits() {
        let mut picture = Picture::<u16>::new(8, 8).unwrap();
        picture.luma_mut().row_mut(0).unwrap()[3] = 0x2f1;
        let sum = plane_checksum(picture.luma());
        // Every other sample is 0, and adds its mask twice. At x = 3, y = 0, the mask
        // is 3: ( 0xf1 ^ 3 ) + ( 2 ^ 3 ).
        let mut expected = 0u32;
        for y in 0..8u32 {
            for x in 0..8u32 {
                expected += 2 * (x ^ y);
            }
        }
        expected = expected - 2 * 3 + (0xf1 ^ 3) + (2 ^ 3);
        assert_eq!(sum, expected);
    }

    #[test]
    fn matching_values_pass_and_a_changed_plane_names_itself() {
        let picture = small_picture();
        let mut scratch = Vec::new();
        let md5 = [
            plane_md5(picture.luma(), &mut scratch).unwrap(),
            plane_md5(picture.cb(), &mut scratch).unwrap(),
            plane_md5(picture.cr(), &mut scratch).unwrap(),
        ];
        assert_eq!(check_picture_hash(&picture, &PictureHash::Md5(md5)), Ok(()));
        let mut wrong = md5;
        wrong[1][15] ^= 1;
        assert!(matches!(
            check_picture_hash(&picture, &PictureHash::Md5(wrong)),
            Err(HevcError::PictureHashMismatch {
                method: "MD5",
                c_idx: 1,
                ..
            })
        ));
        let crc = [
            plane_crc(picture.luma(), &mut scratch).unwrap(),
            plane_crc(picture.cb(), &mut scratch).unwrap(),
            plane_crc(picture.cr(), &mut scratch).unwrap(),
        ];
        assert_eq!(check_picture_hash(&picture, &PictureHash::Crc(crc)), Ok(()));
        assert!(matches!(
            check_picture_hash(&picture, &PictureHash::Crc([0, crc[1], crc[2]])),
            Err(HevcError::PictureHashMismatch {
                method: "CRC",
                c_idx: 0,
                ..
            })
        ));
        let checksum = [
            plane_checksum(picture.luma()),
            plane_checksum(picture.cb()),
            plane_checksum(picture.cr()),
        ];
        assert_eq!(
            check_picture_hash(&picture, &PictureHash::Checksum(checksum)),
            Ok(())
        );
        assert!(matches!(
            check_picture_hash(
                &picture,
                &PictureHash::Checksum([checksum[0], checksum[1], !checksum[2]])
            ),
            Err(HevcError::PictureHashMismatch {
                method: "checksum",
                c_idx: 2,
                ..
            })
        ));
    }
}
