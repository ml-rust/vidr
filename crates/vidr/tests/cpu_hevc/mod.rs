//! `HevcDecoder` on numr's CPU backend, against ffmpeg's decode.
//!
//! Every test skips with a notice when its fixture is missing. Run
//! `python3 crates/vidr/tests/fixtures/gen_nvdec.py` and
//! `python3 crates/vidr/tests/fixtures/gen_hevc.py` to write them.

mod decode;
pub(crate) mod packets;
