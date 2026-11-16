//! A container crop applied after decode, such as Matroska's `PixelCrop*`.
//!
//! No decoder applies a container crop, so [`Cropped`] wraps one and crops
//! each frame it returns. The crop stays on the frame's device: a narrowed
//! view, then one device-to-device copy into dense planes.

mod cropped;
mod packed;
mod rect;
#[cfg(test)]
mod test_frames;
mod window;

pub use cropped::Cropped;
pub use rect::CropRect;
