//! The edges a crop removes.

use std::fmt;

use crate::geometry::Dimensions;

/// The pixels a crop removes from each edge of a decoded picture, as
/// Matroska's `PixelCropTop`, `PixelCropBottom`, `PixelCropLeft`, and
/// `PixelCropRight` state them.
///
/// A `W x H` picture keeps `W - left - right` columns from column `left`, and
/// `H - top - bottom` rows from row `top`. Odd values are exact. The 4:2:0
/// chroma plane starts at chroma row `top / 2` and chroma column `left / 2`,
/// rounded down, as FFmpeg's `av_frame_apply_cropping` does with
/// `AV_FRAME_CROP_UNALIGNED`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct CropRect {
    /// Rows removed at the top.
    pub top: u32,
    /// Rows removed at the bottom.
    pub bottom: u32,
    /// Columns removed at the left.
    pub left: u32,
    /// Columns removed at the right.
    pub right: u32,
}

impl CropRect {
    /// The size a `picture` keeps after the crop, or `None` when the crop
    /// leaves no row or no column of it.
    pub fn fit(self, picture: Dimensions) -> Option<Dimensions> {
        let width = picture
            .width()
            .checked_sub(self.left)?
            .checked_sub(self.right)?;
        let height = picture
            .height()
            .checked_sub(self.top)?
            .checked_sub(self.bottom)?;
        Dimensions::new(width, height).ok()
    }
}

impl fmt::Display for CropRect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "top {}, bottom {}, left {}, right {}",
            self.top, self.bottom, self.left, self.right
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_gives_the_kept_size() {
        let picture = Dimensions::new(64, 48).unwrap();
        let rect = |top, bottom, left, right| CropRect {
            top,
            bottom,
            left,
            right,
        };
        assert_eq!(
            rect(5, 1, 3, 2).fit(picture),
            Some(Dimensions::new(59, 42).unwrap())
        );
        assert_eq!(rect(0, 0, 32, 32).fit(picture), None);
        assert_eq!(rect(48, 0, 0, 0).fit(picture), None);
    }
}
