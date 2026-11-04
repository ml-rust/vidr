//! Motion vector scaling by picture order count distance, shared by (8-179) to
//! (8-183), (8-193) to (8-197), and (8-205) to (8-209).

use super::types::Mv;

/// Scales `mv`. `td_diff` and `tb_diff` are the two `DiffPicOrderCnt` values before
/// their clip. A `td` of 0 occurs only in a corrupt stream, and returns `mv`.
pub(crate) fn scale_mv(mv: Mv, td_diff: i32, tb_diff: i32) -> Mv {
    let td = td_diff.clamp(-128, 127);
    let tb = tb_diff.clamp(-128, 127);
    if td == 0 {
        return mv;
    }
    // The "/" operator of 5.7 truncates toward zero, as Rust does.
    let tx = (16_384 + (td.abs() >> 1)) / td;
    let factor = ((tb * tx + 32) >> 6).clamp(-4096, 4095);
    Mv::new(scale_component(mv.x, factor), scale_component(mv.y, factor))
}

fn scale_component(value: i16, factor: i32) -> i16 {
    let product = factor * i32::from(value);
    let magnitude = (product.abs() + 127) >> 8;
    (product.signum() * magnitude).clamp(-32_768, 32_767) as i16
}

#[cfg(test)]
mod tests {
    use super::*;

    /// td 2, tb 1: tx = 16385 / 2 = 8192. Factor ( 8192 + 32 ) >> 6 = 128.
    /// 128 * 10 = 1280, and ( 1280 + 127 ) >> 8 = 5.
    #[test]
    fn half_the_distance_halves_the_vector() {
        assert_eq!(scale_mv(Mv::new(10, -10), 2, 1), Mv::new(5, -5));
    }

    /// td 1, tb 1: tx = 16384. Factor ( 16384 + 32 ) >> 6 = 256.
    /// 256 * 7 = 1792, and ( 1792 + 127 ) >> 8 = 7.
    #[test]
    fn an_equal_distance_keeps_the_vector() {
        assert_eq!(scale_mv(Mv::new(7, -3), 1, 1), Mv::new(7, -3));
    }

    /// td -1, tb 1: tx = -16384. Factor ( -16384 + 32 ) >> 6 = -256.
    /// -256 * 4 = -1024, and ( 1024 + 127 ) >> 8 = 4, with a negative sign.
    #[test]
    fn opposite_directions_flip_the_sign() {
        assert_eq!(scale_mv(Mv::new(4, -4), -1, 1), Mv::new(-4, 4));
    }

    /// td 3, tb 1: tx = 16385 / 3 = 5461. Factor ( 5461 + 32 ) >> 6 = 85.
    /// 85 * 9 = 765, and ( 765 + 127 ) >> 8 = 3.
    #[test]
    fn a_third_of_the_distance_follows_the_rounding() {
        assert_eq!(scale_mv(Mv::new(9, 0), 3, 1), Mv::new(3, 0));
    }

    /// td 1, tb 127: tb * tx + 32 is 2080800, and >> 6 gives 32512. The clip gives
    /// 4095. 4095 * 32767 = 134180865. ( 134180865 + 127 ) >> 8 = 524144. The clip
    /// gives 32767.
    #[test]
    fn the_factor_and_the_result_clip() {
        assert_eq!(scale_mv(Mv::new(32_767, 1), 1, 127), Mv::new(32_767, 16));
    }

    /// The distances clip to -128 and 127 first. td 1000 acts as 127: tx = 16447 /
    /// 127 = 129. tb -1000 acts as -128: ( -128 * 129 + 32 ) >> 6 = -258.
    /// -258 * 100 = -25800, and ( 25800 + 127 ) >> 8 = 101, with a negative sign.
    #[test]
    fn the_distances_clip_first() {
        assert_eq!(scale_mv(Mv::new(100, 0), 1000, -1000), Mv::new(-101, 0));
    }

    #[test]
    fn a_zero_distance_keeps_the_vector() {
        assert_eq!(scale_mv(Mv::new(5, 6), 0, 3), Mv::new(5, 6));
    }
}
