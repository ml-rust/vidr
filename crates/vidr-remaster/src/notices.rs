//! The licence and patent notices of what this build links.
//!
//! `notices/` holds each text unchanged. A binary distribution reproduces the text
//! of each codec it links. These notices state what a distribution needs, and they
//! are not legal advice.

/// One line on rav1r.
pub const RAV1R_NOTICE: &str = "rav1r is a fork of rav1e. Its code is BSD-2-Clause, under the \
                                Alliance for Open Media Patent License 1.0. --notices prints both.";
/// One line on dav1d.
pub const DAV1D_NOTICE: &str = "dav1d is BSD-2-Clause. AV1 decode falls under the Alliance for \
                                Open Media Patent License 1.0. --notices prints both.";
/// One line on OpenH264 compiled from source.
pub const OPENH264_NOTICE: &str = "OpenH264 is BSD-2-Clause. A source build gets no Cisco patent \
                                   cover. Whoever ships it answers for the H.264 patent licence.";
/// One line on Cisco's OpenH264 binary.
pub const OPENH264_CISCO_NOTICE: &str = "OpenH264 Video Codec provided by Cisco Systems, Inc. \
                                         --notices prints Cisco's binary licence.";
/// One line on HEVC.
pub const HEVC_NOTICE: &str = "HEVC has patent pools. Shipping a binary with cpu-hevc on needs \
                               legal review.";
/// One line on VP9.
pub const VP9_NOTICE: &str = "VP9 has a patent pool. Shipping a binary with cpu-vp9 on needs \
                              legal review.";

/// The one-line notices of the backend `name`, as `DecoderBackend::name` and
/// `EncoderBackend::name` state it. A hardware backend has none.
pub fn backend_notices(name: &str) -> Vec<&'static str> {
    match name {
        "rav1r" => vec![RAV1R_NOTICE],
        "dav1d" => vec![DAV1D_NOTICE],
        "openh264" => vec![OPENH264_NOTICE],
        "openh264-cisco" => vec![OPENH264_CISCO_NOTICE],
        "hevc" => vec![HEVC_NOTICE],
        "rusty_vp9" => vec![VP9_NOTICE],
        _ => Vec::new(),
    }
}

/// One section of the printed notices.
#[cfg_attr(
    not(any(
        feature = "dav1d",
        feature = "cpu-vp9",
        feature = "cpu-hevc",
        feature = "openh264",
        feature = "openh264-cisco",
        feature = "cpu-av1"
    )),
    allow(dead_code)
)]
fn push_section(out: &mut String, title: &str, summary: &str, texts: &[&str]) {
    out.push_str("\n\n== ");
    out.push_str(title);
    out.push_str(" ==\n");
    out.push_str(summary);
    for text in texts {
        out.push_str("\n\n");
        out.push_str(text.trim_end());
    }
}

/// The notices of every codec in this build, in full. A build with no third-party
/// codec prints the crate's own licence line alone.
pub fn notices() -> String {
    #[allow(
        unused_mut,
        reason = "a build with no third-party codec adds no section"
    )]
    let mut out = format!(
        "vidr-remaster {} is licensed under Apache-2.0. These notices state what a binary \
         distribution needs. They are not legal advice.",
        env!("CARGO_PKG_VERSION")
    );
    #[cfg(feature = "cpu-av1")]
    push_section(
        &mut out,
        "rav1r (cpu-av1)",
        RAV1R_NOTICE,
        &[
            include_str!("../notices/rav1r-NOTICE"),
            include_str!("../notices/rav1r-LICENSE"),
            include_str!("../notices/aom-patent-license-1.0"),
        ],
    );
    #[cfg(feature = "dav1d")]
    push_section(
        &mut out,
        "dav1d (dav1d)",
        DAV1D_NOTICE,
        &[
            include_str!("../notices/dav1d-COPYING"),
            include_str!("../notices/aom-patent-license-1.0"),
        ],
    );
    #[cfg(feature = "openh264")]
    push_section(
        &mut out,
        "OpenH264 (openh264)",
        OPENH264_NOTICE,
        &[include_str!("../notices/openh264-LICENSE")],
    );
    #[cfg(feature = "openh264-cisco")]
    push_section(
        &mut out,
        "OpenH264 binary (openh264-cisco)",
        OPENH264_CISCO_NOTICE,
        &[include_str!("../notices/openh264-cisco-BINARY_LICENSE")],
    );
    #[cfg(feature = "cpu-hevc")]
    push_section(&mut out, "HEVC (cpu-hevc)", HEVC_NOTICE, &[]);
    #[cfg(feature = "cpu-vp9")]
    push_section(
        &mut out,
        "rusty_vp9 (cpu-vp9)",
        VP9_NOTICE,
        &[include_str!("../notices/rusty_vp9-LICENSE")],
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_section_follows_its_feature() {
        let text = notices();
        assert!(text.starts_with("vidr-remaster "), "{text}");
        assert!(text.contains("Apache-2.0"), "{text}");
        assert_eq!(text.contains("== rav1r"), cfg!(feature = "cpu-av1"));
        assert_eq!(text.contains("== dav1d"), cfg!(feature = "dav1d"));
        assert_eq!(
            text.contains("== OpenH264 (openh264)"),
            cfg!(feature = "openh264")
        );
        assert_eq!(
            text.contains("== OpenH264 binary"),
            cfg!(feature = "openh264-cisco")
        );
        assert_eq!(text.contains("== HEVC"), cfg!(feature = "cpu-hevc"));
        assert_eq!(text.contains("== rusty_vp9"), cfg!(feature = "cpu-vp9"));
    }

    #[cfg(feature = "cpu-av1")]
    #[test]
    fn the_rav1r_section_names_its_origin_and_both_licences() {
        let text = notices();
        assert!(text.contains("derives from rav1e"), "{text}");
        assert!(text.contains("BSD 2-Clause License"), "{text}");
        assert!(
            text.contains("Alliance for Open Media Patent License 1.0"),
            "{text}"
        );
    }

    #[cfg(feature = "openh264-cisco")]
    #[test]
    fn the_cisco_section_holds_the_text_its_licence_requires() {
        let text = notices();
        assert!(
            text.contains("OpenH264 Video Codec provided by Cisco Systems, Inc."),
            "{text}"
        );
        assert!(
            text.contains("AVC/H.264 Patent Portfolio License"),
            "{text}"
        );
    }

    #[test]
    fn each_backend_with_terms_has_a_line_and_a_hardware_backend_has_none() {
        for name in [
            "rav1r",
            "dav1d",
            "openh264",
            "openh264-cisco",
            "hevc",
            "rusty_vp9",
        ] {
            assert_eq!(backend_notices(name).len(), 1, "{name}");
        }
        assert!(backend_notices("nvdec").is_empty());
        assert!(backend_notices("nvenc").is_empty());
    }
}
