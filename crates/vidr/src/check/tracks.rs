//! The copied-track check: the output holds the streams and chapters the copy
//! plan says it holds, and each copied track keeps its packet count.

use super::deviation::{CheckError, Deviation, TrackClass};

/// One track's packet count.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TrackPacketCount {
    /// The track id in its own file.
    pub track_id: u32,
    /// The packets. A text subtitle's count leaves out empty samples, which MP4
    /// writes for the gaps between cues and Matroska does not.
    pub packets: u64,
}

/// The copied streams of one file, each list in file order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TrackFacts {
    /// Audio tracks.
    pub audio: Vec<TrackPacketCount>,
    /// Subtitle tracks.
    pub subtitles: Vec<TrackPacketCount>,
    /// Chapters.
    pub chapters: usize,
}

/// Checks `output` against `source`, the facts of what the output must hold.
///
/// `source` comes from `CopyPlan::expected_tracks`, which leaves out the tracks
/// the plan drops. A copy keeps track order, so tracks pair by position within
/// each kind. A mismatch is an error, never a warning.
///
/// # Errors
///
/// [`CheckError`] holding one [`Deviation`] per mismatch, in this order: audio
/// tracks, subtitle tracks, chapters. A track count mismatch also reports the
/// packet counts of the tracks both files hold.
pub fn check_tracks(source: &TrackFacts, output: &TrackFacts) -> Result<(), CheckError> {
    let mut deviations = Vec::new();
    compare(
        TrackClass::Audio,
        &source.audio,
        &output.audio,
        &mut deviations,
    );
    compare(
        TrackClass::Subtitle,
        &source.subtitles,
        &output.subtitles,
        &mut deviations,
    );
    if source.chapters != output.chapters {
        deviations.push(Deviation::ChapterCount {
            source: source.chapters,
            output: output.chapters,
        });
    }
    if deviations.is_empty() {
        Ok(())
    } else {
        Err(CheckError::new(deviations))
    }
}

fn compare(
    class: TrackClass,
    source: &[TrackPacketCount],
    output: &[TrackPacketCount],
    deviations: &mut Vec<Deviation>,
) {
    if source.len() != output.len() {
        deviations.push(Deviation::TrackCount {
            class,
            source: source.len(),
            output: output.len(),
        });
    }
    for (from, to) in source.iter().zip(output) {
        if from.packets != to.packets {
            deviations.push(Deviation::PacketCount {
                class,
                source_track: from.track_id,
                output_track: to.track_id,
                source: from.packets,
                output: to.packets,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn count(track_id: u32, packets: u64) -> TrackPacketCount {
        TrackPacketCount { track_id, packets }
    }

    fn facts() -> TrackFacts {
        TrackFacts {
            audio: vec![count(2, 94), count(3, 50)],
            subtitles: vec![count(4, 2)],
            chapters: 2,
        }
    }

    #[test]
    fn equal_counts_pass_whatever_the_track_ids() {
        let mut output = facts();
        output.audio[0].track_id = 7;
        assert_eq!(check_tracks(&facts(), &output), Ok(()));
    }

    #[test]
    fn every_mismatch_is_reported() {
        let output = TrackFacts {
            audio: vec![count(2, 93)],
            subtitles: Vec::new(),
            chapters: 1,
        };
        let error = check_tracks(&facts(), &output).unwrap_err();
        assert_eq!(
            error.deviations(),
            [
                Deviation::TrackCount {
                    class: TrackClass::Audio,
                    source: 2,
                    output: 1,
                },
                Deviation::PacketCount {
                    class: TrackClass::Audio,
                    source_track: 2,
                    output_track: 2,
                    source: 94,
                    output: 93,
                },
                Deviation::TrackCount {
                    class: TrackClass::Subtitle,
                    source: 1,
                    output: 0,
                },
                Deviation::ChapterCount {
                    source: 2,
                    output: 1,
                },
            ]
            .as_slice()
        );
        let message = error.to_string();
        assert!(message.contains("audio track"), "{message}");
        assert!(!message.contains(';'), "{message}");
    }
}
