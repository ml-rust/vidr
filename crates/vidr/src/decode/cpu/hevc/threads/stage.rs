//! The part of each picture on the calling thread: the buffer steps at its start, the
//! reference lists of each slice, and the order of its events. Slice data waits here until
//! the picture ends, then the pool takes the picture whole.

use std::collections::VecDeque;
use std::sync::Arc;

use crate::bitstream::hevc_pps::PicParameterSet;
use crate::bitstream::hevc_slice::SliceSegmentHeader;
use crate::bitstream::hevc_sps_full::SeqParameterSet;
use crate::decode::cpu::dsp::sample::Sample;
use crate::decode::cpu::hevc::dpb::buffer::{Dpb, DpbLimits, RefPicSets};
use crate::decode::cpu::hevc::dpb::ref_lists::SliceLists;
use crate::decode::cpu::hevc::error::HevcError;
use crate::decode::cpu::hevc::hash::picture_hash::PictureHash;
use crate::decode::cpu::hevc::params::CodingParams;
use crate::decode::cpu::hevc::picture_decoder::{
    BufferedStart, PictureBegin, PictureConstants, start_in_buffer,
};
use crate::decode::cpu::hevc::stage::{PictureStage, UnitSite};

use super::handle::PictureHandle;
use super::job::{PictureJob, PicturePools, SegmentJob};
use super::pool::Spawner;
use super::run::{start_abandoned, start_picture};

/// Bytes of slice data one picture holds before its parse starts. Past this, the calling
/// thread parses what it holds, which keeps corrupt input from growing the queue.
const MAX_QUEUED_SLICE_BYTES: usize = 64 << 20;

/// What leaves a [`ThreadedPictures`], in the order the calling-thread path acts.
#[derive(Clone, Debug)]
pub(crate) enum StageEvent<S> {
    /// The place of a picture's end. Its result is the picture's error, if it has one.
    Retire(PictureHandle<S>),
    /// A picture in output order. Its `Retire` comes first.
    Output(PictureHandle<S>),
}

/// The picture whose slices are arriving.
struct BuildingPicture<S> {
    job: PictureJob<S>,
    /// The reference picture set of this picture.
    sets: RefPicSets<PictureHandle<S>>,
    limits: DpbLimits,
    /// `PicOutputFlag`.
    output: bool,
    /// The pictures of every list so far, once each.
    references: Vec<PictureHandle<S>>,
    /// `CtbAddrRsToTs` of the last slice segment.
    last_ts: Option<u32>,
    /// Bytes of slice data not parsed yet.
    queued_bytes: usize,
}

/// Decodes pictures of one sample type on the pool. The buffer holds handles, and buffer
/// order and output order come from the calling thread alone.
pub(crate) struct ThreadedPictures<S> {
    spawner: Spawner,
    pools: Arc<PicturePools<S>>,
    constants: Option<Arc<PictureConstants>>,
    active: Option<BuildingPicture<S>>,
    dpb: Dpb<S, PictureHandle<S>>,
    events: VecDeque<StageEvent<S>>,
    site: UnitSite,
    /// A test sets this to panic in the next picture's parse.
    #[cfg(test)]
    pub(crate) panic_next: bool,
}

impl<S: Sample> ThreadedPictures<S> {
    /// A stage with no tables yet, on the pool of `spawner`.
    pub(crate) fn new(spawner: Spawner) -> Self {
        Self {
            spawner,
            pools: Arc::new(PicturePools::new()),
            constants: None,
            active: None,
            dpb: Dpb::new(),
            events: VecDeque::new(),
            site: UnitSite::End,
            #[cfg(test)]
            panic_next: false,
        }
    }

    /// Moves the pictures the buffer has output to the event order.
    fn move_bumps(&mut self) {
        while let Some(picture) = self.dpb.pop_output() {
            self.events.push_back(StageEvent::Output(picture));
        }
    }
}

impl<S: Sample> PictureStage<S> for ThreadedPictures<S> {
    type Output = StageEvent<S>;

    fn is_active(&self) -> bool {
        self.active.is_some()
    }

    fn begin(
        &mut self,
        sps: &SeqParameterSet,
        pps: &PicParameterSet,
        header: &SliceSegmentHeader,
        begin: PictureBegin,
    ) -> Result<(), HevcError> {
        let constants = match &self.constants {
            Some(constants) if constants.holds(sps, pps) => Arc::clone(constants),
            _ => {
                let (constants, _) = PictureConstants::new(sps, pps, header)?;
                let constants = Arc::new(constants);
                self.constants = Some(Arc::clone(&constants));
                constants
            }
        };
        let params = CodingParams::from_headers(sps, pps, header)?;
        let work = self.pools.take_work(&params, &constants.layout)?;
        let BufferedStart {
            frame,
            sets,
            limits,
        } = start_in_buffer(&mut self.dpb, sps, header, &params, begin)?;
        self.move_bumps();
        let job = PictureJob::new(frame, params, constants, work, self.site);
        #[cfg(test)]
        let job = PictureJob {
            panics: std::mem::take(&mut self.panic_next),
            ..job
        };
        self.active = Some(BuildingPicture {
            job,
            sets,
            limits,
            output: header.pic_output_flag,
            references: Vec::new(),
            last_ts: None,
            queued_bytes: 0,
        });
        Ok(())
    }

    fn decode_slice(
        &mut self,
        header: &SliceSegmentHeader,
        data: Vec<u8>,
    ) -> Result<(), HevcError> {
        let Some(active) = &mut self.active else {
            return Ok(());
        };
        let job = &mut active.job;
        let constants = Arc::clone(&job.constants);
        let params = CodingParams::from_headers(&constants.sps, &constants.pps, header)?;
        let lists = if header.dependent_slice_segment_flag {
            None
        } else {
            let lists = SliceLists::build(header, &active.sets, job.frame.poc, &constants.sps)?;
            for picture in lists.frames.iter().flatten() {
                if !active
                    .references
                    .iter()
                    .any(|known| known.same_picture(picture))
                {
                    active.references.push(picture.clone());
                }
            }
            Some(lists)
        };
        // The parse refuses this address too, so the picture holds no more slices. Each kept
        // segment has a later address, which bounds their count by the CTBs of the picture.
        let ts = constants
            .layout
            .rs_to_ts(header.slice_segment_address as usize);
        if let Some(ts) = ts {
            if active.last_ts.is_some_and(|last| ts <= last) {
                return Err(HevcError::InvalidSyntax {
                    element: "slice_segment_address",
                    value: i64::from(header.slice_segment_address),
                    allowed: "an address after the last CTB of the slice segment before it",
                });
            }
            active.last_ts = Some(ts);
        }
        active.queued_bytes = active.queued_bytes.saturating_add(data.len());
        job.segments.push(SegmentJob {
            params,
            lists,
            data,
            start: header.slice_data_rbsp_offset,
            first_ctb_rs: header.slice_segment_address,
            dependent: header.dependent_slice_segment_flag,
            site: self.site,
            ctus: 0..0,
        });
        if active.queued_bytes > MAX_QUEUED_SLICE_BYTES {
            active.job.parse_pending(&self.spawner);
            active.queued_bytes = 0;
        }
        Ok(())
    }

    fn set_hash(&mut self, hash: PictureHash) {
        if let Some(active) = &mut self.active {
            active.job.hash = Some(hash);
        }
    }

    /// The buffer steps of C.5.2.3 run here with a handle, and the stages start. An error
    /// of the picture comes through its `Retire` event.
    fn finish(&mut self, check_hash: bool) -> Result<(), HevcError> {
        let Some(active) = self.active.take() else {
            return Ok(());
        };
        let BuildingPicture {
            mut job,
            limits,
            output,
            references,
            ..
        } = active;
        job.check_hash = check_hash;
        job.finish_site = self.site;
        self.move_bumps();
        let (handle, resolver) = PictureHandle::pending(&job.frame, self.site);
        self.events.push_back(StageEvent::Retire(handle.clone()));
        self.dpb.insert_picture(handle, output, &limits);
        self.move_bumps();
        start_picture(job, resolver, &references, &self.pools, &self.spawner);
        Ok(())
    }

    /// The picture leaves no frame. Its slices still parse: an error among them comes
    /// before the error that ended the picture.
    fn abandon(&mut self) {
        let Some(active) = self.active.take() else {
            return;
        };
        let mut job = active.job;
        job.finish_site = self.site;
        self.move_bumps();
        let (handle, resolver) = PictureHandle::pending(&job.frame, self.site);
        self.events.push_back(StageEvent::Retire(handle));
        start_abandoned(job, resolver, &self.pools, &self.spawner);
    }

    fn flush_output(&mut self) {
        self.dpb.flush();
        self.move_bumps();
    }

    fn close_sequence(&mut self, discard: bool) {
        self.dpb.close(discard);
        self.move_bumps();
    }

    fn pop_output(&mut self) -> Option<StageEvent<S>> {
        self.move_bumps();
        self.events.pop_front()
    }

    fn enter_unit(&mut self, site: UnitSite) {
        self.site = site;
    }
}
