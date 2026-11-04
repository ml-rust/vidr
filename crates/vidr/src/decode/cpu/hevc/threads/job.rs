//! One picture as the stages take it, and the memory the pictures of a decoder reuse.

use std::ops::Range;
use std::sync::{Arc, Mutex, MutexGuard};

use crate::decode::cpu::dsp::sample::Sample;
use crate::decode::cpu::hevc::dpb::buffer::Frame;
use crate::decode::cpu::hevc::dpb::ref_lists::{SliceLists, SliceRefs};
use crate::decode::cpu::hevc::error::HevcError;
use crate::decode::cpu::hevc::filter::deblock::{DeblockContext, deblock_picture_in_rows};
use crate::decode::cpu::hevc::filter::edges::StrengthInput;
use crate::decode::cpu::hevc::filter::sao::{SaoContext, sao_is_active, sao_picture_in_rows};
use crate::decode::cpu::hevc::hash::picture_hash::{PictureHash, check_picture_hash};
use crate::decode::cpu::hevc::layout::PicLayout;
use crate::decode::cpu::hevc::params::CodingParams;
use crate::decode::cpu::hevc::picture::Picture;
use crate::decode::cpu::hevc::picture_decoder::{PictureConstants, PictureMaps};
use crate::decode::cpu::hevc::recon::ctu::{ReconContext, Reconstructor};
use crate::decode::cpu::hevc::recon::inter::InterContext;
use crate::decode::cpu::hevc::stage::UnitSite;
use crate::decode::cpu::hevc::syntax::records::{CtuRecords, push_record};
use crate::decode::cpu::hevc::syntax::slice_data::{
    EntropyCarry, SegmentInput, parse_slice_segment,
};

use super::completion::PictureFailure;
use super::handle::PictureHandle;
use super::pool::Spawner;

/// What the size of the maps depends on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MapShape {
    width: u32,
    height: u32,
    min_cb_log2: u32,
    ctb_log2: u32,
}

impl MapShape {
    const fn of(params: &CodingParams) -> Self {
        Self {
            width: params.width,
            height: params.height,
            min_cb_log2: params.min_cb_log2,
            ctb_log2: params.ctb_log2,
        }
    }
}

/// The memory one picture writes outside its samples. It keeps its capacity between
/// pictures.
pub(crate) struct PictureWork {
    shape: MapShape,
    maps: PictureMaps,
    /// The parsed CTUs of the picture, in decoding order. Entries past `ctus_used` of the
    /// job are spare memory.
    ctus: Vec<CtuRecords>,
}

/// The memory the pictures of one decoder reuse. Each lock runs once per picture.
pub(crate) struct PicturePools<S> {
    work: Mutex<Vec<PictureWork>>,
    reconstructors: Mutex<Vec<Box<Reconstructor>>>,
    deblocked: Mutex<Vec<Picture<S>>>,
}

/// Takes `lock`, and names it when a panic has poisoned it.
fn locked<'a, T>(lock: &'a Mutex<T>, what: &'static str) -> Result<MutexGuard<'a, T>, HevcError> {
    lock.lock().map_err(|_| HevcError::LockPoisoned { what })
}

impl<S: Sample> PicturePools<S> {
    /// Empty pools.
    pub(crate) const fn new() -> Self {
        Self {
            work: Mutex::new(Vec::new()),
            reconstructors: Mutex::new(Vec::new()),
            deblocked: Mutex::new(Vec::new()),
        }
    }

    /// Maps for a new picture, reset. Spare maps of another size drop.
    ///
    /// # Errors
    ///
    /// - [`HevcError::LockPoisoned`] after a panic with the lock held.
    /// - [`HevcError::Allocation`] when a map does not fit in memory.
    pub(crate) fn take_work(
        &self,
        params: &CodingParams,
        layout: &PicLayout,
    ) -> Result<PictureWork, HevcError> {
        let shape = MapShape::of(params);
        let reused = {
            let mut spare = locked(&self.work, "the spare picture maps")?;
            spare.retain(|work| work.shape == shape);
            spare.pop()
        };
        let mut work = match reused {
            Some(work) => work,
            None => PictureWork {
                shape,
                maps: PictureMaps::new(params, layout)?,
                ctus: Vec::new(),
            },
        };
        work.maps.reset();
        Ok(work)
    }

    /// Returns maps. A poisoned lock drops them, and the next take names it.
    fn give_work(&self, work: PictureWork) {
        if let Ok(mut spare) = self.work.lock() {
            spare.push(work);
        }
    }

    fn take_reconstructor(&self) -> Result<Box<Reconstructor>, HevcError> {
        let reused = locked(&self.reconstructors, "the spare reconstructors")?.pop();
        Ok(reused.unwrap_or_else(|| Box::new(Reconstructor::new())))
    }

    fn give_reconstructor(&self, reconstructor: Box<Reconstructor>) {
        if let Ok(mut spare) = self.reconstructors.lock() {
            spare.push(reconstructor);
        }
    }

    fn take_deblocked(&self, width: usize, height: usize) -> Result<Picture<S>, HevcError> {
        let reused = {
            let mut spare = locked(&self.deblocked, "the spare SAO pictures")?;
            spare.retain(|picture: &Picture<S>| {
                picture.luma().width() == width && picture.luma().height() == height
            });
            spare.pop()
        };
        match reused {
            Some(picture) => Ok(picture),
            None => Picture::new(width, height),
        }
    }

    fn give_deblocked(&self, picture: Picture<S>) {
        if let Ok(mut spare) = self.deblocked.lock() {
            spare.push(picture);
        }
    }
}

/// One slice segment waiting for its stages.
pub(crate) struct SegmentJob<S> {
    pub(crate) params: CodingParams,
    /// The lists of a new slice. `None` for a dependent slice segment.
    pub(crate) lists: Option<SliceLists<PictureHandle<S>>>,
    /// The NAL unit without emulation prevention bytes. Empty after the parse.
    pub(crate) data: Vec<u8>,
    /// The byte where the slice data starts.
    pub(crate) start: usize,
    /// `slice_segment_address`.
    pub(crate) first_ctb_rs: u32,
    /// `dependent_slice_segment_flag`.
    pub(crate) dependent: bool,
    /// The NAL unit an error of this segment belongs to.
    pub(crate) site: UnitSite,
    /// The entries of [`PictureWork::ctus`] this segment parsed.
    pub(crate) ctus: Range<usize>,
}

/// What a picture's stages own: its buffers, its slices, and what they have found.
pub(crate) struct PictureJob<S> {
    /// The buffer that becomes a decoded picture.
    pub(crate) frame: Frame<S>,
    /// The first slice's values. In-loop filters read fields that hold per picture.
    pub(crate) params: CodingParams,
    pub(crate) hash: Option<PictureHash>,
    pub(crate) check_hash: bool,
    pub(crate) constants: Arc<PictureConstants>,
    pub(crate) work: PictureWork,
    pub(crate) carry: EntropyCarry,
    pub(crate) segments: Vec<SegmentJob<S>>,
    /// The count of leading segments the parse has taken.
    pub(crate) parsed: usize,
    /// The count of leading entries of [`PictureWork::ctus`] in use.
    pub(crate) ctus_used: usize,
    /// The first error of the parse.
    pub(crate) failure: Option<PictureFailure>,
    /// True when an error on the calling thread ended this picture.
    pub(crate) abandoned: bool,
    /// The NAL unit an error at the end of the picture belongs to.
    pub(crate) finish_site: UnitSite,
    /// A test sets this to panic in the parse stage.
    #[cfg(test)]
    pub(crate) panics: bool,
}

impl<S: Sample> PictureJob<S> {
    /// A job with no slices.
    pub(crate) fn new(
        frame: Frame<S>,
        params: CodingParams,
        constants: Arc<PictureConstants>,
        work: PictureWork,
        site: UnitSite,
    ) -> Self {
        Self {
            frame,
            params,
            hash: None,
            check_hash: false,
            constants,
            work,
            carry: EntropyCarry::default(),
            segments: Vec::new(),
            parsed: 0,
            ctus_used: 0,
            failure: None,
            abandoned: false,
            finish_site: site,
            #[cfg(test)]
            panics: false,
        }
    }

    /// Parses every segment not parsed yet, and stops at the first error. The slice data
    /// needs no reference samples, so this runs before any other picture finishes.
    pub(crate) fn parse_pending(&mut self, spawner: &Spawner) {
        #[cfg(test)]
        assert!(!self.panics, "the test asks for a panic");
        let Self {
            constants,
            work,
            carry,
            segments,
            parsed,
            ctus_used,
            failure,
            ..
        } = self;
        for segment in segments.iter_mut().skip(*parsed) {
            if failure.is_none() && spawner.stopped() {
                *failure = Some(PictureFailure::Decode {
                    error: HevcError::WorkerStopped,
                    site: segment.site,
                });
            }
            if failure.is_some() {
                segment.data = Vec::new();
                *parsed += 1;
                continue;
            }
            let input = SegmentInput {
                params: &segment.params,
                layout: &constants.layout,
                data: &segment.data,
                start: segment.start,
                first_ctb_rs: segment.first_ctb_rs,
                dependent: segment.dependent,
            };
            let first = *ctus_used;
            let (maps, ctus) = (&mut work.maps.maps, &mut work.ctus);
            let result = parse_slice_segment(&input, maps, carry, |_, records| {
                keep_records(ctus, ctus_used, records)
            });
            segment.ctus = first..*ctus_used;
            segment.data = Vec::new();
            *parsed += 1;
            if let Err(error) = result {
                *failure = Some(PictureFailure::Decode {
                    error,
                    site: segment.site,
                });
            }
        }
    }

    /// What ends the picture after the parse, if anything does.
    pub(crate) fn parse_outcome(&mut self) -> Option<PictureFailure> {
        if let Some(failure) = self.failure.take() {
            return Some(failure);
        }
        if self.abandoned {
            return Some(PictureFailure::Abandoned);
        }
        let expected = self.constants.layout.ctb_count();
        if self.carry.decoded_ctbs != expected {
            return Some(PictureFailure::Decode {
                error: HevcError::IncompletePicture {
                    decoded: self.carry.decoded_ctbs,
                    expected,
                },
                site: self.finish_site,
            });
        }
        None
    }

    /// The reconstruct stage, in-loop filters, hash check, and motion compression. Call it
    /// when every picture of the lists has finished, so nothing here waits.
    ///
    /// # Errors
    ///
    /// - [`HevcError::ReferenceFailed`] for a list over a failed picture.
    /// - [`HevcError::WorkerStopped`] after a shutdown request.
    /// - [`HevcError::PictureHashMismatch`] when a plane differs from its hash.
    /// - [`HevcError::Allocation`] when a buffer does not fit in memory.
    pub(crate) fn reconstruct(
        &mut self,
        pools: &PicturePools<S>,
        spawner: &Spawner,
    ) -> Result<(), HevcError> {
        let mut reconstructor = pools.take_reconstructor()?;
        let result = self.reconstruct_ctus(&mut reconstructor, spawner);
        pools.give_reconstructor(reconstructor);
        result?;
        self.filter(pools)?;
        if self.check_hash
            && let Some(hash) = &self.hash
        {
            check_picture_hash(&self.frame.picture, hash)?;
        }
        let (width, height) = (self.params.width, self.params.height);
        self.work
            .maps
            .field
            .compress(width, height, &mut self.frame.col)
    }

    /// Each CTU in decoding order. The maps hold the whole picture here, and each location
    /// these stages read comes earlier in decoding order: 6.4.1 refuses a later one. So each
    /// read gives the value it gives with the parse and this stage interleaved.
    fn reconstruct_ctus(
        &mut self,
        reconstructor: &mut Reconstructor,
        spawner: &Spawner,
    ) -> Result<(), HevcError> {
        let layout = &self.constants.layout;
        let factors = self.constants.factors.as_ref();
        let (maps, edges, field) = (
            &self.work.maps.maps,
            &mut self.work.maps.edges,
            &mut self.work.maps.field,
        );
        let ctus = self.work.ctus.as_slice();
        let poc = self.frame.poc;
        let picture = &mut self.frame.picture;
        let mut refs: SliceRefs<S> = SliceRefs::intra();
        for segment in &self.segments {
            if spawner.stopped() {
                return Err(HevcError::WorkerStopped);
            }
            if let Some(lists) = &segment.lists {
                refs = lists.resolve()?;
                field.begin_slice(refs.lists)?;
            }
            let params = &segment.params;
            for records in ctus.get(segment.ctus.clone()).unwrap_or_default() {
                let context = ReconContext {
                    params,
                    layout,
                    maps,
                    factors,
                };
                let mut inter = InterContext {
                    refs: &refs,
                    field: &mut *field,
                    poc,
                };
                reconstructor.reconstruct_ctu(&context, &mut inter, records, picture);
                edges.mark_ctu(
                    &StrengthInput {
                        params,
                        layout,
                        maps,
                        field: &*field,
                    },
                    records,
                );
            }
        }
        Ok(())
    }

    /// The in-loop filters (8.7), over the rows of the picture in parallel.
    fn filter(&mut self, pools: &PicturePools<S>) -> Result<(), HevcError> {
        let layout = &self.constants.layout;
        let picture = &mut self.frame.picture;
        deblock_picture_in_rows(
            picture,
            &DeblockContext {
                params: &self.params,
                layout,
                maps: &self.work.maps.maps,
                edges: &self.work.maps.edges,
            },
        );
        let sao = SaoContext {
            params: &self.params,
            layout,
            maps: &self.work.maps.maps,
        };
        if sao_is_active(&sao) {
            let (width, height) = (picture.luma().width(), picture.luma().height());
            let mut deblocked = pools.take_deblocked(width, height)?;
            sao_picture_in_rows(picture, &mut deblocked, &sao);
            pools.give_deblocked(deblocked);
        }
        Ok(())
    }

    /// The finished frame. Its maps return to `pools`, and its handles to other pictures
    /// drop, so those buffers can return too.
    pub(crate) fn into_frame(self, pools: &PicturePools<S>) -> Frame<S> {
        let Self { frame, work, .. } = self;
        pools.give_work(work);
        frame
    }
}

/// Keeps the parser's records for the reconstruct stage. The parser takes the next entry's
/// memory in exchange, so a picture copies no coefficient.
fn keep_records(
    ctus: &mut Vec<CtuRecords>,
    used: &mut usize,
    records: &mut CtuRecords,
) -> Result<(), HevcError> {
    if *used == ctus.len() {
        push_record(ctus, CtuRecords::default(), "parsed CTUs")?;
    }
    if let Some(entry) = ctus.get_mut(*used) {
        std::mem::swap(entry, records);
    }
    *used += 1;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::cpu::hevc::syntax::records::Coeff;

    fn records(level: i16) -> CtuRecords {
        CtuRecords {
            coeffs: vec![Coeff { x: 0, y: 0, level }],
            ..CtuRecords::default()
        }
    }

    #[test]
    fn records_swap_with_spare_memory() {
        let mut ctus = vec![CtuRecords::default()];
        let mut used = 0;
        let mut current = records(5);
        keep_records(&mut ctus, &mut used, &mut current).unwrap();
        assert_eq!(used, 1);
        assert_eq!(ctus[0].coeffs[0].level, 5);
        assert!(current.coeffs.is_empty());
        let mut current = records(6);
        keep_records(&mut ctus, &mut used, &mut current).unwrap();
        assert_eq!(ctus.len(), 2);
        assert_eq!(ctus[1].coeffs[0].level, 6);
        // A second picture takes the memory of the first, and the parser clears it.
        used = 0;
        let mut current = records(7);
        keep_records(&mut ctus, &mut used, &mut current).unwrap();
        assert_eq!(ctus[0].coeffs[0].level, 7);
        assert_eq!(current.coeffs[0].level, 5);
    }

    #[test]
    fn a_poisoned_pool_lock_is_named() {
        let lock = Mutex::new(0u8);
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = lock.lock().unwrap();
            panic!("poison");
        }));
        assert!(matches!(
            locked(&lock, "a test lock"),
            Err(HevcError::LockPoisoned {
                what: "a test lock"
            })
        ));
    }
}
