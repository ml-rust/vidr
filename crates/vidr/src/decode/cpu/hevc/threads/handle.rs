//! A picture that finishes later, as the decoded picture buffer and reference lists hold it.

use std::sync::Arc;

use crate::decode::cpu::dsp::sample::Sample;
use crate::decode::cpu::hevc::dpb::buffer::Frame;
use crate::decode::cpu::hevc::dpb::ref_lists::{SliceLists, SliceRefs};
use crate::decode::cpu::hevc::dpb::stored::{Reclaim, StoredPicture};
use crate::decode::cpu::hevc::error::HevcError;
use crate::decode::cpu::hevc::picture::DisplayWindow;
use crate::decode::cpu::hevc::stage::UnitSite;
use crate::decode::cpu::hevc::syntax::records::push_record;

use super::completion::{Completion, Gate, PictureFailure, Resolver};

/// The facts of a picture known at its start, and the cell of its samples.
#[derive(Clone, Debug)]
pub(crate) struct PictureHandle<S> {
    poc: i32,
    decode_order: u64,
    window: DisplayWindow,
    cell: Arc<Completion<Arc<Frame<S>>>>,
}

impl<S: Sample> PictureHandle<S> {
    /// A handle for the picture that `frame` becomes, and the way to finish it. A resolver
    /// dropped with no result reports at `site`.
    pub(crate) fn pending(frame: &Frame<S>, site: UnitSite) -> (Self, Resolver<Arc<Frame<S>>>) {
        let (cell, resolver) = Completion::pending(site);
        let handle = Self {
            poc: frame.poc,
            decode_order: frame.decode_order,
            window: frame.picture.window(),
            cell,
        };
        (handle, resolver)
    }

    /// The value [`Frame::decode_order`] holds.
    pub(crate) const fn decode_order(&self) -> u64 {
        self.decode_order
    }

    /// The output area in luma samples.
    pub(crate) const fn window(&self) -> DisplayWindow {
        self.window
    }

    /// True when both handles name one picture.
    pub(crate) fn same_picture(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.cell, &other.cell)
    }

    /// See [`Completion::subscribe`].
    pub(crate) fn subscribe(&self, gate: &Arc<Gate>) -> bool {
        self.cell.subscribe(gate)
    }

    /// The result without its samples: blocks with `block`, and otherwise gives `None`
    /// while the picture runs.
    pub(crate) fn settled(&self, block: bool) -> Option<Result<(), PictureFailure>> {
        let result = if block {
            Some(self.cell.wait())
        } else {
            self.cell.poll()
        };
        result.map(|result| result.map(|_| ()))
    }

    /// The finished picture. Blocks until it is here.
    pub(crate) fn frame(&self) -> Result<Arc<Frame<S>>, PictureFailure> {
        self.cell.wait()
    }
}

impl<S: Sample> StoredPicture for PictureHandle<S> {
    type Sample = S;

    const WAITS_TO_RECLAIM: bool = true;

    fn finished(frame: Frame<S>) -> Self {
        Self {
            poc: frame.poc,
            decode_order: frame.decode_order,
            window: frame.picture.window(),
            cell: Completion::ready(Arc::new(frame)),
        }
    }

    fn poc(&self) -> i32 {
        self.poc
    }

    fn reclaim(self) -> Reclaim<Self> {
        let Self {
            poc,
            decode_order,
            window,
            cell,
        } = self;
        let busy = |cell| {
            Reclaim::Busy(Self {
                poc,
                decode_order,
                window,
                cell,
            })
        };
        let completion = match Arc::try_unwrap(cell) {
            Ok(completion) => completion,
            Err(cell) => return busy(cell),
        };
        match completion.into_result() {
            Some(Ok(frame)) => match Arc::try_unwrap(frame) {
                Ok(frame) => Reclaim::Buffer(Box::new(frame)),
                Err(frame) => busy(Completion::ready(frame)),
            },
            _ => Reclaim::Lost,
        }
    }
}

impl<S: Sample> SliceLists<PictureHandle<S>> {
    /// The same lists over finished pictures. Call it after each picture finishes.
    ///
    /// # Errors
    ///
    /// - [`HevcError::ReferenceFailed`] for a picture whose decoding failed.
    /// - [`HevcError::Allocation`] when a list does not fit in memory.
    pub(crate) fn resolve(&self) -> Result<SliceRefs<S>, HevcError> {
        let finished = |handle: &PictureHandle<S>| {
            handle
                .frame()
                .map_err(|_| HevcError::ReferenceFailed { poc: handle.poc })
        };
        let mut refs = SliceRefs {
            frames: [Vec::new(), Vec::new()],
            lists: self.lists,
            col: None,
            no_backward_pred: self.no_backward_pred,
            collocated_from_l0: self.collocated_from_l0,
            weights: self.weights.clone(),
        };
        for (list, handles) in self.frames.iter().enumerate() {
            for handle in handles {
                push_record(
                    &mut refs.frames[list],
                    finished(handle)?,
                    "reference picture list",
                )?;
            }
        }
        if let Some(col) = &self.col {
            refs.col = Some(finished(col)?);
        }
        Ok(refs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::cpu::hevc::motion::col::ColMotion;
    use crate::decode::cpu::hevc::picture::Picture;

    fn frame(poc: i32) -> Frame<u8> {
        Frame {
            picture: Picture::new(8, 8).unwrap(),
            col: ColMotion::new(),
            poc,
            decode_order: 3,
        }
    }

    #[test]
    fn a_handle_states_its_facts_before_the_picture_finishes() {
        let source = frame(9);
        let (handle, resolver) = PictureHandle::pending(&source, UnitSite::End);
        assert_eq!(handle.poc(), 9);
        assert_eq!(handle.decode_order(), 3);
        assert_eq!(handle.window().width, 8);
        assert_eq!(handle.settled(false), None);
        resolver.resolve(Ok(Arc::new(source)));
        assert_eq!(handle.settled(false), Some(Ok(())));
        assert_eq!(handle.frame().unwrap().poc, 9);
    }

    /// A buffer comes back only when no list, task, or output holds the picture.
    #[test]
    fn a_buffer_returns_after_its_last_holder() {
        let source = frame(1);
        let (handle, resolver) = PictureHandle::pending(&source, UnitSite::End);
        let Reclaim::Busy(handle) = handle.reclaim() else {
            panic!("the resolver holds the cell");
        };
        resolver.resolve(Ok(Arc::new(source)));
        let held = handle.frame().unwrap();
        let Reclaim::Busy(handle) = handle.reclaim() else {
            panic!("a reader holds the frame");
        };
        let copy = handle.clone();
        let Reclaim::Busy(handle) = handle.reclaim() else {
            panic!("a second handle holds the cell");
        };
        drop(copy);
        drop(held);
        assert!(matches!(handle.reclaim(), Reclaim::Buffer(_)));
    }

    #[test]
    fn a_failed_picture_returns_no_buffer() {
        let source = frame(1);
        let (handle, resolver) = PictureHandle::pending(&source, UnitSite::End);
        resolver.resolve(Err(PictureFailure::Abandoned));
        assert!(matches!(handle.reclaim(), Reclaim::Lost));
    }

    #[test]
    fn a_list_over_a_failed_picture_names_it() {
        let source = frame(5);
        let (handle, resolver) = PictureHandle::pending(&source, UnitSite::End);
        resolver.resolve(Err(PictureFailure::Abandoned));
        let mut lists = SliceLists::<PictureHandle<u8>>::intra();
        lists.frames[0].push(handle);
        assert_eq!(
            lists.resolve().map(|_| ()),
            Err(HevcError::ReferenceFailed { poc: 5 })
        );
    }

    #[test]
    fn a_list_over_finished_pictures_keeps_its_order() {
        let mut lists = SliceLists::<PictureHandle<u8>>::intra();
        lists.frames[0].push(PictureHandle::finished(frame(4)));
        lists.frames[0].push(PictureHandle::finished(frame(2)));
        lists.frames[1].push(PictureHandle::finished(frame(8)));
        lists.col = Some(PictureHandle::finished(frame(8)));
        let refs = lists.resolve().unwrap();
        let pocs: Vec<i32> = refs.frames[0].iter().map(|frame| frame.poc).collect();
        assert_eq!(pocs, [4, 2]);
        assert_eq!(refs.frames[1][0].poc, 8);
        assert_eq!(refs.col.unwrap().poc, 8);
    }
}
