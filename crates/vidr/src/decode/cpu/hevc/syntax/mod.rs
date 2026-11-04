//! The parse stage: the slice data loop (7.3.8.1), SAO syntax (7.3.8.3), coding
//! quadtree, coding units, prediction units, transform trees, and residual coding
//! (7.3.8.4 to 7.3.8.11).

pub(crate) mod coding_unit;
pub(crate) mod inter_pu;
pub(crate) mod intra_mode;
pub(crate) mod maps;
pub(crate) mod parser;
pub(crate) mod pcm;
pub(crate) mod quadtree;
pub(crate) mod records;
pub(crate) mod residual;
pub(crate) mod residual_ctx;
pub(crate) mod sao;
pub(crate) mod slice_data;
pub(crate) mod transform_tree;

#[cfg(test)]
mod coding_unit_tests;
#[cfg(test)]
mod inter_pu_tests;
#[cfg(test)]
mod residual_tests;
#[cfg(test)]
mod slice_data_tests;
#[cfg(test)]
mod test_residual_writer;
#[cfg(test)]
mod test_writer;
#[cfg(test)]
mod transform_tree_tests;
