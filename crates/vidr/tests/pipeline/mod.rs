//! The threaded pump. Every test but those in `gpu` runs fakes over CPU tensors
//! and needs no GPU. A test whose run can hang fails on a watchdog timeout.

mod fakes;
mod faults;
#[cfg(all(feature = "nvdec", feature = "nvenc", feature = "mp4", feature = "mkv"))]
mod gpu;
mod pressure;
mod routes;
mod timing;
