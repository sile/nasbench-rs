//! A Rust port of [NASBench].
//!
//! [NASBench]: https://github.com/google-research/nasbench
//!
//! # References
//!
//! - [NAS-Bench-101: Towards Reproducible Neural Architecture Search](https://arxiv.org/abs/1902.09635)
#![warn(missing_docs)]

#[macro_use]
extern crate trackable;

pub use self::dataset::NasBench;

pub mod model;

mod dataset;
mod protobuf;
mod tfrecord;

/// This crate specific `Result` type.
pub type Result<T> = std::result::Result<T, trackable::error::Failure>;
