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

pub mod api;
pub mod model;

mod protobuf;
mod tfrecord;

/// This crate specific `Result` type.
pub type Result<T> = std::result::Result<T, trackable::error::Failure>;
