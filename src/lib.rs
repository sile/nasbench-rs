//! A Rust port of [google-research/nasbench][NASBench].
//!
//! [NASBench]: https://github.com/google-research/nasbench
//!
//! # Motivations
//!
//! # Limitations
//!
//! # Examples
//!
//! TODO
//!
//! # References
//!
//! - [NAS-Bench-101: Towards Reproducible Neural Architecture Search](https://arxiv.org/abs/1902.09635)
#![warn(missing_docs)]

#[macro_use]
extern crate trackable;

pub use self::dataset::NasBench;
pub use self::model::{AdjacencyMatrix, EpochStats, EvaluationMetrics, ModelSpec, ModelStats, Op};

mod dataset;
mod model;
mod protobuf;
mod tfrecord;

/// This crate specific `Result` type.
pub type Result<T> = std::result::Result<T, trackable::error::Failure>;
