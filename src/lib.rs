#[macro_use]
extern crate trackable;

pub mod api;
pub mod model;
pub mod protobuf;
pub mod tfrecord;

pub type Result<T> = std::result::Result<T, trackable::error::Failure>;
