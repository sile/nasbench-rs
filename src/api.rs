use crate::tfrecord::TfRecordStream;
use crate::Result;
use serde_json::{self, Value as JsonValue};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::u128;
use trackable::error::Failed;

#[derive(Debug)]
pub struct NasBench {}
impl NasBench {
    pub fn from_tfrecord_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = track_any_err!(File::open(&path); path.as_ref())?;
        for record in TfRecordStream::new(BufReader::new(file)) {
            let record = track!(record)?;
            let json: Vec<JsonValue> = track_any_err!(serde_json::from_slice(&record.data))?;
            let record = track!(RawRecord::from_json(json))?;
        }
        panic!()
    }
}

#[derive(Debug)]
struct RawRecord {
    module_hash: u128,
    epochs: u8,
}
impl RawRecord {
    fn from_json(array: Vec<JsonValue>) -> Result<Self> {
        track_assert_eq!(array.len(), 5, Failed);

        // module hash
        let module_hash = track_assert_some!(array[0].as_str(), Failed);
        track_assert_eq!(module_hash.len(), 32, Failed);
        let module_hash = track_any_err!(u128::from_str_radix(&module_hash, 16))?;

        // epochs
        let epochs = track_assert_some!(array[1].as_i64(), Failed) as u8;

        // adjacency
        let raw_adjacency = track_assert_some!(array[2].as_str(), Failed);

        Ok(Self {
            module_hash,
            epochs,
        })
    }
}
