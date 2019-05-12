use crate::model::{EpochStats, EvaluationMetrics, ModelSpec, ModelStats};
use crate::protobuf::{ModelMetrics, ModelMetricsDecoder};
use crate::tfrecord::TfRecordStream;
use crate::Result;
use base64;
use bytecodec::DecodeExt;
use serde_json::{self, Value as JsonValue};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::path::Path;
use std::u128;
use trackable::error::{ErrorKindExt, Failed};

/// NAS benchmark dataset.
#[derive(Debug)]
pub struct NasBench {
    models: HashMap<ModelSpec, ModelStats>,
}
impl NasBench {
    /// Loads a `NasBench` instance from the given binary file.
    ///
    /// Note that this function assumes the file contains bytes produced by `NasBench::to_writer` method.
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = track_any_err!(File::open(&path); path.as_ref())?;
        track!(Self::from_reader(BufReader::new(file)))
    }

    /// Returns models contained in this dataset.
    pub fn models(&self) -> &HashMap<ModelSpec, ModelStats> {
        &self.models
    }

    /// Serializes the state of this dataset to the given writer.
    pub fn to_writer<W: Write>(&self, mut writer: W) -> Result<()> {
        for (spec, stats) in &self.models {
            track!(spec.to_writer(&mut writer))?;
            track!(stats.to_writer(&mut writer))?;
        }
        Ok(())
    }

    /// Deserializes a `NasBench` instance from the given reader.
    pub fn from_reader<R: Read>(mut reader: R) -> Result<Self> {
        let mut models = HashMap::new();
        loop {
            let mut peek = [0; 1];
            if track_any_err!(reader.read(&mut peek))? == 0 {
                break;
            }

            let mut reader = peek.chain(&mut reader);
            let spec = track!(ModelSpec::from_reader(&mut reader))?;
            let stats = track!(ModelStats::from_reader(&mut reader))?;
            models.insert(spec, stats);
        }

        Ok(Self { models })
    }

    /// Deserializes a `NasBench` instance from the given reader (tfrecord format).
    ///
    /// See [Download the dataset] for available dataset.
    ///
    /// [Download the dataset]: https://github.com/google-research/nasbench#download-the-dataset
    pub fn from_tfrecord_reader<R: Read>(reader: R) -> Result<Self> {
        let mut models = HashMap::<_, ModelStats>::new();

        for record in TfRecordStream::new(reader) {
            let record = track!(record)?;
            let json: Vec<JsonValue> = track_any_err!(serde_json::from_slice(&record.data))?;
            let record = track!(NasBenchRecord::from_json(json))?;

            let mut model = models.entry(record.spec.clone()).or_default();
            model.trainable_parameters = record.metrics.trainable_parameters as u32;

            let epoch_stats = EpochStats {
                halfway: EvaluationMetrics::from(record.metrics.evaluation_data_list[1].clone()),
                complete: EvaluationMetrics::from(record.metrics.evaluation_data_list[2].clone()),
            };
            model
                .epochs
                .entry(record.epochs)
                .or_default()
                .push(epoch_stats);
        }

        Ok(Self { models })
    }
}

#[derive(Debug)]
struct NasBenchRecord {
    module_hash: u128,
    epochs: u8,
    spec: ModelSpec,
    metrics: ModelMetrics,
}
impl NasBenchRecord {
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
        let adjacency = track!(raw_adjacency.parse())?;

        // operations
        let raw_ops = track_assert_some!(array[3].as_str(), Failed);
        let mut ops = Vec::new();
        for op in raw_ops.split(',') {
            ops.push(track!(op.parse())?);
        }

        // metrics
        let raw_metrics = track_assert_some!(array[4].as_str(), Failed).replace('\n', "");
        let raw_metrics_bytes = track_any_err!(base64::decode(&raw_metrics); raw_metrics)?;
        let metrics = track!(ModelMetricsDecoder::default().decode_from_bytes(&raw_metrics_bytes))
            .map_err(|e| Failed.takes_over(e))?;

        Ok(Self {
            module_hash,
            epochs,
            spec: ModelSpec { ops, adjacency },
            metrics,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trackable::result::TopLevelResult;

    #[test]
    fn nasbench_works() -> TopLevelResult {
        let tfrecord_bytes = [
            145, 1, 0, 0, 0, 0, 0, 0, 0x98, 0x25, 0xed, 0x9b, 91, 34, 48, 48, 48, 48, 53, 99, 49,
            52, 50, 101, 54, 102, 52, 56, 97, 99, 55, 52, 102, 100, 99, 102, 55, 51, 101, 51, 52,
            51, 57, 56, 55, 52, 34, 44, 32, 52, 44, 32, 34, 48, 49, 48, 48, 49, 49, 48, 48, 48, 49,
            48, 48, 48, 48, 48, 48, 48, 49, 48, 48, 49, 48, 48, 48, 48, 48, 49, 48, 48, 48, 48, 48,
            48, 49, 48, 48, 48, 48, 48, 48, 48, 49, 48, 48, 48, 48, 48, 48, 48, 34, 44, 32, 34,
            105, 110, 112, 117, 116, 44, 99, 111, 110, 118, 51, 120, 51, 45, 98, 110, 45, 114, 101,
            108, 117, 44, 109, 97, 120, 112, 111, 111, 108, 51, 120, 51, 44, 99, 111, 110, 118, 51,
            120, 51, 45, 98, 110, 45, 114, 101, 108, 117, 44, 99, 111, 110, 118, 51, 120, 51, 45,
            98, 110, 45, 114, 101, 108, 117, 44, 99, 111, 110, 118, 49, 120, 49, 45, 98, 110, 45,
            114, 101, 108, 117, 44, 111, 117, 116, 112, 117, 116, 34, 44, 32, 34, 67, 105, 48, 74,
            65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 82, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65,
            65, 90, 65, 65, 65, 65, 111, 75, 113, 113, 117, 84, 56, 104, 65, 65, 65, 65, 89, 74,
            65, 71, 117, 84, 56, 112, 65, 65, 65, 65, 65, 71, 109, 81, 117, 84, 56, 75, 76, 81,
            107, 65, 65, 65, 65, 65, 65, 65, 65, 65, 92, 110, 81, 66, 69, 65, 65, 65, 68, 65, 100,
            74, 78, 71, 81, 66, 107, 65, 65, 65, 66, 103, 107, 65, 97, 54, 80, 121, 69, 65, 65, 65,
            65, 103, 51, 47, 75, 54, 80, 121, 107, 65, 65, 65, 67, 65, 121, 55, 101, 54, 80, 119,
            111, 116, 67, 81, 65, 65, 65, 65, 65, 65, 65, 66, 66, 65, 69, 81, 65, 65, 65, 67, 68,
            98, 101, 86, 90, 65, 92, 110, 71, 81, 65, 65, 65, 69, 65, 97, 90, 78, 89, 47, 73, 81,
            65, 65, 65, 71, 66, 86, 86, 100, 89, 47, 75, 81, 65, 65, 65, 79, 68, 121, 98, 100, 89,
            47, 69, 73, 113, 89, 105, 103, 81, 90, 65, 65, 66, 65, 104, 122, 110, 98, 101, 85, 65,
            61, 92, 110, 34, 93, 0x68, 0x16, 0x69, 0x03,
        ];
        let nasbench0 = track!(NasBench::from_tfrecord_reader(&tfrecord_bytes[..]))?;

        let mut bytes = Vec::new();
        track!(nasbench0.to_writer(&mut bytes))?;
        let nasbench1 = track!(NasBench::from_reader(&bytes[..]))?;

        assert_eq!(nasbench0.models(), nasbench1.models());

        Ok(())
    }
}
