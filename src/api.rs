use crate::model::{DataPoint, EvalStats, ModelSpec, ModelStats};
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

#[derive(Debug)]
pub struct NasBench {
    models: HashMap<ModelSpec, ModelStats>,
}
impl NasBench {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = track_any_err!(File::open(&path); path.as_ref())?;
        let mut reader = BufReader::new(file);

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

    pub fn models(&self) -> &HashMap<ModelSpec, ModelStats> {
        &self.models
    }

    pub fn from_tfrecord_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut models = HashMap::<_, ModelStats>::new();

        let file = track_any_err!(File::open(&path); path.as_ref())?;
        for record in TfRecordStream::new(BufReader::new(file)) {
            let record = track!(record)?;
            let json: Vec<JsonValue> = track_any_err!(serde_json::from_slice(&record.data))?;
            let record = track!(RawRecord::from_json(json))?;

            let mut model = models.entry(record.spec.clone()).or_default();
            model.trainable_parameters = record.metrics.trainable_parameters as u32;

            let data_point = DataPoint {
                halfway: EvalStats::from(record.metrics.evaluation_data_list[1].clone()),
                complete: EvalStats::from(record.metrics.evaluation_data_list[2].clone()),
            };
            model
                .epochs
                .entry(record.epochs)
                .or_default()
                .push(data_point);
        }

        Ok(Self { models })
    }

    pub fn to_writer<W: Write>(&self, mut writer: W) -> Result<()> {
        for (spec, stats) in &self.models {
            track!(spec.to_writer(&mut writer))?;
            track!(stats.to_writer(&mut writer))?;
        }
        Ok(())
    }

    // pub fn query(&self, model_spec: &ModelSpec, epochs: usize, stop_halfway:bool, TODO: sample_index) {}
    //
}

#[derive(Debug)]
struct RawRecord {
    module_hash: u128,
    epochs: u8,
    spec: ModelSpec,
    metrics: ModelMetrics,
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
        let dim = (raw_adjacency.len() as f64).sqrt() as usize;
        let mut adjacency = vec![vec![false; dim]; dim];
        for i in 0..dim {
            for j in 0..dim {
                adjacency[i][j] = raw_adjacency.as_bytes()[i * dim + j] == '1' as u8;
            }
        }

        // operations
        let raw_operations = track_assert_some!(array[3].as_str(), Failed);
        let mut operations = Vec::new();
        for op in raw_operations.split(',') {
            operations.push(track!(op.parse())?);
        }

        // metrics
        let raw_metrics = track_assert_some!(array[4].as_str(), Failed).replace('\n', "");
        let raw_metrics_bytes = track_any_err!(base64::decode(&raw_metrics); raw_metrics)?;
        let metrics = track!(ModelMetricsDecoder::default().decode_from_bytes(&raw_metrics_bytes))
            .map_err(|e| Failed.takes_over(e))?;

        Ok(Self {
            module_hash,
            epochs,
            spec: ModelSpec {
                adjacency,
                operations,
            },
            metrics,
        })
    }
}
