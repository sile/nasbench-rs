use crate::protobuf::EvaluationData;
use crate::Result;
use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};
use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::str::FromStr;
use trackable::error::{Failed, Failure};

#[derive(Debug, Clone)]
pub struct EvalStats {
    pub training_time: f64,
    pub training_accuracy: f64,
    pub validation_accuracy: f64,
    pub test_accuracy: f64,
}
impl EvalStats {
    pub fn from_reader<R: Read>(mut reader: R) -> Result<Self> {
        let training_time = track_any_err!(reader.read_f64::<BigEndian>())?;
        let training_accuracy = track_any_err!(reader.read_f64::<BigEndian>())?;
        let validation_accuracy = track_any_err!(reader.read_f64::<BigEndian>())?;
        let test_accuracy = track_any_err!(reader.read_f64::<BigEndian>())?;
        Ok(Self {
            training_time,
            training_accuracy,
            validation_accuracy,
            test_accuracy,
        })
    }

    pub fn to_writer<W: Write>(&self, mut writer: W) -> Result<()> {
        track_any_err!(writer.write_f64::<BigEndian>(self.training_time))?;
        track_any_err!(writer.write_f64::<BigEndian>(self.training_accuracy))?;
        track_any_err!(writer.write_f64::<BigEndian>(self.validation_accuracy))?;
        track_any_err!(writer.write_f64::<BigEndian>(self.test_accuracy))?;
        Ok(())
    }
}
impl From<EvaluationData> for EvalStats {
    fn from(f: EvaluationData) -> Self {
        Self {
            training_time: f.training_time,
            training_accuracy: f.train_accuracy,
            validation_accuracy: f.validation_accuracy,
            test_accuracy: f.test_accuracy,
        }
    }
}

#[derive(Debug, Default)]
pub struct ModelStats {
    pub trainable_parameters: u32,
    pub epochs: BTreeMap<u8, Vec<DataPoint>>,
}
impl ModelStats {
    pub fn from_reader<R: Read>(mut reader: R) -> Result<Self> {
        let trainable_parameters = track_any_err!(reader.read_u32::<BigEndian>())?;

        let len = track_any_err!(reader.read_u8())? as usize;
        let mut epochs_map = BTreeMap::new();
        for _ in 0..len {
            let epochs = track_any_err!(reader.read_u8())?;

            let len = track_any_err!(reader.read_u8())? as usize;
            let mut data_points = Vec::with_capacity(len);
            for _ in 0..len {
                let dp = track!(DataPoint::from_reader(&mut reader))?;
                data_points.push(dp);
            }

            epochs_map.insert(epochs, data_points);
        }

        Ok(Self {
            trainable_parameters,
            epochs: epochs_map,
        })
    }

    pub fn to_writer<W: Write>(&self, mut writer: W) -> Result<()> {
        track_any_err!(writer.write_u32::<BigEndian>(self.trainable_parameters))?;

        track_any_err!(writer.write_u8(self.epochs.len() as u8))?;
        for (epochs, data_points) in &self.epochs {
            track_any_err!(writer.write_u8(*epochs))?;

            track_any_err!(writer.write_u8(data_points.len() as u8))?;
            for dp in data_points {
                track!(dp.to_writer(&mut writer))?;
            }
        }

        Ok(())
    }
}

#[derive(Debug)]
pub struct DataPoint {
    pub halfway: EvalStats,
    pub complete: EvalStats,
}
impl DataPoint {
    pub fn from_reader<R: Read>(mut reader: R) -> Result<Self> {
        let halfway = track!(EvalStats::from_reader(&mut reader))?;
        let complete = track!(EvalStats::from_reader(&mut reader))?;
        Ok(Self { halfway, complete })
    }

    pub fn to_writer<W: Write>(&self, mut writer: W) -> Result<()> {
        track!(self.halfway.to_writer(&mut writer))?;
        track!(self.complete.to_writer(&mut writer))?;
        Ok(())
    }
}

// a.k.a. module
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ModelSpec {
    pub adjacency: Vec<Vec<bool>>, // TODO: AdjacencyMatrix (upper triangular)
    pub operations: Vec<Op>,
}
impl ModelSpec {
    // pub fn new() -> Result<Self> {
    // }

    pub fn from_reader<R: Read>(mut reader: R) -> Result<Self> {
        let dim = track_any_err!(reader.read_u8())? as usize;
        let mut adjacency = vec![vec![false; dim]; dim];
        for i in 0..dim {
            for j in 0..dim {
                adjacency[i][j] = track_any_err!(reader.read_u8())? == 1;
            }
        }

        let len = track_any_err!(reader.read_u8())? as usize;
        let mut operations = Vec::with_capacity(len);
        for _ in 0..len {
            let op = match track_any_err!(reader.read_u8())? {
                0 => Op::Input,
                1 => Op::Conv1x1,
                2 => Op::Conv3x3,
                3 => Op::MaxPool3x3,
                4 => Op::Output,
                n => track_panic!(Failed, "Unknown operation number: {}", n),
            };
            operations.push(op);
        }

        Ok(Self {
            adjacency,
            operations,
        })
    }

    pub fn to_writer<W: Write>(&self, mut writer: W) -> Result<()> {
        let dim = self.adjacency.len();
        track_any_err!(writer.write_u8(dim as u8))?;
        for i in 0..dim {
            for j in 0..dim {
                track_any_err!(writer.write_u8(self.adjacency[i][j] as u8))?;
            }
        }

        track_any_err!(writer.write_u8(self.operations.len() as u8))?;
        for op in &self.operations {
            track_any_err!(writer.write_u8(*op as u8))?;
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Op {
    Input,
    Conv1x1,
    Conv3x3,
    MaxPool3x3,
    Output,
}
impl FromStr for Op {
    type Err = Failure;

    fn from_str(op: &str) -> Result<Self> {
        Ok(match op {
            "input" => Op::Input,
            "conv1x1-bn-relu" => Op::Conv1x1,
            "conv3x3-bn-relu" => Op::Conv3x3,
            "maxpool3x3" => Op::MaxPool3x3,
            "output" => Op::Output,
            _ => track_panic!(Failed, "Unknown operator: {:?}", op),
        })
    }
}
