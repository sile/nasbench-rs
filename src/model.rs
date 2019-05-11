use crate::protobuf::EvaluationData;
use crate::Result;
use byteorder::{BigEndian, WriteBytesExt};
use std::collections::HashMap;
use std::io::Write;

#[derive(Debug, Clone)]
pub struct EvalStats {
    pub training_time: f64,
    pub training_accuracy: f64,
    pub validation_accuracy: f64,
    pub test_accuracy: f64,
}
impl EvalStats {
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
    pub epochs: HashMap<u8, Vec<DataPoint>>,
}
impl ModelStats {
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
