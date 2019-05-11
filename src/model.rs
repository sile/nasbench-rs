use crate::protobuf::EvaluationData;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct EvalStats {
    pub training_time: f64,
    pub training_accuracy: f64,
    pub validation_accuracy: f64,
    pub test_accuracy: f64,
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

#[derive(Debug)]
pub struct DataPoint {
    pub halfway: EvalStats,
    pub complete: EvalStats,
}

// a.k.a. module
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ModelSpec {
    pub adjacency: Vec<Vec<bool>>, // TODO: AdjacencyMatrix (upper triangular)
    pub operations: Vec<Op>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Op {
    Input,
    Conv1x1,
    Conv3x3,
    MaxPool3x3,
    Output,
}
