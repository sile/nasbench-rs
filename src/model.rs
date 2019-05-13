use crate::protobuf::EvaluationData;
use crate::Result;
use byteorder::{BigEndian, ByteOrder, ReadBytesExt, WriteBytesExt};
use md5;
use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::str::FromStr;
use trackable::error::{Failed, Failure};

/// Model specification given adjacency matrix and operations (a.k.a. "module").
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ModelSpec {
    /// Operations.
    pub ops: Vec<Op>,

    /// Adjacency matrix.
    pub adjacency: AdjacencyMatrix,
}
impl ModelSpec {
    pub(crate) fn module_hash(&self) -> u128 {
        let dim = self.ops.len();

        let mut hashes = Vec::with_capacity(dim);
        for (row, op) in self.ops.iter().enumerate() {
            let in_edges = self.adjacency.in_edges(row);
            let out_edges = self.adjacency.out_edges(row);
            let s = format!("({}, {}, {})", out_edges, in_edges, op.as_index());
            hashes.push(format!("{:032x}", md5::compute(s.as_bytes())));
        }

        for _ in 0..dim {
            let mut new_hashes = Vec::with_capacity(dim);
            for v in 0..dim {
                let mut in_neighbors = (0..dim)
                    .filter(|&w| self.adjacency.is_adjacent(w, v))
                    .map(|w| hashes[w].as_str())
                    .collect::<Vec<_>>();
                let mut out_neighbors = (0..dim)
                    .filter(|&w| self.adjacency.is_adjacent(v, w))
                    .map(|w| hashes[w].as_str())
                    .collect::<Vec<_>>();
                in_neighbors.sort();
                out_neighbors.sort();

                let s = format!(
                    "{}|{}|{}",
                    in_neighbors.join(""),
                    out_neighbors.join(""),
                    hashes[v]
                );
                new_hashes.push(format!("{:032x}", md5::compute(s.as_bytes())));
            }
            hashes = new_hashes;
        }

        hashes.sort();
        let temp = hashes
            .iter()
            .map(|h| format!("'{}'", h))
            .collect::<Vec<_>>();
        let fingerprint = format!("[{}]", temp.join(", "));
        BigEndian::read_u128(&md5::compute(fingerprint.as_bytes()).0)
    }

    pub(crate) fn from_reader<R: Read>(mut reader: R) -> Result<Self> {
        let adjacency = track!(AdjacencyMatrix::from_reader(&mut reader))?;

        let len = track_any_err!(reader.read_u8())? as usize;
        let mut ops = Vec::with_capacity(len);
        for _ in 0..len {
            let op = match track_any_err!(reader.read_u8())? {
                0 => Op::Input,
                1 => Op::Conv1x1,
                2 => Op::Conv3x3,
                3 => Op::MaxPool3x3,
                4 => Op::Output,
                n => track_panic!(Failed, "Unknown operation number: {}", n),
            };
            ops.push(op);
        }

        Ok(Self { adjacency, ops })
    }

    pub(crate) fn to_writer<W: Write>(&self, mut writer: W) -> Result<()> {
        track!(self.adjacency.to_writer(&mut writer))?;

        track_any_err!(writer.write_u8(self.ops.len() as u8))?;
        for op in &self.ops {
            track_any_err!(writer.write_u8(*op as u8))?;
        }

        Ok(())
    }
}

/// Operation.
///
/// # Examples
///
/// ```
/// use nasbench::Op;
///
/// assert_eq!("input".parse().ok(), Some(Op::Input));
/// assert_eq!("conv1x1-bn-relu".parse().ok(), Some(Op::Conv1x1));
/// assert_eq!("conv3x3-bn-relu".parse().ok(), Some(Op::Conv3x3));
/// assert_eq!("maxpool3x3".parse().ok(), Some(Op::MaxPool3x3));
/// assert_eq!("output".parse().ok(), Some(Op::Output));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Op {
    /// Input tensor.
    Input,

    /// 1x1 convolution -> batch-norm -> ReLU.
    Conv1x1,

    /// 3x3 convolution -> batch-norm -> ReLU.
    Conv3x3,

    /// 3x3 max-pool.
    MaxPool3x3,

    /// Output tensor.
    Output,
}
impl Op {
    fn as_index(&self) -> isize {
        match self {
            Op::Input => -1,
            Op::Conv3x3 => 0,
            Op::Conv1x1 => 1,
            Op::MaxPool3x3 => 2,
            Op::Output => -2,
        }
    }
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

/// Adjacency (upper-triangular) matrix of a module.
///
/// # Examples
///
/// ```
/// use nasbench::AdjacencyMatrix;
/// # use trackable::result::TopLevelResult;
///
/// # fn main() -> TopLevelResult {
/// let matrix0 = AdjacencyMatrix::new(vec![
///     vec![false, true, false, false, true, true, false],
///     vec![false, false, true, false, false, false, false],
///     vec![false, false, false, true, false, false, true],
///     vec![false, false, false, false, false, true, false],
///     vec![false, false, false, false, false, true, false],
///     vec![false, false, false, false, false, false, true],
///     vec![false, false, false, false, false, false, false]
/// ])?;
/// assert_eq!(matrix0.dimension(), 7);
///
/// let matrix1 = "0100110001000000010010000010000001000000010000000".parse()?;
/// assert_eq!(matrix0, matrix1);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AdjacencyMatrix {
    triangular_matrix: Vec<bool>,
}
impl AdjacencyMatrix {
    // TODO: has_edge
    fn is_adjacent(&self, row: usize, column: usize) -> bool {
        if column <= row {
            return false;
        }

        let dim = self.dimension();

        let mut width = dim - 1;
        let mut offset = 0;
        for _ in 0..row {
            offset += width;
            width -= 1;
        }

        self.triangular_matrix[offset..][..width]
            .into_iter()
            .nth(column - row - 1)
            .cloned()
            .unwrap_or_else(|| unreachable!())
    }

    fn in_edges(&self, row: usize) -> usize {
        let dim = self.dimension();
        (0..dim) // TODO: (0..row)
            .filter(|&column| self.is_adjacent(column, row))
            .count()
    }

    fn out_edges(&self, row: usize) -> usize {
        let mut size = self.dimension() - 1;
        let mut offset = 0;
        for _ in 0..row {
            offset += size;
            size -= 1;
        }
        self.triangular_matrix[offset..][..size]
            .iter()
            .filter(|b| **b)
            .count()
    }

    /// Makes a new `AdjacencyMatrix` instance.
    pub fn new(matrix: Vec<Vec<bool>>) -> Result<Self> {
        track_assert!(!matrix.is_empty(), Failed);
        let dim = matrix.len();
        let mut triangular_matrix = Vec::with_capacity((1..dim).sum());
        for (i, row) in matrix.into_iter().enumerate() {
            track_assert_eq!(row.len(), dim, Failed);

            for (j, adjacent) in row.into_iter().enumerate() {
                if j <= i {
                    track_assert!(!adjacent, Failed; i, j);
                    continue;
                }
                triangular_matrix.push(adjacent);
            }
        }
        Ok(Self { triangular_matrix })
    }

    /// Returns the dimension of this matrix.
    pub fn dimension(&self) -> usize {
        let mut n = 0;
        for dim in 1.. {
            if n >= self.triangular_matrix.len() {
                return dim;
            }
            n += dim;
        }
        unreachable!();
    }

    pub(crate) fn from_reader<R: Read>(mut reader: R) -> Result<Self> {
        let len = track_any_err!(reader.read_u8())? as usize;
        let mut triangular_matrix = Vec::with_capacity(len);
        for _ in 0..len {
            triangular_matrix.push(track_any_err!(reader.read_u8())? == 1);
        }
        Ok(Self { triangular_matrix })
    }

    pub(crate) fn to_writer<W: Write>(&self, mut writer: W) -> Result<()> {
        let len = self.triangular_matrix.len();
        track_any_err!(writer.write_u8(len as u8))?;
        for &b in &self.triangular_matrix {
            track_any_err!(writer.write_u8(b as u8))?;
        }
        Ok(())
    }
}
impl FromStr for AdjacencyMatrix {
    type Err = Failure;

    fn from_str(s: &str) -> Result<Self> {
        let dim = (s.len() as f64).sqrt() as usize;
        track_assert_eq!(dim * dim, s.len(), Failed, "Not a matrix: {:?}", s);

        let mut matrix = vec![vec![false; dim]; dim];
        for (i, row) in matrix.iter_mut().enumerate() {
            for (j, v) in row.iter_mut().enumerate() {
                *v = s.as_bytes()[i * dim + j] == b'1';
            }
        }

        track!(Self::new(matrix), "Not an upper triangular matrix; {:?}", s)
    }
}

/// Model statistics.
#[derive(Debug, Default, PartialEq)]
pub struct ModelStats {
    /// Number of trainable parameters in the model.
    pub trainable_parameters: u32,

    /// Map from epoch number to evaluation metrics at that epoch.
    ///
    /// Because each model has evaluated multiple times, each epoch is associated with multiple metrics.
    pub epochs: BTreeMap<u8, Vec<EpochStats>>,
}
impl ModelStats {
    pub(crate) fn from_reader<R: Read>(mut reader: R) -> Result<Self> {
        let trainable_parameters = track_any_err!(reader.read_u32::<BigEndian>())?;

        let len = track_any_err!(reader.read_u8())? as usize;
        let mut epochs = BTreeMap::new();
        for _ in 0..len {
            let epoch_num = track_any_err!(reader.read_u8())?;

            let len = track_any_err!(reader.read_u8())? as usize;
            let mut stats_list = Vec::with_capacity(len);
            for _ in 0..len {
                stats_list.push(track!(EpochStats::from_reader(&mut reader))?);
            }

            epochs.insert(epoch_num, stats_list);
        }

        Ok(Self {
            trainable_parameters,
            epochs,
        })
    }

    pub(crate) fn to_writer<W: Write>(&self, mut writer: W) -> Result<()> {
        track_any_err!(writer.write_u32::<BigEndian>(self.trainable_parameters))?;

        track_any_err!(writer.write_u8(self.epochs.len() as u8))?;
        for (epoch_num, stats_list) in &self.epochs {
            track_any_err!(writer.write_u8(*epoch_num))?;

            track_any_err!(writer.write_u8(stats_list.len() as u8))?;
            for s in stats_list {
                track!(s.to_writer(&mut writer))?;
            }
        }

        Ok(())
    }
}

/// Epoch statistics.
#[derive(Debug, PartialEq)]
pub struct EpochStats {
    /// Evaluation metrics at the half-way point of training.
    pub halfway: EvaluationMetrics,

    /// Evaluation metrics at the end of training.
    pub complete: EvaluationMetrics,
}
impl EpochStats {
    pub(crate) fn from_reader<R: Read>(mut reader: R) -> Result<Self> {
        let halfway = track!(EvaluationMetrics::from_reader(&mut reader))?;
        let complete = track!(EvaluationMetrics::from_reader(&mut reader))?;
        Ok(Self { halfway, complete })
    }

    pub(crate) fn to_writer<W: Write>(&self, mut writer: W) -> Result<()> {
        track!(self.halfway.to_writer(&mut writer))?;
        track!(self.complete.to_writer(&mut writer))?;
        Ok(())
    }
}

/// Evaluation metrics.
#[derive(Debug, Clone, PartialEq)]
pub struct EvaluationMetrics {
    /// The total training time in seconds up to this point.
    pub training_time: f64,

    /// Training accuracy.
    pub training_accuracy: f64,

    /// Validation accuracy.
    pub validation_accuracy: f64,

    /// Test accuracy.
    pub test_accuracy: f64,
}
impl EvaluationMetrics {
    pub(crate) fn from_reader<R: Read>(mut reader: R) -> Result<Self> {
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

    pub(crate) fn to_writer<W: Write>(&self, mut writer: W) -> Result<()> {
        track_any_err!(writer.write_f64::<BigEndian>(self.training_time))?;
        track_any_err!(writer.write_f64::<BigEndian>(self.training_accuracy))?;
        track_any_err!(writer.write_f64::<BigEndian>(self.validation_accuracy))?;
        track_any_err!(writer.write_f64::<BigEndian>(self.test_accuracy))?;
        Ok(())
    }
}
impl From<EvaluationData> for EvaluationMetrics {
    fn from(f: EvaluationData) -> Self {
        Self {
            training_time: f.training_time,
            training_accuracy: f.train_accuracy,
            validation_accuracy: f.validation_accuracy,
            test_accuracy: f.test_accuracy,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trackable::result::TopLevelResult;

    #[test]
    fn op_works() {
        assert_eq!("input".parse().ok(), Some(Op::Input));
        assert_eq!("conv1x1-bn-relu".parse().ok(), Some(Op::Conv1x1));
        assert_eq!("conv3x3-bn-relu".parse().ok(), Some(Op::Conv3x3));
        assert_eq!("maxpool3x3".parse().ok(), Some(Op::MaxPool3x3));
        assert_eq!("output".parse().ok(), Some(Op::Output));
    }

    #[test]
    fn adjacency_matrix_works() -> TopLevelResult {
        let matrix0 = track!(AdjacencyMatrix::new(vec![
            vec![false, true, false, false, true, true, false],
            vec![false, false, true, false, false, false, false],
            vec![false, false, false, true, false, false, true],
            vec![false, false, false, false, false, true, false],
            vec![false, false, false, false, false, true, false],
            vec![false, false, false, false, false, false, true],
            vec![false, false, false, false, false, false, false]
        ]))?;
        assert_eq!(matrix0.dimension(), 7);

        let matrix1 = track!("0100110001000000010010000010000001000000010000000".parse())?;
        assert_eq!(matrix0, matrix1);

        Ok(())
    }
}
