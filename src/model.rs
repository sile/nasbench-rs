use crate::protobuf::EvaluationData;
use crate::Result;
use byteorder::{BigEndian, ByteOrder, ReadBytesExt, WriteBytesExt};
use md5;
use std::collections::BTreeMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::io::{Read, Write};
use std::str::FromStr;
use trackable::error::{Failed, Failure};

/// Model specification given adjacency matrix and operations (a.k.a. "module").
///
/// Note that two instances of `ModuleSpec` are regarded as the same
/// if their structures are semantically equivalent (see below).
///
/// ```rust
/// use nasbench::{ModelSpec, Op};
/// # use trackable::result::TopLevelResult;
///
/// # fn main() -> TopLevelResult {
/// let model0 = ModelSpec::new(vec![Op::Input, Op::Output], "0100".parse()?)?;
/// let model1 = ModelSpec::new(
///     vec![Op::Input, Op::Conv1x1, Op::Output],
///     "001000000".parse()?,
/// )?;
///
/// assert_eq!(model0, model1);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct ModelSpec {
    ops: Vec<Op>,
    adjacency: AdjacencyMatrix,
    module_hash: u128,
}
impl ModelSpec {
    /// Makes a new `ModelSpec` instance.
    pub fn new(mut ops: Vec<Op>, mut adjacency: AdjacencyMatrix) -> Result<Self> {
        track_assert_eq!(ops.len(), adjacency.dimension(), Failed);
        track_assert!(adjacency.dimension() >= 2, Failed);

        Self::prune(&mut ops, &mut adjacency);
        let module_hash = Self::module_hash(&ops, &adjacency);
        Ok(Self {
            ops,
            adjacency,
            module_hash,
        })
    }

    pub(crate) fn with_module_hash(
        mut ops: Vec<Op>,
        mut adjacency: AdjacencyMatrix,
        module_hash: u128,
    ) -> Self {
        Self::prune(&mut ops, &mut adjacency);
        Self {
            ops,
            adjacency,
            module_hash,
        }
    }

    /// Returns a reference to the operations of this model.
    pub fn ops(&self) -> &[Op] {
        &self.ops
    }

    /// Returns a reference to the adjacency matrix of this model.
    pub fn adjacency(&self) -> &AdjacencyMatrix {
        &self.adjacency
    }

    fn prune(ops: &mut Vec<Op>, adjacency: &mut AdjacencyMatrix) {
        let mut deleted = true;
        while deleted {
            deleted = false;

            for row in 1..adjacency.dimension() - 1 {
                let in_edges = adjacency.in_edges(row);
                if in_edges == 0 {
                    deleted = true;
                    ops.remove(row);
                    adjacency.remove(row);
                    break;
                }

                let out_edges = adjacency.out_edges(row);
                if out_edges == 0 {
                    deleted = true;
                    ops.remove(row);
                    adjacency.remove(row);
                    break;
                }
            }
        }
    }

    fn module_hash(ops: &[Op], adjacency: &AdjacencyMatrix) -> u128 {
        let dim = ops.len();

        let mut hashes = Vec::with_capacity(dim);
        for (row, op) in ops.iter().enumerate() {
            let in_edges = adjacency.in_edges(row);
            let out_edges = adjacency.out_edges(row);
            let s = format!("({}, {}, {})", out_edges, in_edges, op.to_hash_index());
            hashes.push(format!("{:032x}", md5::compute(s.as_bytes())));
        }

        for _ in 0..dim {
            let mut new_hashes = Vec::with_capacity(dim);
            for v in 0..dim {
                let mut in_neighbors = (0..dim)
                    .filter(|&w| adjacency.has_edge(w, v))
                    .map(|w| hashes[w].as_str())
                    .collect::<Vec<_>>();
                let mut out_neighbors = (0..dim)
                    .filter(|&w| adjacency.has_edge(v, w))
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
        let hashes = hashes
            .iter()
            .map(|h| format!("'{}'", h))
            .collect::<Vec<_>>();
        let fingerprint = format!("[{}]", hashes.join(", "));
        BigEndian::read_u128(&md5::compute(fingerprint.as_bytes()).0)
    }

    pub(crate) fn from_reader<R: Read>(mut reader: R) -> Result<Self> {
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

        let adjacency = track!(AdjacencyMatrix::from_reader(&mut reader))?;
        let module_hash = track_any_err!(reader.read_u128::<BigEndian>())?;

        Ok(Self {
            ops,
            adjacency,
            module_hash,
        })
    }

    pub(crate) fn to_writer<W: Write>(&self, mut writer: W) -> Result<()> {
        track_any_err!(writer.write_u8(self.ops.len() as u8))?;
        for op in &self.ops {
            track_any_err!(writer.write_u8(*op as u8))?;
        }

        track!(self.adjacency.to_writer(&mut writer))?;
        track_any_err!(writer.write_u128::<BigEndian>(self.module_hash))?;

        Ok(())
    }
}
impl PartialEq for ModelSpec {
    fn eq(&self, other: &Self) -> bool {
        self.module_hash == other.module_hash
    }
}
impl Eq for ModelSpec {}
impl Hash for ModelSpec {
    fn hash<H: Hasher>(&self, h: &mut H) {
        self.module_hash.hash(h);
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
    fn to_hash_index(&self) -> isize {
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
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct AdjacencyMatrix {
    dim: u8,
    triangle: u32,
}
impl AdjacencyMatrix {
    /// Makes a new `AdjacencyMatrix` instance.
    pub fn new(matrix: Vec<Vec<bool>>) -> Result<Self> {
        let dim = matrix.len();
        track_assert_ne!(dim, 0, Failed);
        track_assert!(dim <= 7, Failed; dim);

        let mut triangle = 0;
        let mut offset = 0;
        for (i, row) in matrix.into_iter().enumerate() {
            track_assert_eq!(row.len(), dim, Failed);

            for (j, adjacent) in row.into_iter().enumerate() {
                if j <= i {
                    track_assert!(!adjacent, Failed; i, j);
                    continue;
                }

                offset += 1;
                if !adjacent {
                    continue;
                }

                triangle |= 1 << (offset - 1);
            }
        }

        let dim = dim as u8;
        Ok(Self { dim, triangle })
    }

    /// Returns the dimension of this matrix.
    pub fn dimension(&self) -> usize {
        usize::from(self.dim)
    }

    fn from_reader<R: Read>(mut reader: R) -> Result<Self> {
        let dim = track_any_err!(reader.read_u8())?;
        let triangle = track_any_err!(reader.read_u32::<BigEndian>())?;
        Ok(Self { dim, triangle })
    }

    fn to_writer<W: Write>(&self, mut writer: W) -> Result<()> {
        track_any_err!(writer.write_u8(self.dim))?;
        track_any_err!(writer.write_u32::<BigEndian>(self.triangle))?;
        Ok(())
    }

    fn remove(&mut self, row: usize) {
        let mut triangle = 0;
        let mut offset = 0;
        for i in (0..self.dimension()).filter(|&i| i != row) {
            for j in (i + 1..self.dimension()).filter(|&j| j != row) {
                offset += 1;
                if !self.has_edge(i, j) {
                    continue;
                }

                triangle |= 1 << (offset - 1);
            }
        }

        self.dim -= 1;
        self.triangle = triangle;
    }

    fn has_edge(&self, row: usize, column: usize) -> bool {
        if column <= row {
            return false;
        }

        let offset = match self.dim {
            7 => &[0, 6, 11, 15, 18, 20, 21][..],
            6 => &[0, 5, 9, 12, 14, 15][..],
            5 => &[0, 4, 7, 9, 10][..],
            4 => &[0, 3, 5, 6][..],
            3 => &[0, 2, 1][..],
            2 => &[0, 1][..],
            1 => &[0][..],
            _ => {
                unreachable!("dim={}", self.dim);
            }
        };
        let i = offset[row] + column - row - 1;
        (self.triangle & (1 << i)) != 0
    }

    fn in_edges(&self, row: usize) -> usize {
        (0..row)
            .filter(|&column| self.has_edge(column, row))
            .count()
    }

    fn out_edges(&self, row: usize) -> usize {
        (row + 1..self.dimension())
            .filter(|&column| self.has_edge(row, column))
            .count()
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
impl fmt::Debug for AdjacencyMatrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "AdjacencyMatrix(0b")?;
        for row in 0..self.dimension() {
            for column in 0..self.dimension() {
                write!(f, "{}", self.has_edge(row, column) as u8)?;
            }
            if row != self.dimension() - 1 {
                write!(f, "_")?;
            }
        }
        write!(f, ")")?;
        Ok(())
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
    fn from_reader<R: Read>(mut reader: R) -> Result<Self> {
        let halfway = track!(EvaluationMetrics::from_reader(&mut reader))?;
        let complete = track!(EvaluationMetrics::from_reader(&mut reader))?;
        Ok(Self { halfway, complete })
    }

    fn to_writer<W: Write>(&self, mut writer: W) -> Result<()> {
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
    fn from_reader<R: Read>(mut reader: R) -> Result<Self> {
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

    fn to_writer<W: Write>(&self, mut writer: W) -> Result<()> {
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
    fn model_spec_works() -> TopLevelResult {
        let model0 = ModelSpec::new(vec![Op::Input, Op::Output], "0100".parse()?)?;
        let model1 = ModelSpec::new(
            vec![Op::Input, Op::Conv1x1, Op::Output],
            "001000000".parse()?,
        )?;
        assert_eq!(model0, model1);

        let model2 = ModelSpec::new(
            vec![Op::Input, Op::Conv3x3, Op::MaxPool3x3, Op::Output],
            "0101001000010000".parse()?,
        )?;
        let model3 = ModelSpec::new(
            vec![
                Op::Input,
                Op::Conv1x1,
                Op::Conv3x3,
                Op::MaxPool3x3,
                Op::Conv3x3,
                Op::Output,
            ],
            "001001000000000100000001000000000000".parse()?,
        )?;
        assert_eq!(model2, model3);

        let model4 = ModelSpec::new(vec![Op::Input, Op::Output], "0000".parse()?)?;
        let model5 = ModelSpec::new(
            vec![
                Op::Input,
                Op::Conv1x1,
                Op::MaxPool3x3,
                Op::Conv3x3,
                Op::Output,
            ],
            "0000000000000000000000000".parse()?,
        )?;
        assert_eq!(model4, model5);

        let model6 = ModelSpec::new(vec![Op::Input, Op::Output], "0100".parse()?)?;
        let model7 = ModelSpec::new(
            vec![Op::Input, Op::Conv3x3, Op::Conv1x1, Op::Output],
            "0111000000000000".parse()?,
        )?;
        assert_eq!(model6, model7);

        Ok(())
    }

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
        let original_matrix = vec![
            vec![false, true, false, false, true, true, false],
            vec![false, false, true, false, false, false, false],
            vec![false, false, false, true, false, false, true],
            vec![false, false, false, false, false, true, false],
            vec![false, false, false, false, false, true, false],
            vec![false, false, false, false, false, false, true],
            vec![false, false, false, false, false, false, false],
        ];

        let matrix0 = track!(AdjacencyMatrix::new(original_matrix.clone()))?;
        assert_eq!(matrix0.dimension(), 7);

        let matrix1 = track!("0100110001000000010010000010000001000000010000000".parse())?;
        assert_eq!(matrix0, matrix1);

        for row in 0..original_matrix.len() {
            for column in 0..original_matrix.len() {
                assert_eq!(
                    matrix0.has_edge(row, column),
                    original_matrix[row][column],
                    "row={}, column={}",
                    row,
                    column
                );
            }
        }
        Ok(())
    }
}
