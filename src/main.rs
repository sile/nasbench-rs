#[macro_use]
extern crate trackable;

use nasbench::{AdjacencyMatrix, ModelSpec, NasBench, Op};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::PathBuf;
use structopt::StructOpt;
use trackable::error::Failed;
use trackable::result::MainResult;

#[derive(Debug, StructOpt)]
#[structopt(rename_all = "kebab-case")]
enum Opt {
    #[structopt(about = "Converts tfrecord format dataset to more compact binary format")]
    Convert {
        tfrecord_format_dataset_path: PathBuf,
        binary_format_dataset_path: PathBuf,
    },

    #[structopt(about = "Queris evaluation metrics of a model")]
    Query {
        dataset_path: PathBuf,

        #[structopt(long, default_value = "108")]
        epochs: u8,

        #[structopt(long)]
        ops: Vec<Op>,

        #[structopt(long)]
        adjacency: AdjacencyMatrix,

        #[structopt(long)]
        stop_halfway: bool,

        #[structopt(long, default_value = "0")]
        sample_index: usize,
    },
}

fn main() -> MainResult {
    let opt = Opt::from_args();
    match opt {
        Opt::Convert {
            tfrecord_format_dataset_path,
            binary_format_dataset_path,
        } => {
            let file = track_any_err!(File::open(&tfrecord_format_dataset_path); tfrecord_format_dataset_path)?;
            let nasbench = track!(NasBench::from_tfrecord_reader(BufReader::new(file)))?;

            let file = track_any_err!(File::create(binary_format_dataset_path))?;
            track!(nasbench.to_writer(BufWriter::new(file)))?
        }
        Opt::Query {
            dataset_path,
            epochs,
            ops,
            adjacency,
            stop_halfway,
            sample_index,
        } => {
            let nasbench = track!(NasBench::new(dataset_path))?;
            let model_spec = track!(ModelSpec::new(ops, adjacency))?;
            let model_stats =
                track_assert_some!(nasbench.models().get(&model_spec), Failed, "Unknown model");
            let epoch_stats_list =
                track_assert_some!(model_stats.epochs.get(&epochs), Failed, "Unknown epochs");
            let epoch_stats =
                track_assert_some!(epoch_stats_list.get(sample_index), Failed, "Out of range");
            if stop_halfway {
                println!("{:?}", epoch_stats.halfway);
            } else {
                println!("{:?}", epoch_stats.complete);
            }
        }
    }
    Ok(())
}
