#[macro_use]
extern crate trackable;

use nasbench::model::{AdjacencyMatrix, ModelSpec, Op};
use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;
use structopt::StructOpt;
use trackable::error::Failed;
use trackable::result::MainResult;

#[derive(Debug, StructOpt)]
#[structopt(rename_all = "kebab-case")]
enum Opt {
    Convert {
        tfrecord_format_dataset: PathBuf,
        binary_format_dataset: PathBuf,
    },
    Query {
        dataset: PathBuf,

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
            tfrecord_format_dataset,
            binary_format_dataset,
        } => {
            let nasbench = track!(nasbench::api::NasBench::from_tfrecord_file(
                tfrecord_format_dataset
            ))?;

            let file = track_any_err!(File::create(binary_format_dataset))?;
            track!(nasbench.to_writer(BufWriter::new(file)))?
        }
        Opt::Query {
            dataset,
            epochs,
            ops,
            adjacency,
            stop_halfway,
            sample_index,
        } => {
            let nasbench = track!(nasbench::api::NasBench::new(dataset))?;
            let model_spec = ModelSpec { ops, adjacency };
            let model_stats =
                track_assert_some!(nasbench.models().get(&model_spec), Failed, "Unknown model");
            let epoch_stats =
                track_assert_some!(model_stats.epochs.get(&epochs), Failed, "Unknown epochs");
            let data_point =
                track_assert_some!(epoch_stats.get(sample_index), Failed, "Out of range");
            if stop_halfway {
                println!("{:?}", data_point.halfway);
            } else {
                println!("{:?}", data_point.complete);
            }
        }
    }
    Ok(())
}
