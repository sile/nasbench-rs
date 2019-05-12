#[macro_use]
extern crate trackable;

use nasbench::model::Op;
use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;
use structopt::StructOpt;
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
        adjacency: Vec<u8>,
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
        Opt::Query { dataset, .. } => {
            let nasbench = track!(nasbench::api::NasBench::new(dataset))?;
        }
    }
    Ok(())
}
