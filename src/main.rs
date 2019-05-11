#[macro_use]
extern crate trackable;

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
    }
    Ok(())
}
