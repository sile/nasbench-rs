#[macro_use]
extern crate trackable;

use std::path::PathBuf;
use structopt::StructOpt;
use trackable::result::MainResult;

#[derive(Debug, StructOpt)]
struct Opt {
    dataset: PathBuf,
}

fn main() -> MainResult {
    let opt = Opt::from_args();
    let nasbench = track!(nasbench::api::NasBench::from_tfrecord_file(&opt.dataset))?;
    Ok(())
}
