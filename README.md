nasbench
=========

[![nasbench](https://img.shields.io/crates/v/nasbench.svg)](https://crates.io/crates/nasbench)
[![Documentation](https://docs.rs/nasbench/badge.svg)](https://docs.rs/nasbench)
[![Build Status](https://travis-ci.org/sile/nasbench.svg?branch=master)](https://travis-ci.org/sile/nasbench)
[![Code Coverage](https://codecov.io/gh/sile/nasbench/branch/master/graph/badge.svg)](https://codecov.io/gh/sile/nasbench/branch/master)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A Rust port of [google-research/nasbench][NASBench].

[NASBench]: https://github.com/google-research/nasbench

Motivations
-----------

Of course, the primary purpose of this crate is to make [NASBench] dataset available in Rust.
Besides, another aim is to reduce dataset loading time.
By using a compact binary data format, this crate can reduce the loading time drastically.
For example, on my laptop, [google-research/nasbench][NASBench] requires about 200 seconds
for loading the full dataset. By contrast, this crate only needs a few seconds to complete the loading.

Examples
--------

First of all, you have to convert a NASBench dataset to this crate's format as follows:
```console
$ wget https://storage.googleapis.com/nasbench/nasbench_full.tfrecord
$ nasbench nasbench_full.tfrecord nasbench_full.bin
$ ls -lh
-rw-rw-rw- 1 foo foo 328M May 12 16:47 nasbench_full.bin
-rw-rw-rw- 1 foo foo 2.0G May 12 16:45 nasbench_full.tfrecord
```

Then, you can query the evaluation metrics associated with a model (`ops` and `adjacency`):
```console
$ nasbench query nasbench_full.bin \
                 --adjacency 0100110001000000010010000010000001000000010000000 \
                 --ops input conv3x3-bn-relu maxpool3x3 conv3x3-bn-relu \
                       conv3x3-bn-relu conv1x1-bn-relu output
EvaluationMetrics {
    training_time: 1769.1279296875,
    training_accuracy: 1.0,
    validation_accuracy: 0.9241786599159241,
    test_accuracy: 0.9211738705635071
}
```

Rust code corresponded to the above command:
```rust
use nasbench::{AdjacencyMatrix, ModelSpec, NasBench, Op};

// Loads the dataset.
let nasbench = NasBench::new("nasbench_full.bin")?;

// Queries a model.
let ops = vec![Op::Input, Op::Conv3x3, Op::MaxPool3x3, Op::Conv3x3,
               Op::Conv3x3, Op::Conv1x1, Op::Output];
let adjacency = "0100110001000000010010000010000001000000010000000".parse()?;
let model_spec = ModelSpec{ops, adjacency};
println!("{:?}", nasbench.models().get(&model_spec));
```

Limitations
------------

[google-research/nasbench][NASBench] provides `NASBench.evaluate()` method to train and evaluate
models from scratch, but this crate does not.

References
----------

- [NAS-Bench-101: Towards Reproducible Neural Architecture Search](https://arxiv.org/abs/1902.09635)
