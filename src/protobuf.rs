//! See: [model_metrics.proto] file.
//!
//! [model_metrics.proto]: https://github.com/google-research/nasbench/blob/master/nasbench/lib/model_metrics.proto
use protobuf_codec::field::num::{F1, F2, F3, F4, F5, F6};
use protobuf_codec::field::{FieldDecoder, Fields, MaybeDefault, MessageFieldDecoder, Repeated};
use protobuf_codec::message::MessageDecoder;
use protobuf_codec::scalar::{DoubleDecoder, Int32Decoder, StringDecoder};

#[derive(Debug, Clone, PartialEq)]
pub struct ModelMetrics {
    /// Metrics that are evaluated at each checkpoint.
    ///
    /// Each ModelMetrics will
    /// contain multiple EvaluationData messages evaluated at various points during
    /// training, including the initialization before any steps are taken.
    pub evaluation_data_list: Vec<EvaluationData>,

    /// Parameter count of all trainable variables.
    pub trainable_parameters: i32,

    /// Total time for all training and evaluation.
    ///
    /// Mostly used for diagnostic purposes.
    pub total_time: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EvaluationData {
    /// Current epoch at the time of this evaluation.
    pub current_epoch: f64,

    /// Training time in seconds up to this point.
    ///
    /// Does not include evaluation time.
    pub training_time: f64,

    /// Accuracy on a fixed 10,000 images from the train set.
    pub train_accuracy: f64,

    /// Accuracy on a held-out validation set of 10,000 images.
    pub validation_accuracy: f64,

    /// Accuracy on the test set of 10,000 images.
    pub test_accuracy: f64,

    /// Location of checkpoint file.
    ///
    /// Note: checkpoint_path will look like
    /// /path/to/model_dir/model.ckpt-1234 but the actual checkpoint files may have
    /// an extra ".data", ".index", ".meta" suffix. For purposes of loading a
    /// checkpoint file in TensorFlow, the path without the suffix is sufficient.
    /// This field may be left blank because the checkpoint can be programmatically
    /// generated from the model specifications.
    pub checkpoint_path: String,
}

macro_rules! impl_message_decode {
    ($decoder:ty, $item:ty, $map:expr) => {
        impl ::bytecodec::Decode for $decoder {
            type Item = $item;

            fn decode(&mut self, buf: &[u8], eos: ::bytecodec::Eos) -> ::bytecodec::Result<usize> {
                track!(self.inner.decode(buf, eos))
            }
            fn finish_decoding(&mut self) -> ::bytecodec::Result<Self::Item> {
                let item = track!(self.inner.finish_decoding())?;

                $map(item)
            }
            fn is_idle(&self) -> bool {
                self.inner.is_idle()
            }
            fn requiring_bytes(&self) -> ::bytecodec::ByteCount {
                self.inner.requiring_bytes()
            }
        }
        impl ::protobuf_codec::message::MessageDecode for $decoder {}
    };
}

#[derive(Debug, Default)]
pub struct ModelMetricsDecoder {
    inner: MessageDecoder<
        Fields<(
            Repeated<MessageFieldDecoder<F1, EvaluationDataDecoder>, Vec<EvaluationData>>,
            MaybeDefault<FieldDecoder<F2, Int32Decoder>>,
            MaybeDefault<FieldDecoder<F3, DoubleDecoder>>,
        )>,
    >,
}
impl_message_decode!(ModelMetricsDecoder, ModelMetrics, |t: (_, _, _)| {
    Ok(ModelMetrics {
        evaluation_data_list: t.0,
        trainable_parameters: t.1,
        total_time: t.2,
    })
});

#[derive(Debug, Default)]
pub struct EvaluationDataDecoder {
    inner: MessageDecoder<
        Fields<(
            MaybeDefault<FieldDecoder<F1, DoubleDecoder>>,
            MaybeDefault<FieldDecoder<F2, DoubleDecoder>>,
            MaybeDefault<FieldDecoder<F3, DoubleDecoder>>,
            MaybeDefault<FieldDecoder<F4, DoubleDecoder>>,
            MaybeDefault<FieldDecoder<F5, DoubleDecoder>>,
            MaybeDefault<FieldDecoder<F6, StringDecoder>>,
        )>,
    >,
}
impl_message_decode!(EvaluationDataDecoder, EvaluationData, |t: (
    _,
    _,
    _,
    _,
    _,
    _
)| Ok(
    EvaluationData {
        current_epoch: t.0,
        training_time: t.1,
        train_accuracy: t.2,
        validation_accuracy: t.3,
        test_accuracy: t.4,
        checkpoint_path: t.5
    }
));

#[cfg(test)]
mod tests {
    use super::*;
    use base64;
    use bytecodec::DecodeExt;
    use trackable::result::TestResult;

    #[test]
    fn decoder_works() -> TestResult {
        let base64_bytes = "Ci0JAAAAAAAAAAARAAAAAAAAAAAZAAAAoKqquT8hAAAAYJAGuT8pAAAAAGmQuT8KLQkAAAAAAAAAQBEAAADAdJNGQBkAAABgkAa6PyEAAAAg3/K6PykAAACAy7e6PwotCQAAAAAAABBAEQAAACDbeVZAGQAAAEAaZNY/IQAAAGBVVdY/KQAAAODybdY/EIqYigQZAABAhznbeUA=";
        let bytes = track_any_err!(base64::decode(&base64_bytes))?;

        let metrics = track!(ModelMetricsDecoder::default().decode_from_bytes(&bytes))?;

        let expected = ModelMetrics {
            evaluation_data_list: vec![
                EvaluationData {
                    current_epoch: 0.0,
                    training_time: 0.0,
                    train_accuracy: 0.1002604141831398,
                    validation_accuracy: 0.09775640815496445,
                    test_accuracy: 0.09985977411270142,
                    checkpoint_path: "".to_owned(),
                },
                EvaluationData {
                    current_epoch: 2.0,
                    training_time: 45.152000427246094,
                    train_accuracy: 0.10166265815496445,
                    validation_accuracy: 0.10526842623949051,
                    test_accuracy: 0.10436698794364929,
                    checkpoint_path: "".to_owned(),
                },
                EvaluationData {
                    current_epoch: 4.0,
                    training_time: 89.90399932861328,
                    train_accuracy: 0.3498597741127014,
                    validation_accuracy: 0.3489583432674408,
                    test_accuracy: 0.3504607379436493,
                    checkpoint_path: "".to_owned(),
                },
            ],
            trainable_parameters: 8555530,
            total_time: 413.7015450000763,
        };
        assert_eq!(metrics, expected);

        Ok(())
    }
}
