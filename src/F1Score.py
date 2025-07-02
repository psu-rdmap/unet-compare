from keras.src.metrics.metric import Metric
from keras.src.metrics import metrics_utils
from keras.src import initializers
from keras.src import ops
from keras.src.utils.python_utils import to_list

class F1ScoreBinSeg(Metric):
    """Computes the F1-Score for binary segmentation like Precision and Recall for consistency."""

    def __init__(
        self, thresholds=None, top_k=None, class_id=None, name=None, dtype=None
    ):
        super().__init__(name=name, dtype=dtype)
        # Metric should be maximized during optimization.
        self._direction = "up"

        self.init_thresholds = thresholds
        self.top_k = top_k
        self.class_id = class_id

        default_threshold = 0.5 if top_k is None else metrics_utils.NEG_INF
        self.thresholds = metrics_utils.parse_init_thresholds(
            thresholds, default_threshold=default_threshold
        )
        self._thresholds_distributed_evenly = (
            metrics_utils.is_evenly_distributed_thresholds(self.thresholds)
        )
        self.true_positives = self.add_variable(
            shape=(len(self.thresholds),),
            initializer=initializers.Zeros(),
            name="true_positives",
        )
        self.false_positives = self.add_variable(
            shape=(len(self.thresholds),),
            initializer=initializers.Zeros(),
            name="false_positives",
        )
        self.false_negatives = self.add_variable(
            shape=(len(self.thresholds),),
            initializer=initializers.Zeros(),
            name="false_negatives",
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates true positive, false positive, and false negative statistics."""

        metrics_utils.update_confusion_matrix_variables(
            {
                metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,  # noqa: E501
                metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.true_positives,
                metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self.false_negatives,  # noqa: E501
            },
            y_true,
            y_pred,
            thresholds=self.thresholds,
            thresholds_distributed_evenly=self._thresholds_distributed_evenly,
            top_k=self.top_k,
            class_id=self.class_id,
            sample_weight=sample_weight,
        )

    def result(self):
        """Calculate F1-Score"""
        true_positives_double = ops.multiply(2.0, self.true_positives)
        result = ops.divide_no_nan(
            true_positives_double,
            ops.add(true_positives_double, ops.add(self.false_positives, self.false_negatives)),
        )
        return result[0] if len(self.thresholds) == 1 else result

    def reset_state(self):
        num_thresholds = len(to_list(self.thresholds))
        self.true_positives.assign(ops.zeros((num_thresholds,)))
        self.false_positives.assign(ops.zeros((num_thresholds,)))
        self.false_negatives.assign(ops.zeros((num_thresholds,)))

    def get_config(self):
        config = {
            "thresholds": self.init_thresholds,
            "top_k": self.top_k,
            "class_id": self.class_id,
        }
        base_config = super().get_config()
        return {**base_config, **config}