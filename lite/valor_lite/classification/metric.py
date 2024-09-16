from dataclasses import dataclass
from enum import Enum

from valor_lite.schemas import Metric


class MetricType(Enum):
    Accuracy = "Accuracy"
    Precision = "Precision"
    Recall = "Recall"
    F1 = "F1"
    ROCAUC = "ROCAUC"
    PrecisionRecallCurve = "PrecisionRecallCurve"
    DetailedPrecisionRecallCurve = "DetailedPrecisionRecallCurve"
    Counts = "Counts"


@dataclass
class Counts:
    tp: int
    fp: int
    tn: int
    fn: int
    label: tuple[str, str]
    score: float

    @property
    def metric(self) -> Metric:
        return Metric(
            type=type(self).__name__,
            value={
                "tp": self.tp,
                "fp": self.fp,
                "fn": self.fn,
                "tn": self.tn,
            },
            parameters={
                "score": self.score,
                "label": {
                    "key": self.label[0],
                    "value": self.label[1],
                },
            },
        )

    def to_dict(self) -> dict:
        return self.metric.to_dict()


@dataclass
class ClassMetric:
    value: float
    label: tuple[str, str]
    score: float

    @property
    def metric(self) -> Metric:
        return Metric(
            type=type(self).__name__,
            value=self.value,
            parameters={
                "score": self.score,
                "label": {
                    "key": self.label[0],
                    "value": self.label[1],
                },
            },
        )

    def to_dict(self) -> dict:
        return self.metric.to_dict()


class Precision(ClassMetric):
    pass


class Recall(ClassMetric):
    pass


class Accuracy(ClassMetric):
    pass


class F1(ClassMetric):
    pass


@dataclass
class ROCAUC:
    value: float


@dataclass
class PrecisionRecallCurve:
    """
    Interpolated over recalls 0.0, 0.01, ..., 1.0.
    """

    precision: list[float]
    iou: float
    label: tuple[str, str]

    @property
    def metric(self) -> Metric:
        return Metric(
            type=type(self).__name__,
            value=self.precision,
            parameters={
                "iou": self.iou,
                "label": {"key": self.label[0], "value": self.label[1]},
            },
        )

    def to_dict(self) -> dict:
        return self.metric.to_dict()


@dataclass
class DetailedPrecisionRecallPoint:
    score: float
    tp: int
    fp_misclassification: int
    fp_hallucination: int
    fn_misclassification: int
    fn_missing_prediction: int
    tp_examples: list[str]
    fp_misclassification_examples: list[str]
    fp_hallucination_examples: list[str]
    fn_misclassification_examples: list[str]
    fn_missing_prediction_examples: list[str]

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "tp": self.tp,
            "fp_misclassification": self.fp_misclassification,
            "fp_hallucination": self.fp_hallucination,
            "fn_misclassification": self.fn_misclassification,
            "fn_missing_prediction": self.fn_missing_prediction,
            "tp_examples": self.tp_examples,
            "fp_misclassification_examples": self.fp_misclassification_examples,
            "fp_hallucination_examples": self.fp_hallucination_examples,
            "fn_misclassification_examples": self.fn_misclassification_examples,
            "fn_missing_prediction_examples": self.fn_missing_prediction_examples,
        }


@dataclass
class DetailedPrecisionRecallCurve:
    iou: float
    value: list[DetailedPrecisionRecallPoint]
    label: tuple[str, str]

    def to_dict(self) -> dict:
        return {
            "value": [pt.to_dict() for pt in self.value],
            "iou": self.iou,
            "label": {
                "key": self.label[0],
                "value": self.label[1],
            },
            "type": "DetailedPrecisionRecallCurve",
        }
