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
    TP = "TruePositiveCount"
    FP = "FalsePositiveCount"
    FN = "FalseNegativeCount"
    TN = "TrueNegativeCount"


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
                "tn": self.tn,
                "fn": self.fn,
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
