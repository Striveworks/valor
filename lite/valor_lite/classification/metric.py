from dataclasses import dataclass
from enum import Enum

from valor_lite.schemas import Metric


class MetricType(Enum):
    Counts = "Counts"
    ROCAUC = "ROCAUC"
    mROCAUC = "mROCAUC"
    DetailedCounts = "DetailedCounts"
    Precision = "Precision"
    Recall = "Recall"
    Accuracy = "Accuracy"
    F1 = "F1"
    ConfusionMatrix = "ConfusionMatrix"


@dataclass
class Counts:
    tp: list[int]
    fp: list[int]
    fn: list[int]
    tn: list[int]
    score_thresholds: list[float]
    hardmax: bool
    label: tuple[str, str]

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
                "score_thresholds": self.score_thresholds,
                "hardmax": self.hardmax,
                "label": {
                    "key": self.label[0],
                    "value": self.label[1],
                },
            },
        )

    def to_dict(self) -> dict:
        return self.metric.to_dict()


@dataclass
class ConfusionMatrix:
    """
    Confusion matrix mapping predictions to ground truths.
    """

    counts: dict[str, dict[str, int]]
    label_key: str
    score_threshold: float
    hardmax: bool

    @property
    def metric(self) -> Metric:
        return Metric(
            type=type(self).__name__,
            value=self.counts,
            parameters={
                "score_threshold": self.score_threshold,
                "hardmax": self.hardmax,
                "label_key": self.label_key,
            },
        )

    def to_dict(self) -> dict:
        return self.metric.to_dict()


@dataclass
class _ThresholdValue:
    value: list[float]
    score_thresholds: list[float]
    hardmax: bool
    label: tuple[str, str]

    @property
    def metric(self) -> Metric:
        return Metric(
            type=type(self).__name__,
            value=self.value,
            parameters={
                "score_thresholds": self.score_thresholds,
                "hardmax": self.hardmax,
                "label": {
                    "key": self.label[0],
                    "value": self.label[1],
                },
            },
        )

    def to_dict(self) -> dict:
        return self.metric.to_dict()


class Precision(_ThresholdValue):
    pass


class Recall(_ThresholdValue):
    pass


class Accuracy(_ThresholdValue):
    pass


class F1(_ThresholdValue):
    pass


@dataclass
class ROCAUC:
    value: float
    label: tuple[str, str]

    @property
    def metric(self) -> Metric:
        return Metric(
            type=type(self).__name__,
            value=self.value,
            parameters={
                "label": {
                    "key": self.label[0],
                    "value": self.label[1],
                },
            },
        )

    def to_dict(self) -> dict:
        return self.metric.to_dict()


@dataclass
class mROCAUC:
    value: float
    label_key: str

    @property
    def metric(self) -> Metric:
        return Metric(
            type=type(self).__name__,
            value=self.value,
            parameters={
                "label_key": self.label_key,
            },
        )

    def to_dict(self) -> dict:
        return self.metric.to_dict()


@dataclass
class DetailedCounts:
    tp: list[int]
    fp_misclassification: list[int]
    fn_misclassification: list[int]
    fn_missing_prediction: list[int]
    tn: list[int]
    tp_examples: list[list[str]]
    fp_misclassification_examples: list[list[str]]
    fn_misclassification_examples: list[list[str]]
    fn_missing_prediction_examples: list[list[str]]
    tn_examples: list[list[str]]
    label: tuple[str, str]
    scores: list[float]

    def to_dict(self) -> dict:
        return {
            "value": {
                "tp": self.tp,
                "fp_misclassification": self.fp_misclassification,
                "fn_misclassification": self.fn_misclassification,
                "fn_missing_prediction": self.fn_missing_prediction,
                "tn": self.tn,
                "tp_examples": self.tp_examples,
                "fp_misclassification_examples": self.fp_misclassification_examples,
                "fn_misclassification_examples": self.fn_misclassification_examples,
                "fn_missing_prediction_examples": self.fn_missing_prediction_examples,
                "tn_examples": self.tn_examples,
            },
            "parameters": {
                "score_thresholds": self.scores,
                "label": {
                    "key": self.label[0],
                    "value": self.label[1],
                },
            },
            "type": type(self).__name__,
        }
