from dataclasses import dataclass
from enum import Enum

from valor_lite.schemas import Metric


class MetricType(str, Enum):
    Counts = "Counts"
    Accuracy = "Accuracy"
    Precision = "Precision"
    Recall = "Recall"
    F1 = "F1"
    AP = "AP"
    AR = "AR"
    mAP = "mAP"
    mAR = "mAR"
    APAveragedOverIOUs = "APAveragedOverIOUs"
    mAPAveragedOverIOUs = "mAPAveragedOverIOUs"
    ARAveragedOverScores = "ARAveragedOverScores"
    mARAveragedOverScores = "mARAveragedOverScores"
    PrecisionRecallCurve = "PrecisionRecallCurve"
    DetailedPrecisionRecallCurve = "DetailedPrecisionRecallCurve"


@dataclass
class Counts:
    tp: int
    fp: int
    fn: int
    label: tuple[str, str]
    iou_threshold: float
    score_threshold: float

    @property
    def metric(self) -> Metric:
        return Metric(
            type=type(self).__name__,
            value={
                "tp": self.tp,
                "fp": self.fp,
                "fn": self.fn,
            },
            parameters={
                "iou_threshold": self.iou_threshold,
                "score_threshold": self.score_threshold,
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
    iou_threshold: float
    score_threshold: float

    @property
    def metric(self) -> Metric:
        return Metric(
            type=type(self).__name__,
            value=self.value,
            parameters={
                "iou_threshold": self.iou_threshold,
                "score_threshold": self.score_threshold,
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
class AP:
    value: float
    iou_threshold: float
    label: tuple[str, str]

    @property
    def metric(self) -> Metric:
        return Metric(
            type=type(self).__name__,
            value=self.value,
            parameters={
                "iou_threshold": self.iou_threshold,
                "label": {
                    "key": self.label[0],
                    "value": self.label[1],
                },
            },
        )

    def to_dict(self) -> dict:
        return self.metric.to_dict()


@dataclass
class mAP:
    value: float
    iou_threshold: float
    label_key: str

    @property
    def metric(self) -> Metric:
        return Metric(
            type=type(self).__name__,
            value=self.value,
            parameters={
                "iou_threshold": self.iou_threshold,
                "label_key": self.label_key,
            },
        )

    def to_dict(self) -> dict:
        return self.metric.to_dict()


@dataclass
class APAveragedOverIOUs:
    value: float
    iou_thresholds: list[float]
    label: tuple[str, str]

    @property
    def metric(self) -> Metric:
        return Metric(
            type=type(self).__name__,
            value=self.value,
            parameters={
                "iou_thresholds": self.iou_thresholds,
                "label": {
                    "key": self.label[0],
                    "value": self.label[1],
                },
            },
        )

    def to_dict(self) -> dict:
        return self.metric.to_dict()


@dataclass
class mAPAveragedOverIOUs:
    value: float
    iou_thresholds: list[float]
    label_key: str

    @property
    def metric(self) -> Metric:
        return Metric(
            type=type(self).__name__,
            value=self.value,
            parameters={
                "iou_thresholds": self.iou_thresholds,
                "label_key": self.label_key,
            },
        )

    def to_dict(self) -> dict:
        return self.metric.to_dict()


@dataclass
class AR:
    value: float
    score_threshold: float
    iou_thresholds: list[float]
    label: tuple[str, str]

    @property
    def metric(self) -> Metric:
        return Metric(
            type=type(self).__name__,
            value=self.value,
            parameters={
                "score_threshold": self.score_threshold,
                "iou_thresholds": self.iou_thresholds,
                "label": {
                    "key": self.label[0],
                    "value": self.label[1],
                },
            },
        )

    def to_dict(self) -> dict:
        return self.metric.to_dict()


@dataclass
class mAR:
    value: float
    score_threshold: float
    iou_thresholds: list[float]
    label_key: str

    @property
    def metric(self) -> Metric:
        return Metric(
            type=type(self).__name__,
            value=self.value,
            parameters={
                "score_threshold": self.score_threshold,
                "iou_thresholds": self.iou_thresholds,
                "label_key": self.label_key,
            },
        )

    def to_dict(self) -> dict:
        return self.metric.to_dict()


@dataclass
class ARAveragedOverScores:
    value: float
    score_thresholds: list[float]
    iou_thresholds: list[float]
    label: tuple[str, str]

    @property
    def metric(self) -> Metric:
        return Metric(
            type=type(self).__name__,
            value=self.value,
            parameters={
                "score_thresholds": self.score_thresholds,
                "iou_thresholds": self.iou_thresholds,
                "label": {
                    "key": self.label[0],
                    "value": self.label[1],
                },
            },
        )

    def to_dict(self) -> dict:
        return self.metric.to_dict()


@dataclass
class mARAveragedOverScores:
    value: float
    score_thresholds: list[float]
    iou_thresholds: list[float]
    label_key: str

    @property
    def metric(self) -> Metric:
        return Metric(
            type=type(self).__name__,
            value=self.value,
            parameters={
                "score_thresholds": self.score_thresholds,
                "iou_thresholds": self.iou_thresholds,
                "label_key": self.label_key,
            },
        )

    def to_dict(self) -> dict:
        return self.metric.to_dict()


@dataclass
class PrecisionRecallCurve:
    """
    Interpolated over recalls 0.0, 0.01, ..., 1.0.
    """

    precision: list[float]
    iou_threshold: float
    label: tuple[str, str]

    @property
    def metric(self) -> Metric:
        return Metric(
            type=type(self).__name__,
            value=self.precision,
            parameters={
                "iou_threshold": self.iou_threshold,
                "label": {"key": self.label[0], "value": self.label[1]},
            },
        )

    def to_dict(self) -> dict:
        return self.metric.to_dict()


@dataclass
class DetailedPrecisionRecallPoint:
    score_threshold: float
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
            "score_threshold": self.score_threshold,
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
    iou_threshold: float
    value: list[DetailedPrecisionRecallPoint]
    label: tuple[str, str]

    def to_dict(self) -> dict:
        return {
            "value": [pt.to_dict() for pt in self.value],
            "iou_threshold": self.iou_threshold,
            "label": {
                "key": self.label[0],
                "value": self.label[1],
            },
            "type": "DetailedPrecisionRecallCurve",
        }
