from dataclasses import dataclass
from enum import Enum

from valor_lite.schemas import Metric


class MetricType(Enum):
    Counts = "Counts"
    ROCAUC = "ROCAUC"
    mROCAUC = "mROCAUC"
    Precision = "Precision"
    Recall = "Recall"
    Accuracy = "Accuracy"
    F1 = "F1"
    ConfusionMatrix = "ConfusionMatrix"

    @classmethod
    def base(cls):
        return [
            cls.Counts,
            cls.ROCAUC,
            cls.mROCAUC,
            cls.Precision,
            cls.Recall,
            cls.Accuracy,
            cls.F1,
        ]


@dataclass
class Counts:
    tp: list[int]
    fp: list[int]
    fn: list[int]
    tn: list[int]
    score_thresholds: list[float]
    hardmax: bool
    label: str

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
                "label": self.label,
            },
        )

    def to_dict(self) -> dict:
        return self.metric.to_dict()


@dataclass
class _ThresholdValue:
    value: list[float]
    score_thresholds: list[float]
    hardmax: bool
    label: str

    @property
    def metric(self) -> Metric:
        return Metric(
            type=type(self).__name__,
            value=self.value,
            parameters={
                "score_thresholds": self.score_thresholds,
                "hardmax": self.hardmax,
                "label": self.label,
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
    label: str

    @property
    def metric(self) -> Metric:
        return Metric(
            type=type(self).__name__,
            value=self.value,
            parameters={"label": self.label},
        )

    def to_dict(self) -> dict:
        return self.metric.to_dict()


@dataclass
class mROCAUC:
    value: float

    @property
    def metric(self) -> Metric:
        return Metric(
            type=type(self).__name__,
            value=self.value,
            parameters={},
        )

    def to_dict(self) -> dict:
        return self.metric.to_dict()


@dataclass
class ConfusionMatrix:
    confusion_matrix: dict[
        str,  # ground truth label value
        dict[
            str,  # prediction label value
            dict[
                str,  # either `count` or `examples`
                int
                | list[
                    dict[
                        str,  # either `datum` or `score`
                        str | float,  # datum uid  # prediction score
                    ]
                ],
            ],
        ],
    ]
    missing_predictions: dict[
        str,  # ground truth label value
        dict[
            str,  # either `count` or `examples`
            int | list[dict[str, str]],  # count or datum examples
        ],
    ]
    score_threshold: float
    number_of_examples: int

    @property
    def metric(self) -> Metric:
        return Metric(
            type=type(self).__name__,
            value={
                "confusion_matrix": self.confusion_matrix,
                "missing_predictions": self.missing_predictions,
            },
            parameters={
                "score_threshold": self.score_threshold,
            },
        )

    def to_dict(self) -> dict:
        return self.metric.to_dict()
