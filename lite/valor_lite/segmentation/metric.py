from dataclasses import dataclass
from enum import Enum

from valor_lite.schemas import Metric


class MetricType(Enum):
    Precision = "Precision"
    Recall = "Recall"
    Accuracy = "Accuracy"
    F1 = "F1"
    IoU = "IoU"
    mIoU = "mIoU"
    ConfusionMatrix = "ConfusionMatrix"

    @classmethod
    def base(cls):
        return [
            cls.Precision,
            cls.Recall,
            cls.Accuracy,
            cls.F1,
            cls.IoU,
            cls.mIoU,
            cls.ConfusionMatrix,
        ]


@dataclass
class _ThresholdLabelValue:
    value: list[float]
    label: str

    @property
    def metric(self) -> Metric:
        return Metric(
            type=type(self).__name__,
            value=self.value,
            parameters={
                "label": self.label,
            },
        )

    def to_dict(self) -> dict:
        return self.metric.to_dict()


class Precision(_ThresholdLabelValue):
    pass


class Recall(_ThresholdLabelValue):
    pass


class F1(_ThresholdLabelValue):
    pass


class IoU(_ThresholdLabelValue):
    pass


@dataclass
class _ThresholdValue:
    value: list[float]

    @property
    def metric(self) -> Metric:
        return Metric(
            type=type(self).__name__,
            value=self.value,
            parameters={},
        )

    def to_dict(self) -> dict:
        return self.metric.to_dict()


class Accuracy(_ThresholdValue):
    pass


class mIoU(_ThresholdValue):
    pass


@dataclass
class ConfusionMatrix:
    confusion_matrix: dict[
        str,  # ground truth label value
        dict[
            str,  # prediction label value
            dict[str, float],  # iou
        ],
    ]
    missing_predictions: dict[
        str,  # ground truth label value
        dict[str, float],  # iou
    ]

    @property
    def metric(self) -> Metric:
        return Metric(
            type=type(self).__name__,
            value={
                "confusion_matrix": self.confusion_matrix,
                "missing_predictions": self.missing_predictions,
            },
            parameters={},
        )

    def to_dict(self) -> dict:
        return self.metric.to_dict()
