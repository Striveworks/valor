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
    ConfusionMatrix = "ConfusionMatrix"

    @classmethod
    def base_metrics(cls):
        return [
            cls.Counts,
            cls.Accuracy,
            cls.Precision,
            cls.Recall,
            cls.F1,
            cls.AP,
            cls.AR,
            cls.mAP,
            cls.mAR,
            cls.APAveragedOverIOUs,
            cls.mAPAveragedOverIOUs,
            cls.ARAveragedOverScores,
            cls.mARAveragedOverScores,
            cls.PrecisionRecallCurve,
        ]


@dataclass
class Counts:
    tp: int
    fp: int
    fn: int
    label: str
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
                "label": self.label,
            },
        )

    def to_dict(self) -> dict:
        return self.metric.to_dict()


@dataclass
class ClassMetric:
    value: float
    label: str
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
                "label": self.label,
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
    label: str

    @property
    def metric(self) -> Metric:
        return Metric(
            type=type(self).__name__,
            value=self.value,
            parameters={
                "iou_threshold": self.iou_threshold,
                "label": self.label,
            },
        )

    def to_dict(self) -> dict:
        return self.metric.to_dict()


@dataclass
class mAP:
    value: float
    iou_threshold: float

    @property
    def metric(self) -> Metric:
        return Metric(
            type=type(self).__name__,
            value=self.value,
            parameters={
                "iou_threshold": self.iou_threshold,
            },
        )

    def to_dict(self) -> dict:
        return self.metric.to_dict()


@dataclass
class APAveragedOverIOUs:
    value: float
    iou_thresholds: list[float]
    label: str

    @property
    def metric(self) -> Metric:
        return Metric(
            type=type(self).__name__,
            value=self.value,
            parameters={
                "iou_thresholds": self.iou_thresholds,
                "label": self.label,
            },
        )

    def to_dict(self) -> dict:
        return self.metric.to_dict()


@dataclass
class mAPAveragedOverIOUs:
    value: float
    iou_thresholds: list[float]

    @property
    def metric(self) -> Metric:
        return Metric(
            type=type(self).__name__,
            value=self.value,
            parameters={
                "iou_thresholds": self.iou_thresholds,
            },
        )

    def to_dict(self) -> dict:
        return self.metric.to_dict()


@dataclass
class AR:
    value: float
    score_threshold: float
    iou_thresholds: list[float]
    label: str

    @property
    def metric(self) -> Metric:
        return Metric(
            type=type(self).__name__,
            value=self.value,
            parameters={
                "score_threshold": self.score_threshold,
                "iou_thresholds": self.iou_thresholds,
                "label": self.label,
            },
        )

    def to_dict(self) -> dict:
        return self.metric.to_dict()


@dataclass
class mAR:
    value: float
    score_threshold: float
    iou_thresholds: list[float]

    @property
    def metric(self) -> Metric:
        return Metric(
            type=type(self).__name__,
            value=self.value,
            parameters={
                "score_threshold": self.score_threshold,
                "iou_thresholds": self.iou_thresholds,
            },
        )

    def to_dict(self) -> dict:
        return self.metric.to_dict()


@dataclass
class ARAveragedOverScores:
    value: float
    score_thresholds: list[float]
    iou_thresholds: list[float]
    label: str

    @property
    def metric(self) -> Metric:
        return Metric(
            type=type(self).__name__,
            value=self.value,
            parameters={
                "score_thresholds": self.score_thresholds,
                "iou_thresholds": self.iou_thresholds,
                "label": self.label,
            },
        )

    def to_dict(self) -> dict:
        return self.metric.to_dict()


@dataclass
class mARAveragedOverScores:
    value: float
    score_thresholds: list[float]
    iou_thresholds: list[float]

    @property
    def metric(self) -> Metric:
        return Metric(
            type=type(self).__name__,
            value=self.value,
            parameters={
                "score_thresholds": self.score_thresholds,
                "iou_thresholds": self.iou_thresholds,
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
    label: str

    @property
    def metric(self) -> Metric:
        return Metric(
            type=type(self).__name__,
            value=self.precision,
            parameters={
                "iou_threshold": self.iou_threshold,
                "label": self.label,
            },
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
                        str,  # either `datum`, `groundtruth`, `prediction` or score
                        str  # datum uid
                        | dict[
                            str, float
                        ]  # bounding box (xmin, xmax, ymin, ymax)
                        | float,  # prediction score
                    ]
                ],
            ],
        ],
    ]
    hallucinations: dict[
        str,  # prediction label value
        dict[
            str,  # either `count` or `examples`
            int
            | list[
                dict[
                    str,  # either `datum`, `prediction` or score
                    str  # datum uid
                    | float  # prediction score
                    | dict[
                        str, float
                    ],  # bounding box (xmin, xmax, ymin, ymax)
                ]
            ],
        ],
    ]
    missing_predictions: dict[
        str,  # ground truth label value
        dict[
            str,  # either `count` or `examples`
            int
            | list[
                dict[
                    str,  # either `datum` or `groundtruth`
                    str  # datum uid
                    | dict[
                        str, float
                    ],  # bounding box (xmin, xmax, ymin, ymax)
                ]
            ],
        ],
    ]
    score_threshold: float
    iou_threshold: float
    number_of_examples: int

    @property
    def metric(self) -> Metric:
        return Metric(
            type=type(self).__name__,
            value={
                "confusion_matrix": self.confusion_matrix,
                "hallucinations": self.hallucinations,
                "missing_predictions": self.missing_predictions,
            },
            parameters={
                "score_threshold": self.score_threshold,
                "iou_threshold": self.iou_threshold,
            },
        )

    def to_dict(self) -> dict:
        return self.metric.to_dict()
