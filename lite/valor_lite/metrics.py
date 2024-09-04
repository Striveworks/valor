from dataclasses import dataclass


@dataclass
class ValueAtIoU:
    value: float
    iou: float


@dataclass
class AP:
    values: list[ValueAtIoU]
    label: tuple[str, str]

    def to_dict(self) -> dict:
        return {
            "type": "AP",
            "values": {str(value.iou): value.value for value in self.values},
            "label": {
                "key": self.label[0],
                "value": self.label[1],
            },
        }


@dataclass
class CountWithExamples:
    value: int
    examples: list[str]

    def __post_init__(self):
        self.value = int(self.value)

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "examples": self.examples,
        }


@dataclass
class PrecisionRecallCurvePoint:
    score: float
    tp: CountWithExamples
    fp_misclassification: CountWithExamples
    fp_hallucination: CountWithExamples
    fn_misclassification: CountWithExamples
    fn_missing_prediction: CountWithExamples

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "tp": self.tp.to_dict(),
            "fp_misclassification": self.fp_misclassification.to_dict(),
            "fp_hallucination": self.fp_hallucination.to_dict(),
            "fn_misclassification": self.fn_misclassification.to_dict(),
            "fn_missing_prediction": self.fn_missing_prediction.to_dict(),
        }


@dataclass
class PrecisionRecallCurve:
    iou: float
    value: list[PrecisionRecallCurvePoint]
    label: tuple[str, str]

    def to_dict(self) -> dict:
        return {
            "value": [pt.to_dict() for pt in self.value],
            "iou": self.iou,
            "label": {
                "key": self.label[0],
                "value": self.label[1],
            },
            "type": "PrecisionRecallCurve",
        }
