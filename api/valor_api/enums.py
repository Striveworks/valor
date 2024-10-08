from enum import Enum


class AnnotationType(str, Enum):
    NONE = "none"
    BOX = "box"
    POLYGON = "polygon"
    MULTIPOLYGON = "multipolygon"
    RASTER = "raster"

    @property
    def numeric(self) -> int:
        mapping = {
            self.NONE: 0,
            self.BOX: 1,
            self.POLYGON: 2,
            self.MULTIPOLYGON: 3,
            self.RASTER: 4,
        }
        return mapping[self]

    def __gt__(self, other):
        if not isinstance(other, AnnotationType):
            raise TypeError(
                "operator can only be used with other `valor_api.enums.AnnotationType` objects"
            )
        return self.numeric > other.numeric

    def __lt__(self, other):
        if not isinstance(other, AnnotationType):
            raise TypeError(
                "operator can only be used with other `valor_api.enums.AnnotationType` objects"
            )
        return self.numeric < other.numeric

    def __ge__(self, other):
        if not isinstance(other, AnnotationType):
            raise TypeError(
                "operator can only be used with other `valor_api.enums.AnnotationType` objects"
            )
        return self.numeric >= other.numeric

    def __le__(self, other):
        if not isinstance(other, AnnotationType):
            raise TypeError(
                "operator can only be used with other `valor_api.enums.AnnotationType` objects"
            )
        return self.numeric <= other.numeric


class TaskType(str, Enum):
    SKIP = "skip"
    EMPTY = "empty"
    CLASSIFICATION = "classification"
    OBJECT_DETECTION = "object-detection"
    SEMANTIC_SEGMENTATION = "semantic-segmentation"
    EMBEDDING = "embedding"
    TEXT_GENERATION = "text-generation"


class TableStatus(str, Enum):
    CREATING = "creating"
    FINALIZED = "finalized"
    DELETING = "deleting"

    def next(self) -> set["TableStatus"]:
        """
        Returns the set of valid next states based on the current state.
        """
        match self:
            case self.CREATING:
                return {self.CREATING, self.FINALIZED, self.DELETING}
            case self.FINALIZED:
                return {self.FINALIZED, self.DELETING}
            case self.DELETING:
                return {self.DELETING}


class ModelStatus(str, Enum):
    READY = "ready"
    DELETING = "deleting"

    def next(self) -> set["ModelStatus"]:
        """
        Returns the set of valid next states based on the current state.
        """
        match self:
            case self.READY:
                return {self.READY, self.DELETING}
            case self.DELETING:
                return {self.DELETING}


class EvaluationStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    DELETING = "deleting"

    def next(self) -> set["EvaluationStatus"]:
        """
        Returns the set of valid next states based on the current state.
        """
        match (self):
            case EvaluationStatus.PENDING:
                return {self.PENDING, self.RUNNING, self.FAILED}
            case EvaluationStatus.RUNNING:
                return {self.RUNNING, self.DONE, self.FAILED}
            case EvaluationStatus.FAILED:
                return {self.FAILED, self.RUNNING, self.DELETING}
            case EvaluationStatus.DONE:
                return {self.DONE, self.DELETING}
            case EvaluationStatus.DELETING:
                return {self.DELETING}
            case _:
                raise NotImplementedError(
                    f"'{self}' is not a valid evaluation status."
                )


class MetricType(str, Enum):
    Accuracy = ("Accuracy",)
    Precision = ("Precision",)
    Recall = ("Recall",)
    F1 = ("F1",)
    ROCAUC = ("ROCAUC",)
    AP = "AP"
    AR = "AR"
    mAP = "mAP"
    mAR = "mAR"
    APAveragedOverIOUs = "APAveragedOverIOUs"
    mAPAveragedOverIOUs = "mAPAveragedOverIOUs"
    IOU = "IOU"
    mIOU = "mIOU"
    PrecisionRecallCurve = "PrecisionRecallCurve"
    DetailedPrecisionRecallCurve = "DetailedPrecisionRecallCurve"
    AnswerCorrectness = "AnswerCorrectness"
    AnswerRelevance = "AnswerRelevance"
    Bias = "Bias"
    BLEU = "BLEU"
    ContextPrecision = "ContextPrecision"
    ContextRecall = "ContextRecall"
    ContextRelevance = "ContextRelevance"
    Faithfulness = "Faithfulness"
    Hallucination = "Hallucination"
    ROUGE = "ROUGE"
    SummaryCoherence = "SummaryCoherence"
    Toxicity = "Toxicity"


class ROUGEType(str, Enum):
    ROUGE1 = "rouge1"
    ROUGE2 = "rouge2"
    ROUGEL = "rougeL"
    ROUGELSUM = "rougeLsum"
