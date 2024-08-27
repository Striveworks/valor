from enum import Enum
from typing import Set


class AnnotationType(str, Enum):
    NONE = "none"
    BOX = "box"
    POLYGON = "polygon"
    MULTIPOLYGON = "multipolygon"
    RASTER = "raster"


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


class EvaluationStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    DELETING = "deleting"


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

    @classmethod
    def classification(cls) -> Set["MetricType"]:
        """
        MetricTypes for classification tasks.
        """
        return {
            cls.Accuracy,
            cls.Precision,
            cls.Recall,
            cls.F1,
            cls.ROCAUC,
            cls.PrecisionRecallCurve,
            cls.DetailedPrecisionRecallCurve,
        }

    @classmethod
    def object_detection(cls) -> Set["MetricType"]:
        """
        MetricTypes for object-detection tasks.
        """
        return {
            cls.AP,
            cls.AR,
            cls.mAP,
            cls.mAR,
            cls.APAveragedOverIOUs,
            cls.mAPAveragedOverIOUs,
            cls.PrecisionRecallCurve,
            cls.DetailedPrecisionRecallCurve,
        }

    @classmethod
    def semantic_segmentation(cls) -> Set["MetricType"]:
        """
        MetricTypes for semantic-segmentation tasks.
        """
        return {
            cls.IOU,
            cls.mIOU,
            cls.Precision,
            cls.Recall,
            cls.F1,
        }

    @classmethod
    def text_generation(cls) -> Set["MetricType"]:
        """
        MetricTypes for text-generation tasks.
        """
        return {
            cls.AnswerCorrectness,
            cls.AnswerRelevance,
            cls.Bias,
            cls.BLEU,
            cls.ContextPrecision,
            cls.ContextRecall,
            cls.ContextRelevance,
            cls.Faithfulness,
            cls.Hallucination,
            cls.ROUGE,
            cls.SummaryCoherence,
            cls.Toxicity,
        }


class ROUGEType(str, Enum):
    ROUGE1 = "rouge1"
    ROUGE2 = "rouge2"
    ROUGEL = "rougeL"
    ROUGELSUM = "rougeLsum"
