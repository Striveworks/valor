from enum import Enum
from typing import Set


class AnnotationType(str, Enum):
    NONE = "none"
    BOX = "box"
    POLYGON = "polygon"
    RASTER = "raster"


class TaskType(str, Enum):
    SKIP = "skip"
    EMPTY = "empty"
    CLASSIFICATION = "classification"
    OBJECT_DETECTION = "object-detection"
    SEMANTIC_SEGMENTATION = "semantic-segmentation"
    EMBEDDING = "embedding"


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
        }
