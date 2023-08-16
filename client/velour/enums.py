from enum import Enum


class JobStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    FAILED = "failed"
    DONE = "done"


class DataType(Enum):
    IMAGE = "image"
    TABULAR = "tabular"

    @classmethod
    def invert(cls, value: str):
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"the value {value} is not in enum {cls.__name__}.")


class AnnotationType(str, Enum):
    NONE = "none"
    BOX = "box"
    POLYGON = "polygon"
    MULTIPOLYGON = "multipolygon"
    RASTER = "raster"


class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    DETECTION = "detection"
    INSTANCE_SEGMENTATION = "instance_segmentation"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"


class Table(str, Enum):
    DATASET = "dataset"
    MODEL = "model"
    DATUM = "datum"
    ANNOTATION = "annotation"
    GROUND_TRUTH = "groundtruth"
    PREDICTION = "prediction"
    LABEL = "label"
    METADATA = "metadatum"
