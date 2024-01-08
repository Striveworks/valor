from enum import Enum


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
    DETECTION = "object-detection"
    SEGMENTATION = "semantic-segmentation"


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
