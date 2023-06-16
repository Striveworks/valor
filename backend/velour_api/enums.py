from enum import Enum, IntEnum


class Task(Enum):
    BBOX_OBJECT_DETECTION = "Bounding Box Object Detection"
    POLY_OBJECT_DETECTION = "Polygon Object Detection"
    INSTANCE_SEGMENTATION = "Instance Segmentation"
    CLASSIFICATION = "Classification"
    SEMANTIC_SEGMENTATION = "Semantic Segmentation"


class JobStatus(Enum):
    PENDING = "Pending"
    PROCESSING = "Processing"
    FAILED = "Failed"
    DONE = "Done"


class DatumTypes(Enum):
    IMAGE = "Image"
    TABULAR = "Tabular"


class AnnotationType(Enum):
    CLASSIFICATION = "classification"
    BBOX = "bbox"
    BOUNDARY = "boundary"
    RASTER = "raster"
    UNDEFINED = "undefined"


class TableStatus(IntEnum):
    CREATING = 0
    READY = 1
    EVALUATING = 2
    DELETING = 3
