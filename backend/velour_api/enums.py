from enum import Enum


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


class TableStatus(Enum):
    CREATE = "creating"
    READY = "ready"
    EVALUATE = "evaluating"
    DELETE = "deleting"

    def next(self):
        if self == self.CREATE:
            return [self.CREATE, self.READY]
        elif self == self.READY:
            return [self.READY, self.EVALUATE, self.DELETE]
        elif self == self.EVALUATE:
            return [self.EVALUATE, self.READY]
        elif self == self.DELETE:
            return [self.DELETE]
        else:
            raise NotImplementedError
