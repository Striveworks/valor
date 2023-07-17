from enum import Enum


class DataType(Enum):
    IMAGE = "image"
    TABULAR = "tabular"


class AnnotationType(Enum):
    NONE = "none"
    BOX = "box"
    POLYGON = "polygon"
    MULTIPOLYGON = "multipolygon"
    RASTER = "raster"

    def __hash__(self):
        return hash(self.value)

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

    def __eq__(self, other):
        assert isinstance(other, type(self))
        return self.numeric == other.numeric

    def __gt__(self, other):
        assert isinstance(other, type(self))
        return self.numeric > other.numeric
    
    def __lt__(self, other):
        assert isinstance(other, type(self))
        return self.numeric < other.numeric
    
    def __ge__(self, other):
        assert isinstance(other, type(self))
        return self.numeric >= other.numeric
    
    def __le__(self, other):
        assert isinstance(other, type(self))
        return self.numeric <= other.numeric


class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    DETECTION = "detection"
    INSTANCE_SEGMENTATION = "instance_segmentation"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"


class JobStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    FAILED = "failed"
    DONE = "done"
