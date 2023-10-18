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
    DETECTION = "object-detection"
    SEGMENTATION = "semantic-segmentation"


class State(str, Enum):
    NONE = "none"
    CREATE = "create"
    READY = "ready"
    EVALUATE = "evaluate"
    DELETE = "delete"

    def next(self):
        if self == self.NONE:
            return {self.CREATE, self.DELETE}
        elif self == self.CREATE:
            return {self.CREATE, self.READY, self.DELETE}
        elif self == self.READY:
            return {self.READY, self.EVALUATE, self.DELETE}
        elif self == self.EVALUATE:
            return {self.EVALUATE, self.READY}
        elif self == self.DELETE:
            return {self.DELETE}
        else:
            raise ValueError
