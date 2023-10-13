from enum import Enum


class DataType(str, Enum):
    IMAGE = "image"
    TABULAR = "tabular"


class AnnotationType(str, Enum):
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
        if not isinstance(other, type(self)):
            raise TypeError(
                "operator can only be used with other `velour_api.enums.AnnotationType` objects"
            )
        return self.numeric == other.numeric

    def __gt__(self, other):
        if not isinstance(other, type(self)):
            raise TypeError(
                "operator can only be used with other `velour_api.enums.AnnotationType` objects"
            )
        return self.numeric > other.numeric

    def __lt__(self, other):
        if not isinstance(other, type(self)):
            raise TypeError(
                "operator can only be used with other `velour_api.enums.AnnotationType` objects"
            )
        return self.numeric < other.numeric

    def __ge__(self, other):
        if not isinstance(other, type(self)):
            raise TypeError(
                "operator can only be used with other `velour_api.enums.AnnotationType` objects"
            )
        return self.numeric >= other.numeric

    def __le__(self, other):
        if not isinstance(other, type(self)):
            raise TypeError(
                "operator can only be used with other `velour_api.enums.AnnotationType` objects"
            )
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

    def next(self):
        if self == self.PENDING:
            return {self.PENDING, self.PROCESSING}
        elif self == self.PROCESSING:
            return {self.PROCESSING, self.DONE, self.FAILED}
        elif self == self.FAILED:
            return {self.FAILED, self.PENDING}
        elif self == self.DONE:
            return {self.DONE}
        else:
            raise ValueError


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


# @TODO: Fill in methods, will we have multiple per task_type?
class EvaluationType(str, Enum):
    CLF = "classification"
    AP = "average-precision"
    DICE = "semantic-ap"
