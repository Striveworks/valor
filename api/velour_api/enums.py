from enum import Enum


class DataType(str, Enum):
    IMAGE = "image"
    TABULAR = "tabular"


class AnnotationType(str, Enum):
    NONE = "none"
    JSON = "json"
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
            self.JSON: 1,
            self.BOX: 2,
            self.POLYGON: 3,
            self.MULTIPOLYGON: 4,
            self.RASTER: 5,
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
    DETECTION = "object-detection"
    SEGMENTATION = "semantic-segmentation"


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
