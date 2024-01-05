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
    SKIP = "skip"
    EMPTY = "empty"
    CLASSIFICATION = "classification"
    DETECTION = "object-detection"
    SEGMENTATION = "semantic-segmentation"


class JobStatus(str, Enum):
    NONE = "none"
    PENDING = "pending"
    CREATING = "creating"
    PROCESSING = "processing"
    DELETING = "deleting"
    FAILED = "failed"
    DONE = "done"

    def next(self):
        """
        Returns the set of valid next states based on the current state.
        """
        if self == self.NONE:
            return {self.NONE}
        elif self == self.PENDING:
            return {self.PENDING, self.CREATING, self.PROCESSING}
        elif self == self.CREATING:
            return {self.CREATING, self.DONE, self.FAILED, self.DELETING}
        elif self == self.PROCESSING:
            return {self.PROCESSING, self.DONE, self.FAILED, self.DELETING}
        elif self == self.FAILED:
            return {self.FAILED, self.CREATING, self.PROCESSING, self.DELETING}
        elif self == self.DONE:
            return {self.DONE, self.PROCESSING, self.DELETING}
        elif self == self.DELETING:
            return {self.DELETING, self.NONE}
        else:
            raise ValueError


class TableStatus(str, Enum):
    CREATING = "creating"
    FINALIZED = "finalized"
    DELETING = "deleting"

    def next(self) -> set["TableStatus"]:
        """
        Returns the set of valid next states based on the current state.
        """
        if self == self.CREATING:
            return {self.CREATING, self.FINALIZED, self.DELETING}
        elif self == self.FINALIZED:
            return {self.FINALIZED, self.DELETING}
        elif self == self.DELETING:
            return {self.DELETING}
        else:
            raise ValueError
        

class ModelStatus(str, Enum):
    READY = "ready"
    DELETING = "deleting"

    def next(self) -> set["ModelStatus"]:
        """
        Returns the set of valid next states based on the current state.
        """
        if self == self.READY:
            return {self.READY, self.DELETING}
        elif self == self.DELETING:
            return {self.DELETING}
        else:
            raise ValueError
        

class EvaluationStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    DELETING = "deleting"

    def next(self):
        """
        Returns the set of valid next states based on the current state.
        """
        if self == self.PENDING:
            return {self.PENDING, self.RUNNING}
        elif self == self.RUNNING:
            return {self.RUNNING, self.DONE, self.FAILED}
        elif self == self.FAILED:
            return {self.FAILED, self.PENDING, self.DELETING}
        elif self == self.DONE:
            return {self.DONE, self.DELETING}
        elif self == self.DELETING:
            return {self.DELETING}
        else:
            raise ValueError