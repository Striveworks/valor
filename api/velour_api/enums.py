from enum import Enum


class AnnotationType(str, Enum):
    NONE = "none"
    BOX = "box"
    POLYGON = "polygon"
    MULTIPOLYGON = "multipolygon"
    RASTER = "raster"

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
