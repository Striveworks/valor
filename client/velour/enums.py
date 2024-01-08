from enum import Enum


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
    JSON = "json"
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
