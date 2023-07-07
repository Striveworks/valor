from enum import Enum


# class DataType(Enum):
#     IMAGE = ("image",)
#     TABULAR = "tabular"


class DataType(Enum):
    IMAGE = "Image"
    TABULAR = "Tabular"

    @classmethod
    def invert(cls, value: str):
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"the value {value} is not in enum {cls.__name__}.")


class AnnotationType(Enum):
    NONE = "none"
    BOX = "box"
    POLYGON = "polygon"
    MULTIPOLYGON = "multipolygon"
    RASTER = "raster"


class TaskType(Enum):
    CLASSIFICATION = "classification"
    DETECTION = "detection"
    INSTANCE_SEGMENTATION = "instance_segmentation"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
