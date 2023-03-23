from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Union

from velour.data_types import Label


class Task(Enum):
    OBJECT_DETECTION = "Object Detection"
    INSTANCE_SEGMENTATION = "Instance Segmentation"
    IMAGE_CLASSIFICATION = "Image Classification"
    SEMANTIC_SEGMENTATION = "Semantic Segmentation"


@dataclass
class APMetric:
    label: Label
    iou: float
    value: float


@dataclass
class mAPMetric:
    iou: Union[float, List[float]]
    value: float


@dataclass
class EvaluationResut:
    metrics: List[Any]
    info: dict
