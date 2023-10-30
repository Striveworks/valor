from dataclasses import dataclass, field
from typing import List, Union

from velour.enums import AnnotationType
from velour.schemas.filters import Filter


@dataclass
class DetectionParameters:
    # thresholds to iterate over
    iou_thresholds_to_compute: List[float] = None
    iou_thresholds_to_keep: List[float] = None


@dataclass
class EvaluationSettings:
    parameters: Union[DetectionParameters, None] = None
    filters: Filter = field(default=Filter())


@dataclass
class EvaluationJob:
    """General parameters defining any filters of the data such
    as model, dataset, groundtruth and prediction type, model, dataset,
    size constraints, coincidence/intersection constraints, etc.
    """

    model: str
    dataset: str
    settings: EvaluationSettings = field(default=EvaluationSettings())
    id: int = None
