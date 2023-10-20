from dataclasses import dataclass
from typing import List, Union

from velour.enums import AnnotationType


@dataclass
class DetectionParameters:
    # thresholds to iterate over
    iou_thresholds_to_compute: List[float] = None
    iou_thresholds_to_keep: List[float] = None
    # constraints
    annotation_type: AnnotationType = None
    label_key: str = None
    min_area: float = None
    max_area: float = None


@dataclass
class EvaluationSettings:
    """General parameters defining any filters of the data such
    as model, dataset, groundtruth and prediction type, model, dataset,
    size constraints, coincidence/intersection constraints, etc.
    """

    model: str
    dataset: str
    parameters: Union[DetectionParameters, None] = None
    id: int = None
