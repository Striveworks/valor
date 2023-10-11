from dataclasses import dataclass
from typing import List

from velour.enums import AnnotationType, EvaluationType


@dataclass
class EvaluationConstraints:
    # type
    target_type: AnnotationType = None
    label_key: str = None
    # geometric
    min_area: float = None
    max_area: float = None


@dataclass
class EvaluationThresholds:
    iou_thresholds_to_compute: List[float] = None
    iou_thresholds_to_keep: List[float] = None


@dataclass
class EvaluationSettings:
    """General parameters defining any filters of the data such
    as model, dataset, groundtruth and prediction type, model, dataset,
    size constraints, coincidence/intersection constraints, etc.
    """

    model: str
    dataset: str
    evaluation_type: EvaluationType
    constraints: EvaluationConstraints = None
    thresholds: EvaluationThresholds = None
    id: int = None
