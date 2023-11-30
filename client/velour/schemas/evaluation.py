from dataclasses import dataclass, field
from typing import List, Union

from velour.schemas.filters import Filter


@dataclass
class DetectionParameters:
    # thresholds to iterate over
    iou_thresholds_to_compute: List[float] = None
    iou_thresholds_to_keep: List[float] = None


@dataclass
class EvaluationSettings:
    parameters: Union[DetectionParameters, None] = None
    filters: Union[Filter, None] = None


@dataclass
class EvaluationJob:
    """General parameters defining any filters of the data such
    as model, dataset, groundtruth and prediction type, model, dataset,
    size constraints, coincidence/intersection constraints, etc.
    """

    model: str
    dataset: str
    task_type: str
    settings: EvaluationSettings = field(default_factory=EvaluationSettings)
    id: int = None


@dataclass
class EvaluationResult:
    dataset: str
    model: str
    settings: EvaluationSettings
    job_id: int
    status: str
    metrics: List[dict]
    confusion_matrices: List[dict]

