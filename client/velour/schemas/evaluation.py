from dataclasses import dataclass, field
from typing import List

from velour.enums import AnnotationType
from velour.schemas.filters import Filter


@dataclass
class EvaluationParameters:
    """
    Defines paramaters for evaluation methods.

    Attributes
    ----------
    iou_thresholds_to_compute : List[float]
        A list of floats describing which Intersection over Unions (IoUs) to use when calculating metrics (i.e., mAP).
    iou_thresholds_to_return: List[float]
        A list of floats describing which Intersection over Union (IoUs) thresholds to calculate a metric for. Must be a subset of `iou_thresholds_to_compute`.
    """

    # object detection
    force_annotation_type: AnnotationType = None
    iou_thresholds_to_compute: List[float] = None
    iou_thresholds_to_return: List[float] = None


@dataclass
class EvaluationRequest:
    """
    An evaluation request.

    Defines important attributes of the API's `EvaluationRequest`.

    Attributes
    ----------
    model_filter : schemas.Filter
        The filter used to enumerate all the models we want to evaluate.
    dataset_filter : schemas.Filter
        The filter object used to define what the model(s) is evaluating against.
    parameters : EvaluationParameters
        Any parameters that are used to modify an evaluation method.
    """

    model_filter: Filter
    dataset_filter: Filter
    parameters: EvaluationParameters = field(
        default_factory=EvaluationParameters
    )

    def __post_init__(self):
        if isinstance(self.model_filter, dict):
            self.model_filter = Filter(**self.model_filter)
        if isinstance(self.dataset_filter, dict):
            self.dataset_filter = Filter(**self.dataset_filter)
        if isinstance(self.parameters, dict):
            self.parameters = EvaluationParameters(**self.parameters)
