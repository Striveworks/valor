from dataclasses import dataclass, field
from typing import List

from velour.schemas.filters import Filter


@dataclass
class EvaluationParameters:
    """
    Defines important attributes to use when evaluating an object detection model.

    Attributes
    ----------
    iou_thresholds_to_compute : List[float]
        A list of floats describing which Intersection over Unions (IoUs) to use when calculating metrics (i.e., mAP).
    iou_thresholds_to_return: List[float]
        A list of floats describing which Intersection over Union (IoUs) thresholds to calculate a metric for. Must be a subset of `iou_thresholds_to_compute`.
    """

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
    evaluation_filter : schemas.Filter
        The filter object used to define what the model(s) is evaluating against.
    parameters : EvaluationParameters
        Any parameters that are used to modify an evaluation method.
    """

    model_filter: Filter
    evaluation_filter: Filter
    parameters: EvaluationParameters = field(
        default_factory=EvaluationParameters
    )

    def __post_init__(self):
        if isinstance(self.model_filter, dict):
            self.model_filter = Filter(**self.model_filter)
        if isinstance(self.evaluation_filter, dict):
            self.evaluation_filter = Filter(**self.evaluation_filter)
        if isinstance(self.parameters, dict):
            self.parameters = EvaluationParameters(**self.parameters)
