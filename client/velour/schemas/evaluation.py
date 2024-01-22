from dataclasses import dataclass, field
from typing import List, Union

from velour.enums import AnnotationType, TaskType
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

    task_type: TaskType

    # object detection
    convert_annotations_to_type: AnnotationType = None
    iou_thresholds_to_compute: List[float] = None
    iou_thresholds_to_return: List[float] = None


@dataclass
class EvaluationRequest:
    """
    An evaluation request.

    Defines important attributes of the API's `EvaluationRequest`.

    Attributes
    ----------
    model_names : List[str]
        The list of models we want to evaluate by name.
    datum_filter : schemas.Filter
        The filter object used to define what the model(s) is evaluating against.
    parameters : EvaluationParameters
        Any parameters that are used to modify an evaluation method.
    """

    model_names: Union[str, List[str]]
    datum_filter: Filter
    parameters: EvaluationParameters = field(
        default_factory=EvaluationParameters
    )

    def __post_init__(self):
        if isinstance(self.datum_filter, dict):
            self.datum_filter = Filter(**self.datum_filter)
        if isinstance(self.parameters, dict):
            self.parameters = EvaluationParameters(**self.parameters)
