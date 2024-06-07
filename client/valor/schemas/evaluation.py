from dataclasses import dataclass
from typing import List, Optional, Union

from valor.enums import AnnotationType, TaskType
from valor.schemas.filters import Filter


@dataclass
class EvaluationParameters:
    """
    Defines parameters for evaluation methods.

    Attributes
    ----------
    iou_thresholds_to_compute : Optional[List[float]]
        A list of floats describing which Intersection over Unions (IoUs) to use when calculating metrics (i.e., mAP).
    iou_thresholds_to_return: Optional[List[float]]
        A list of floats describing which Intersection over Union (IoUs) thresholds to calculate a metric for. Must be a subset of `iou_thresholds_to_compute`.
    label_map: Optional[List[List[List[str]]]]
        Optional mapping of individual labels to a grouper label. Useful when you need to evaluate performance using labels that differ across datasets and models.
    recall_score_threshold: float, default=0
        The confidence score threshold for use when determining whether to count a prediction as a true positive or not while calculating Average Recall.
    metrics: List[str], optional
        The list of metrics to compute, store, and return to the user.
    pr_curve_iou_threshold: float, optional
            The IOU threshold to use when calculating precision-recall curves for object detection tasks. Defaults to 0.5.

    """

    task_type: TaskType

    # object detection
    convert_annotations_to_type: Optional[AnnotationType] = None
    iou_thresholds_to_compute: Optional[List[float]] = None
    iou_thresholds_to_return: Optional[List[float]] = None
    label_map: Optional[List[List[List[str]]]] = None
    recall_score_threshold: float = 0
    metrics_to_return: Optional[List[str]] = None
    pr_curve_iou_threshold: float = 0.5


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
    parameters: EvaluationParameters

    def __post_init__(self):
        if isinstance(self.datum_filter, dict):
            self.datum_filter = Filter(**self.datum_filter)
        if isinstance(self.parameters, dict):
            self.parameters = EvaluationParameters(**self.parameters)
