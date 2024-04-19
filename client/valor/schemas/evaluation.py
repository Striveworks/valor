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
    compute_pr_curves: bool
        A boolean which determines whether we calculate precision-recall curves or not.
    pr_curve_iou_threshold: float, optional
            The IOU threshold to use when calculating precision-recall curves for object detection tasks. Defaults to 0.5. Does nothing when compute_pr_curves is set to False or None.

    """

    task_type: TaskType

    # object detection
    convert_annotations_to_type: Optional[AnnotationType] = None
    iou_thresholds_to_compute: Optional[List[float]] = None
    iou_thresholds_to_return: Optional[List[float]] = None
    label_map: Optional[List[List[List[str]]]] = None
    recall_score_threshold: float = 0
    compute_pr_curves: bool = False
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
    meta: dict[str, str | float | dict], optional
        Metadata about the evaluation run.
    """

    model_names: Union[str, List[str]]
    datum_filter: Filter
    parameters: EvaluationParameters
    meta: Optional[dict]

    def __post_init__(self):
        if isinstance(self.datum_filter, dict):
            self.datum_filter = Filter(**self.datum_filter)
        if isinstance(self.parameters, dict):
            self.parameters = EvaluationParameters(**self.parameters)
