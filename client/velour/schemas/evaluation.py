from dataclasses import dataclass, field
from typing import List, Union

from velour.schemas.filters import Filter


@dataclass
class DetectionParameters:
    """
    Defines important attributes to use when evaluating an object detection model.

    Attributes
    ----------
    iou_thresholds_to_compute : List[float]
        A list of floats describing which Intersection over Unions (IoUs) to use when calculating metrics (i.e., mAP).
    iou_thresholds_to_keep: List[float]
        A list of floats describing which Intersection over Union (IoUs) thresholds to calculate a metric for. Must be a subset of `iou_thresholds_to_compute`.
    """

    iou_thresholds_to_compute: List[float] = None
    iou_thresholds_to_keep: List[float] = None


@dataclass
class EvaluationSettings:
    """
    Defines important attributes for evaluating a model.

    Attributes
    ----------
    parameters : Union[DetectionParameters, None]
        The parameter object (e.g., `DetectionParameters) to use when creating an evaluation.
    filters: Union[Filter, None]
        The `Filter`object to use when creating an evaluation.
    """

    parameters: Union[DetectionParameters, None] = None
    filters: Union[Filter, None] = None


@dataclass
class EvaluationJob:
    """
    Defines important attributes of the API's `EvaluationJob`.

    Attributes
    ----------
    model : str
        The name of the `Model` invoked during the evaluation.
    dataset : str
        The name of the `Dataset` invoked during the evaluation.
    task_type : str
        The task type of the evaluation.
    settings : EvaluationSettings
        The `EvaluationSettings` object used to configurate the `EvaluationJob`.
    id : int
        The id of the job.
    """

    model: str
    dataset: str
    task_type: str
    settings: EvaluationSettings = field(default_factory=EvaluationSettings)
    id: int = None


@dataclass
class EvaluationResult:
    """
    Defines important attributes of the API's `EvaluationResult`.

    Attributes
    ----------
    dataset : str
        The name of the `Dataset` invoked during the evaluation.
    model : str
        The name of the `Model` invoked during the evaluation.
    settings : EvaluationSettings
        The `EvaluationSettings` object used to configurate the `EvaluationJob`.
    job_id : int
        The id of the job.
    status : str
        The status of the `EvaluationJob`.
    metrics : List[dict]
        A list of metric dictionaries returned by the job.
    confusion_matrices : List[dict]
        A list of confusion matrix dictionaries returned by the job.
    """

    dataset: str
    model: str
    settings: EvaluationSettings
    job_id: int
    status: str
    metrics: List[dict]
    confusion_matrices: List[dict]
