import datetime

from pydantic import BaseModel, ConfigDict

from valor_api.schemas.filters import Filter
from valor_api.schemas.metrics import Metric


class ClassificationParameters(BaseModel):
    score_thresholds: list[float]
    hardmax: bool
    label_map: dict[str, str]


class ObjectDetectionParameters(BaseModel):
    score_thresholds: list[float]
    iou_thresholds: list[float]
    label_map: dict[str, str]


class SemanticSegmentationParameters(BaseModel):
    label_map: dict[str, str]


class Summary(BaseModel):
    """
    Evaluation summary.

    Attributes
    ----------
    num_datums : int
        The number of datums involved in the evaluation.
    num_groundtruths : int
        The number of ground truths involved in the evaluation.
    num_predictions : int
        The number of predictions involved in the evaluation.
    num_labels : int
        The number of labels involved in the evaluation.
    missing_pred_labels: List[Label], optional
        A list of ground truth labels that aren't associated with any predictions.
    ignored_pred_labels: List[Label], optional
        A list of prediction labels that aren't associated with any ground truths.
    """

    num_datums: int
    num_groundtruths: int
    num_predictions: int
    num_labels: int
    ignored_prediction_labels: list[str]
    missing_prediction_labels: list[str]


class Evaluation(BaseModel):
    """
    An object for storing the returned results of a model evaluation (where groundtruths are compared with predictions to measure performance).

    Attributes
    ----------
    id : int
        The ID of the evaluation.
    dataset_names : list[str]
        The names of the evaluated datasets.
    model_name : str
        The name of the evaluated model.
    filters : schemas.Filter
        The evaluation filter used in the evaluation.
    parameters : ClassificationParameters | ObjectDetectionParameters | SemanticSegmentationParameters
        Any parameters used by the evaluation method.
    metrics : List[Metric]
        A list of metrics associated with the evaluation.
    summary : Summary
        A summary of the evaluation.
    created_at: datetime.datetime
        The time the evaluation was created.
    """

    dataset_names: list[str]
    model_name: str
    filters: Filter
    parameters: ClassificationParameters | ObjectDetectionParameters | SemanticSegmentationParameters
    metrics: list[Metric]
    summary: Summary
    created_at: datetime.datetime

    # pydantic setting
    model_config = ConfigDict(
        extra="allow", protected_namespaces=("protected_",)
    )
