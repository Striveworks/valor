from pydantic import BaseModel, ConfigDict, Field, model_validator

from velour_api.enums import EvaluationStatus, TaskType
from velour_api.schemas.core import Label
from velour_api.schemas.filters import Filter
from velour_api.schemas.metrics import ConfusionMatrixResponse, Metric


class DetectionParameters(BaseModel):
    """
    Defines important attributes to use when evaluating an object detection model.

    Attributes
    ----------
    iou_thresholds_to_compute : List[float]
        A list of floats describing which Intersection over Unions (IoUs) to use when calculating metrics (i.e., mAP).
    iou_thresholds_to_return: List[float]
        A list of floats describing which Intersection over Union (IoUs) thresholds to calculate a metric for. Must be a subset of `iou_thresholds_to_compute`.
    """

    # thresholds to iterate over (mutable defaults are ok for pydantic models)
    iou_thresholds_to_compute: list[float] = [
        round(0.5 + 0.05 * i, 2) for i in range(10)
    ]
    iou_thresholds_to_return: list[float] | None = [0.5, 0.75]

    # pydantic setting
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    @classmethod
    def _check_ious(cls, values):
        """Validate the IOU thresholds."""
        if values.iou_thresholds_to_return:
            for iou in values.iou_thresholds_to_return:
                if iou not in values.iou_thresholds_to_compute:
                    raise ValueError(
                        "`iou_thresholds_to_return` must be contained in `iou_thresholds_to_compute`"
                    )
        return values


class EvaluationParameters(BaseModel):
    """
    Parameters for evaluation methods.

    Attributes
    ----------
    classification : undefined, optional
        Placeholder for classification parameters.
    detection : DetectionParameters, optional
        Parameters for the detection evaluation method.
    segmentation : undefined, optional
        Placeholder for segmentation parameters.
    """

    # classification = None
    detection: DetectionParameters | None = None
    # segmentation = None


class EvaluationRequest(BaseModel):
    """
    Request for evaluation.

    A `model_filter` that returns multiple models will lead to the creation of multiple evaluations.

    Attributes
    ----------
    model_filter : schemas.Filter
        The filter used to enumerate all the models we want to evaluate.
    evaluation_filter : schemas.Filter
        The filter object used to define what the model is evaluating against.
    parameters : DetectionParameters, optional
        Any parameters that are used to modify an evaluation method.
    """

    model_filter: Filter
    evaluation_filter: Filter
    parameters: EvaluationParameters = Field(default=EvaluationParameters())

    # pydantic setting
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    @classmethod
    def _validate_filter(cls, values):
        """Validates filters for evaluation."""

        # validate model_filter
        if values.model_filter.task_types is not None:
            raise ValueError("`model_filter` should not define task types.")
        elif values.model_filter.annotation_types is not None:
            raise ValueError(
                "`model_filter` should not define annotation types."
            )

        # validate evaluation_filter
        if values.evaluation_filter.task_types is None:
            raise ValueError("`evaluation_filter.task_types` is empty.")
        for task_type in values.evaluation_filter.task_types:
            match task_type:
                case TaskType.CLASSIFICATION:
                    pass
                case TaskType.DETECTION:
                    if values.parameters.detection is None:
                        values.parameters.detection = DetectionParameters()
                case TaskType.SEGMENTATION:
                    pass
                case _:
                    raise NotImplementedError

        return values


class CreateDetectionEvaluationResponse(BaseModel):
    """
    The response from a job that creates AP metrics.

    Attributes
    ----------
    missing_pred_labels: list[Label]
        A list of missing prediction labels.
    ignored_pred_labels: list[Label]
        A list of ignored preiction labels.
    evaluation_id: int
        The job ID.
    """

    missing_pred_labels: list[Label]
    ignored_pred_labels: list[Label]
    evaluation_id: int


class CreateSemanticSegmentationEvaluationResponse(BaseModel):
    """
    The response from a job that creates segmentation metrics.

    Attributes
    ----------
    missing_pred_labels: list[Label]
        A list of missing prediction labels.
    ignored_pred_labels: list[Label]
        A list of ignored preiction labels.
    evaluation_id: int
        The job ID.
    """

    missing_pred_labels: list[Label]
    ignored_pred_labels: list[Label]
    evaluation_id: int


class CreateClfEvaluationResponse(BaseModel):
    """
    The response from a job that creates classification metrics.

    Attributes
    ----------
    missing_pred_keys: list[str]
        A list of missing prediction keys.
    ignored_pred_keys: list[str]
        A list of ignored preiction keys.
    evaluation_id: int
        The job ID.
    """

    missing_pred_keys: list[str]
    ignored_pred_keys: list[str]
    evaluation_id: int


CreateEvaluationResponse = (
    CreateClfEvaluationResponse
    | CreateDetectionEvaluationResponse
    | CreateSemanticSegmentationEvaluationResponse
)


class Evaluation(BaseModel):
    """
    An object for storing the returned results of a model evaluation (where groundtruths are compared with predictions to measure performance).

    Attributes
    ----------
    dataset : str
        The name of the dataset.
    model : str
        The name of the model.
    settings : EvaluationSettings
        Settings for the evaluation.
    evaluation_id : int
        The ID of the evaluation job.
    status : str
        The status of the evaluation.
    metrics : List[Metric]
        A list of metrics associated with the evaluation.
    confusion_matrices: List[ConfusionMatrixResponse]
        A list of confusion matrices associated with the evaluation.

    """

    model: str
    settings: Filter
    evaluation_id: int
    status: EvaluationStatus
    metrics: list[Metric]
    confusion_matrices: list[ConfusionMatrixResponse]
    task_type: TaskType

    # pydantic setting
    model_config = ConfigDict(extra="forbid")
