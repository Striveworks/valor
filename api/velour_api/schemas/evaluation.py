from pydantic import BaseModel, ConfigDict, Field, model_validator

from velour_api.enums import AnnotationType, EvaluationStatus, TaskType
from velour_api.schemas.filters import Filter
from velour_api.schemas.metrics import ConfusionMatrixResponse, Metric


class EvaluationParameters(BaseModel):
    """
    Defines parameters for evaluation methods.

    Attributes
    ----------

    iou_thresholds_to_compute : List[float], optional
        A list of floats describing which Intersection over Unions (IoUs) to use when calculating metrics (i.e., mAP).
    iou_thresholds_to_return: List[float], optional
        A list of floats describing which Intersection over Union (IoUs) thresholds to calculate a metric for. Must be a subset of `iou_thresholds_to_compute`.
    """
    task_type: TaskType

    # object detection
    force_annotation_type: AnnotationType | None = None
    iou_thresholds_to_compute: list[float] | None = None
    iou_thresholds_to_return: list[float] | None = None

    # pydantic setting
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    @classmethod
    def _validate_by_task_type(cls, values):
        """Validate the IOU thresholds."""

        match values.task_type:
            case TaskType.CLASSIFICATION | TaskType.SEGMENTATION:
                if values.force_annotation_type is not None:
                    raise ValueError("`force_annotation_type` should only be used for object detection evaluations.")
                if values.iou_thresholds_to_compute is not None:
                    raise ValueError("`iou_thresholds_to_compute` should only be used for object detection evaluations.")
                if values.iou_thresholds_to_return is not None:
                    raise ValueError("`iou_thresholds_to_return` should only be used for object detection evaluations.")
            case TaskType.DETECTION:
                if values.iou_thresholds_to_return:
                    if not values.iou_thresholds_to_compute:
                        raise ValueError(
                            "`iou_thresholds_to_compute` must exist as a superset of `iou_thresholds_to_return`."
                        )
                    for iou in values.iou_thresholds_to_return:
                        if iou not in values.iou_thresholds_to_compute:
                            raise ValueError(
                                "`iou_thresholds_to_return` must be a subset of `iou_thresholds_to_compute`"
                            )
            case _:
                raise NotImplementedError(f"Task type `{values.task_type}` is unsupported.")
        return values


class EvaluationRequest(BaseModel):
    """
    Request for evaluation.

    A `model_filter` that returns multiple models will lead to the creation of multiple evaluations.

    Attributes
    ----------
    model_filter : schemas.Filter
        The filter used to enumerate all the models we want to evaluate.
    dataset_filter : schemas.Filter
        The filter object used to define what the model is evaluating against.
    parameters : DetectionParameters, optional
        Any parameters that are used to modify an evaluation method.
    """

    model_filter: Filter
    dataset_filter: Filter
    parameters: EvaluationParameters

    # pydantic setting
    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=("protected_",),
    )

    @model_validator(mode="after")
    @classmethod
    def _validate_no_task_type(cls, values):
        """Validate filters do not contain task type."""
        if values.dataset_filter.task_types is not None:
            raise ValueError("`dataset_filter` should not define the task_types constraint. Please set this in `parameters`.")
        if values.model_filter.task_types is not None:
            raise ValueError("`model_filter` should not define the task_types constraint. Please set this in `parameters`.")
        return values


class EvaluationResponse(BaseModel):
    """
    An object for storing the returned results of a model evaluation (where groundtruths are compared with predictions to measure performance).

    Attributes
    ----------
    id : int
        The id of the evaluation.
    model_filter : schemas.Filter
        The model filter used in the evaluation.
    dataset_filter : schemas.Filter
        The evaluation filter used in the evaluation.
    parameters : schemas.EvaluationParameters
        Any parameters used by the evaluation method.
    status : str
        The status of the evaluation.
    metrics : List[Metric]
        A list of metrics associated with the evaluation.
    confusion_matrices: List[ConfusionMatrixResponse]
        A list of confusion matrices associated with the evaluation.
    """

    id: int
    model_filter: Filter
    dataset_filter: Filter
    parameters: EvaluationParameters
    status: EvaluationStatus
    metrics: list[Metric]
    confusion_matrices: list[ConfusionMatrixResponse]

    # pydantic setting
    model_config = ConfigDict(
        extra="allow", protected_namespaces=("protected_",)
    )
