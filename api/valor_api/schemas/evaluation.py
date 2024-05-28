import datetime

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from valor_api.enums import AnnotationType, EvaluationStatus, TaskType
from valor_api.schemas.filters import Filter
from valor_api.schemas.metrics import ConfusionMatrixResponse, Metric
from valor_api.schemas.types import Label

LabelMapType = list[list[list[str]]]


class EvaluationParameters(BaseModel):
    """
    Defines parameters for evaluation methods.

    Attributes
    ----------
    convert_annotations_to_type: AnnotationType | None = None
        The type to convert all annotations to.
    metrics: List[str], optional
        The list of metrics to compute, store, and return to the user. Will return a default set of metrics is None is passed to the API.
    iou_thresholds_to_compute: List[float], optional
        A list of floats describing which Intersection over Unions (IoUs) to use when calculating metrics (i.e., mAP).
    iou_thresholds_to_return: List[float], optional
        A list of floats describing which Intersection over Union (IoUs) thresholds to calculate a metric for. Must be a subset of `iou_thresholds_to_compute`.
    label_map: LabelMapType, optional
        Optional mapping of individual labels to a grouper label. Useful when you need to evaluate performance using labels that differ across datasets and models.
    recall_score_threshold: float, optional
        The confidence score threshold for use when determining whether to count a prediction as a true positive or not while calculating Average Recall.
    pr_curve_iou_threshold: float, optional
        The IOU threshold to use when calculating precision-recall curves for object detection tasks. Defaults to 0.5.
    pr_curve_max_examples: int, optional
        The maximum number of datum examples to store when calculating PR curves.
    """

    task_type: TaskType
    metrics: list[
        str
    ]  # this parameter can technically be None, but it's converted to a default list by pydantic using get_metric_defaults
    convert_annotations_to_type: AnnotationType | None = None
    iou_thresholds_to_compute: list[float] | None = None
    iou_thresholds_to_return: list[float] | None = None
    label_map: LabelMapType | None = None
    recall_score_threshold: float | None = 0
    pr_curve_iou_threshold: float | None = 0.5
    pr_curve_max_examples: int | None = 1

    # pydantic setting
    model_config = ConfigDict(extra="forbid")

    @field_validator("metrics", mode="before")
    @classmethod
    def get_metric_defaults(cls, metrics, values) -> list[str]:
        """If the user passes None to `metrics`, then use a default list of metrics for the given task type instead."""
        if metrics is None:
            default_metrics = {
                TaskType.CLASSIFICATION: [
                    "Precision",
                    "Recall",
                    "F1",
                    "Accuracy",
                    "ROCAUC",
                ],
                TaskType.OBJECT_DETECTION: [
                    "AP",
                    "AR",
                    "mAP",
                    "APAveragedOverIOUs",
                    "mAR",
                    "mAPAveragedOverIOUs",
                ],
                TaskType.SEMANTIC_SEGMENTATION: ["IOU", "mIOU"],
            }

            return default_metrics[values.data["task_type"]]
        else:
            return metrics

    @model_validator(mode="after")
    @classmethod
    def _validate_parameters(cls, values):
        """Validate EvaluationParameters via type-specific checks."""

        match values.task_type:
            case TaskType.CLASSIFICATION | TaskType.SEMANTIC_SEGMENTATION:
                if values.convert_annotations_to_type is not None:
                    raise ValueError(
                        "`convert_annotations_to_type` should only be used for object detection evaluations."
                    )
                if values.iou_thresholds_to_compute is not None:
                    raise ValueError(
                        "`iou_thresholds_to_compute` should only be used for object detection evaluations."
                    )
                if values.iou_thresholds_to_return is not None:
                    raise ValueError(
                        "`iou_thresholds_to_return` should only be used for object detection evaluations."
                    )
            case TaskType.OBJECT_DETECTION:
                if not 0 <= values.pr_curve_iou_threshold <= 1:
                    raise ValueError(
                        "`pr_curve_iou_threshold` should be a float between 0 and 1 (inclusive)."
                    )
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
                raise NotImplementedError(
                    f"Task type `{values.task_type}` is unsupported."
                )
        return values


class EvaluationRequest(BaseModel):
    """
    Request for evaluation.

    Attributes
    ----------
    model_names : str | list[str]
        The model(s) to evaluate.
    datum_filter : schemas.Filter
        The filter object used to define what datums the model is evaluating over.
    parameters : DetectionParameters, optional
        Any parameters that are used to modify an evaluation method.
    meta: dict[str, str | int | float]
        Metadata about the evaluation run
    """

    model_names: list[str]
    datum_filter: Filter
    parameters: EvaluationParameters
    meta: dict[str, str | int | float] | None

    # pydantic setting
    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=("protected_",),
    )

    @model_validator(mode="after")
    @classmethod
    def _validate_request(cls, values):
        """Validate the request."""

        # verify filters do not contain task type.
        if values.datum_filter.task_types is not None:
            raise ValueError(
                "`datum_filter` should not define the task_types constraint. Please set this in evaluation `parameters`."
            )

        # verify `model_names` is of type list[str]
        if isinstance(values.model_names, list):
            if len(values.model_names) == 0:
                raise ValueError(
                    "`model_names` should specify at least one model."
                )
        elif isinstance(values.model_names, str):
            values.model_names = [values.model_names]

        return values


class EvaluationResponse(BaseModel):
    """
    An object for storing the returned results of a model evaluation (where groundtruths are compared with predictions to measure performance).

    Attributes
    ----------
    id : int
        The ID of the evaluation.
    model_name : str
        The name of the evaluated model.
    datum_filter : schemas.Filter
        The evaluation filter used in the evaluation.
    parameters : schemas.EvaluationParameters
        Any parameters used by the evaluation method.
    status : str
        The status of the evaluation.
    created_at: datetime.datetime
        The time the evaluation was created.
    metrics : List[Metric]
        A list of metrics associated with the evaluation.
    confusion_matrices: List[ConfusionMatrixResponse]
        A list of confusion matrices associated with the evaluation.
    missing_pred_labels: List[Label], optional
        A list of ground truth labels that aren't associated with any predictions.
    ignored_pred_labels: List[Label], optional
        A list of prediction labels that aren't associated with any ground truths.
    meta: dict[str, str | int | float]
        Metadata about the evaluation run.
    """

    id: int
    model_name: str
    datum_filter: Filter
    parameters: EvaluationParameters
    status: EvaluationStatus
    created_at: datetime.datetime
    meta: dict[str, str | int | float] | None
    metrics: list[Metric] | None = None
    confusion_matrices: list[ConfusionMatrixResponse] | None = None
    ignored_pred_labels: list[Label] | None = None
    missing_pred_labels: list[Label] | None = None

    # pydantic setting
    model_config = ConfigDict(
        extra="allow", protected_namespaces=("protected_",)
    )
