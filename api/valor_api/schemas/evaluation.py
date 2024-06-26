import datetime

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from valor_api.enums import (
    AnnotationType,
    EvaluationStatus,
    MetricType,
    TaskType,
)
from valor_api.schemas.filters import Filter
from valor_api.schemas.metrics import ConfusionMatrixResponse, Metric
from valor_api.schemas.migrations import DeprecatedFilter
from valor_api.schemas.types import Label

LabelMapType = list[list[list[str]]]


class EvaluationParameters(BaseModel):
    """
    Defines parameters for evaluation methods.

    Attributes
    ----------
    task_type: TaskType
        The task type of a given evaluation.
    label_map: Optional[List[List[List[str]]]]
        Optional mapping of individual labels to a grouper label. Useful when you need to evaluate performance using labels that differ across datasets and models.
    metrics_to_return: List[str], optional
        The list of metrics to compute, store, and return to the user.
    convert_annotations_to_type: AnnotationType | None = None
        The type to convert all annotations to.
    iou_thresholds_to_compute: List[float], optional
        A list of floats describing which Intersection over Unions (IoUs) to use when calculating metrics (i.e., mAP).
    iou_thresholds_to_return: List[float], optional
        A list of floats describing which Intersection over Union (IoUs) thresholds to calculate a metric for. Must be a subset of `iou_thresholds_to_compute`.
    recall_score_threshold: float, default=0
        The confidence score threshold for use when determining whether to count a prediction as a true positive or not while calculating Average Recall.
    pr_curve_iou_threshold: float, optional
            The IOU threshold to use when calculating precision-recall curves for object detection tasks. Defaults to 0.5.
    pr_curve_max_examples: int
        The maximum number of datum examples to store when calculating PR curves.
    """

    task_type: TaskType
    metrics_to_return: list[MetricType] | None = None
    label_map: LabelMapType | None = None

    convert_annotations_to_type: AnnotationType | None = None
    iou_thresholds_to_compute: list[float] | None = None
    iou_thresholds_to_return: list[float] | None = None
    recall_score_threshold: float | None = 0
    pr_curve_iou_threshold: float = 0.5
    pr_curve_max_examples: int = 1
    limit: int = -1

    # pydantic setting
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    @classmethod
    def _validate_parameters(cls, values):
        """Validate EvaluationParameters via type-specific checks."""

        # set default metrics for each task type
        if values.metrics_to_return is None:
            match values.task_type:
                case TaskType.CLASSIFICATION:
                    values.metrics_to_return = [
                        MetricType.Accuracy,
                        MetricType.Precision,
                        MetricType.Recall,
                        MetricType.F1,
                        MetricType.ROCAUC,
                    ]
                case TaskType.OBJECT_DETECTION:
                    values.metrics_to_return = [
                        MetricType.AP,
                        MetricType.AR,
                        MetricType.mAP,
                        MetricType.APAveragedOverIOUs,
                        MetricType.mAR,
                        MetricType.mAPAveragedOverIOUs,
                    ]
                case TaskType.SEMANTIC_SEGMENTATION:
                    values.metrics_to_return = [
                        MetricType.IOU,
                        MetricType.mIOU,
                    ]
                case TaskType.EMBEDDING:
                    values.metrics_to_return = [
                        MetricType.CramerVonMises,
                        MetricType.KolmgorovSmirnov,
                    ]

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
            case TaskType.EMBEDDING:
                pass
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
    dataset_names : list[str]
        The names of the evaluated datasets.
    model_names : str | list[str]
        The model(s) to evaluate.
    filters : schemas.Filter, optional
        The filter object used to define what data to evaluate.
    parameters : DetectionParameters, optional
        Any parameters that are used to modify an evaluation method.
    """

    dataset_names: list[str]
    model_names: list[str]
    filters: Filter = Filter()
    parameters: EvaluationParameters

    # pydantic setting
    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=("protected_",),
    )

    @field_validator("dataset_names")
    @classmethod
    def _validate_dataset_names(cls, v: list[str]) -> list[str]:
        if len(v) == 0:
            raise ValueError(
                "Evaluation request must contain at least one dataset name."
            )
        return v

    @field_validator("model_names")
    @classmethod
    def _validate_model_names(cls, v: list[str]) -> list[str]:
        if len(v) == 0:
            raise ValueError(
                "Evaluation request must contain at least one model name."
            )
        return v


class EvaluationResponse(BaseModel):
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
    dataset_names: list[str]
    model_name: str
    filters: Filter | DeprecatedFilter
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
