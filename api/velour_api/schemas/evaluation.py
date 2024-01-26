from pydantic import BaseModel, ConfigDict, model_validator

from velour_api.enums import AnnotationType, EvaluationStatus, TaskType
from velour_api.schemas.filters import Filter
from velour_api.schemas.metrics import ConfusionMatrixResponse, Metric


class EvaluationParameters(BaseModel):
    """
    Defines parameters for evaluation methods.

    Attributes
    ----------
    convert_annotations_to_type: AnnotationType | None = None
        The type to convert all annotations to.
    iou_thresholds_to_compute : List[float], optional
        A list of floats describing which Intersection over Unions (IoUs) to use when calculating metrics (i.e., mAP).
    iou_thresholds_to_return: List[float], optional
        A list of floats describing which Intersection over Union (IoUs) thresholds to calculate a metric for. Must be a subset of `iou_thresholds_to_compute`.
    label_map : list
        Optional mapping of individual Labels to a grouper Label. Useful when you need to evaluate performance using Labels that differ across datasets and models.
    """

    task_type: TaskType

    # object detection
    convert_annotations_to_type: AnnotationType | None = None
    iou_thresholds_to_compute: list[float] | None = None
    iou_thresholds_to_return: list[float] | None = None
    label_map: list[list[list[str]]] | None = None

    # pydantic setting
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    @classmethod
    def _validate_by_task_type(cls, values):
        """Validate the IOU thresholds."""

        match values.task_type:
            case TaskType.CLASSIFICATION | TaskType.SEGMENTATION:
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
    """

    model_names: str | list[str]
    datum_filter: Filter
    parameters: EvaluationParameters

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
        The id of the evaluation.
    model_name : str
        The name of the evaluated model.
    datum_filter : schemas.Filter
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
    model_name: str
    datum_filter: Filter
    parameters: EvaluationParameters
    status: EvaluationStatus
    metrics: list[Metric] | None = None
    confusion_matrices: list[ConfusionMatrixResponse] | None = None

    # pydantic setting
    model_config = ConfigDict(
        extra="allow", protected_namespaces=("protected_",)
    )
