from pydantic import BaseModel, ConfigDict, Field, model_validator

from velour_api.enums import EvaluationStatus, TaskType
from velour_api.schemas.filters import Filter
from velour_api.schemas.metrics import ConfusionMatrixResponse, Metric


class EvaluationParameters(BaseModel):
    """
    Defines important attributes to use when evaluating an object detection model.

    Attributes
    ----------
    iou_thresholds_to_compute : List[float]
        A list of floats describing which Intersection over Unions (IoUs) to use when calculating metrics (i.e., mAP).
    iou_thresholds_to_return: List[float]
        A list of floats describing which Intersection over Union (IoUs) thresholds to calculate a metric for. Must be a subset of `iou_thresholds_to_compute`.
    """

    # object detection
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
    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=('protected_',),
    )

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
            raise ValueError(
                "Evaluation requires the definition of `evaluation_filter.task_types`."
            )

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
    evaluation_filter : schemas.Filter
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
    evaluation_filter: Filter
    parameters: EvaluationParameters

    status: EvaluationStatus
    metrics: list[Metric]
    confusion_matrices: list[ConfusionMatrixResponse]

    # pydantic setting
    model_config = ConfigDict(
        extra="allow",
        protected_namespaces=('protected_',)
    )
    
