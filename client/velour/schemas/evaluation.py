from dataclasses import dataclass, field
from typing import List, Union

from velour.enums import EvaluationStatus
from velour.schemas.filters import Filter


@dataclass
class DetectionParameters:
    """
    Defines important attributes to use when evaluating an object detection model.

    Attributes
    ----------
    iou_thresholds_to_compute : List[float]
        A list of floats describing which Intersection over Unions (IoUs) to use when calculating metrics (i.e., mAP).
    iou_thresholds_to_return: List[float]
        A list of floats describing which Intersection over Union (IoUs) thresholds to calculate a metric for. Must be a subset of `iou_thresholds_to_compute`.
    """

    iou_thresholds_to_compute: List[float] = None
    iou_thresholds_to_return: List[float] = None


@dataclass
class EvaluationParameters:
    """
    Defines evaluation method parameters.

    Attributes
    ----------
    detection : DetectionParameters, optional
        Parameters for the detection evaluation method.
    """

    # classification = None
    detection: Union[DetectionParameters, None] = None
    # segmentation = None

    def __post_init__(self):
        if isinstance(self.detection, dict):
            self.detection = DetectionParameters(**self.detection)


@dataclass
class EvaluationRequest:
    """
    An evaluation request.

    Defines important attributes of the API's `EvaluationRequest`.

    Attributes
    ----------
    model_filter : schemas.Filter
        The filter used to enumerate all the models we want to evaluate.
    evaluation_filter : schemas.Filter
        The filter object used to define what the model(s) is evaluating against.
    parameters : EvaluationParameters
        Any parameters that are used to modify an evaluation method.
    """

    model_filter: Filter
    evaluation_filter: Filter
    parameters: EvaluationParameters = field(
        default_factory=EvaluationParameters
    )

    def __post_init__(self):
        if isinstance(self.model_filter, dict):
            self.model_filter = Filter(**self.model_filter)
        if isinstance(self.evaluation_filter, dict):
            self.evaluation_filter = Filter(**self.evaluation_filter)
        if isinstance(self.parameters, dict):
            self.parameters = EvaluationParameters(**self.parameters)


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
    task_type : TaskType
        The task type of the evaluation.
    settings : EvaluationSettings
        The `EvaluationSettings` object used to configurate the `EvaluationJob`.
    evaluation_id : int
        The id of the job.
    status : EvaluationStatus
        The status of the `EvaluationJob`.
    metrics : List[dict]
        A list of metric dictionaries returned by the job.
    confusion_matrices : List[dict]
        A list of confusion matrix dictionaries returned by the job.
    """

    evaluation_id: int
    model: str
    model_filter: Filter
    evaluation_filter: Filter
    parameters: EvaluationParameters
    status: EvaluationStatus
    metrics: List[dict]
    confusion_matrices: List[dict] = field(default_factory=list)

    def __post_init__(self):
        if isinstance(self.model_filter, dict):
            self.model_filter = Filter(**self.model_filter)
        if isinstance(self.evaluation_filter, dict):
            self.evaluation_filter = Filter(**self.evaluation_filter)
        if isinstance(self.parameters, dict):
            self.parameters = EvaluationParameters(**self.parameters)
        if isinstance(self.status, str):
            self.status = EvaluationStatus(self.status)

    # TODO
    # def to_dataframe(
    #     self,
    #     stratify_by: Tuple[str, str] = None,
    # ):
    #     """
    #     Get all metrics associated with a Model and return them in a `pd.DataFrame`.

    #     Returns
    #     ----------
    #     pd.DataFrame
    #         Evaluation metrics being displayed in a `pd.DataFrame`.

    #     Raises
    #     ------
    #     ModuleNotFoundError
    #         This function requires the use of `pandas.DataFrame`.

    #     """
    #     try:
    #         import pandas as pd
    #     except ModuleNotFoundError:
    #         raise ModuleNotFoundError(
    #             "Must have pandas installed to use `get_metric_dataframes`."
    #         )

    #     if not stratify_by:
    #         column_type = "dataset"
    #         column_name = self.dataset
    #     else:
    #         column_type = stratify_by[0]
    #         column_name = stratify_by[1]

    #     metrics = [
    #         {**metric, column_type: column_name} for metric in self.metrics
    #     ]
    #     df = pd.DataFrame(metrics)
    #     for k in ["label", "parameters"]:
    #         df[k] = df[k].fillna("n/a")
    #     df["parameters"] = df["parameters"].apply(json.dumps)
    #     df["label"] = df["label"].apply(
    #         lambda x: f"{x['key']}: {x['value']}" if x != "n/a" else x
    #     )
    #     df = df.pivot(
    #         index=["type", "parameters", "label"], columns=[column_type]
    #     )
    #     return df
