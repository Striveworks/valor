import json
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

from velour.enums import EvaluationStatus, TaskType
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
    label_map : Dict[Label, Label]
        Optional mapping of individual Labels to a grouper Label. Useful when you need to evaluate performance using Labels that differ across datasets and models.
    """

    parameters: Union[DetectionParameters, None] = None
    filters: Union[Filter, None] = None
    label_map: Union[dict, None] = None

    def __post_init__(self):
        if isinstance(self.parameters, dict):
            self.parameters = DetectionParameters(**self.parameters)
        if isinstance(self.filters, dict):
            self.filters = Filter(**self.filters)
        if isinstance(self.label_map, dict):
            self.label_map = tuple(
                (key.dict(), value.dict())
                for key, value in self.label_map.items()
            )


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
    task_type : TaskType
        The task type of the evaluation.
    settings : EvaluationSettings
        The `EvaluationSettings` object used to configurate the `EvaluationJob`.
    id : int
        The id of the job.
    """

    model: str
    dataset: str
    task_type: TaskType
    settings: EvaluationSettings = field(default_factory=EvaluationSettings)
    id: Optional[int] = None

    def __post_init__(self):
        if isinstance(self.settings, dict):
            self.settings = EvaluationSettings(**self.settings)
        if isinstance(self.task_type, str):
            self.task_type = TaskType(self.task_type)


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

    dataset: str
    model: str
    task_type: TaskType
    settings: EvaluationSettings
    evaluation_id: int
    status: EvaluationStatus
    metrics: List[dict]
    confusion_matrices: List[dict] = field(default_factory=list)

    def __post_init__(self):
        if isinstance(self.settings, dict):
            self.settings = EvaluationSettings(**self.settings)
        if isinstance(self.task_type, str):
            self.task_type = TaskType(self.task_type)
        if isinstance(self.status, str):
            self.status = EvaluationStatus(self.status)

    def to_dataframe(
        self,
        stratify_by: Tuple[str, str] = None,
    ):
        """
        Get all metrics associated with a Model and return them in a `pd.DataFrame`.

        Returns
        ----------
        pd.DataFrame
            Evaluation metrics being displayed in a `pd.DataFrame`.

        Raises
        ------
        ModuleNotFoundError
            This function requires the use of `pandas.DataFrame`.

        """
        try:
            import pandas as pd
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Must have pandas installed to use `get_metric_dataframes`."
            )

        if not stratify_by:
            column_type = "dataset"
            column_name = self.dataset
        else:
            column_type = stratify_by[0]
            column_name = stratify_by[1]

        metrics = [
            {**metric, column_type: column_name} for metric in self.metrics
        ]
        df = pd.DataFrame(metrics)
        for k in ["label", "parameters"]:
            df[k] = df[k].fillna("n/a")
        df["parameters"] = df["parameters"].apply(json.dumps)
        df["label"] = df["label"].apply(
            lambda x: f"{x['key']}: {x['value']}" if x != "n/a" else x
        )
        df = df.pivot(
            index=["type", "parameters", "label"], columns=[column_type]
        )
        return df
