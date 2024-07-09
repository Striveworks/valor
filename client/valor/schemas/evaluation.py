from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Union

from valor.enums import AnnotationType, MetricType, ROUGEType, TaskType
from valor.schemas.filters import Filter


@dataclass
class EvaluationParameters:
    """
    Defines parameters for evaluation methods.

    Attributes
    ----------
    task_type: TaskType
        The task type of a given evaluation.
    label_map: Optional[List[List[List[str]]]]
        Optional mapping of individual labels to a grouper label. Useful when you need to evaluate performance using labels that differ across datasets and models.
    metrics_to_return: List[MetricType], optional
        The list of metrics to compute, store, and return to the user.
    llm_api_params: Dict[str, str | dict], optional
        A dictionary of parameters for the LLM API.
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
    bleu_weights: list[float], optional
        The weights to use when calculating BLEU scores.
    rouge_types: list[ROUGEType]
        A list of rouge types to calculate. Options are ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], where `rouge1` is unigram-based scoring, `rouge2` is bigram-based scoring, `rougeL` is scoring based on sentences (i.e., splitting on "." and ignoring "\n"), and `rougeLsum` is scoring based on splitting the text using "\n".
    rouge_use_stemmer: bool
        If True, uses Porter stemmer to strip word suffixes.
    """

    task_type: TaskType
    label_map: Optional[List[List[List[str]]]] = None
    metrics_to_return: Optional[List[MetricType]] = None
    llm_api_params: Optional[Dict[str, Union[str, dict]]] = None

    convert_annotations_to_type: Optional[AnnotationType] = None
    iou_thresholds_to_compute: Optional[List[float]] = None
    iou_thresholds_to_return: Optional[List[float]] = None
    recall_score_threshold: float = 0
    pr_curve_iou_threshold: float = 0.5
    pr_curve_max_examples: int = 1
    bleu_weights: Optional[List[float]] = None
    rouge_types: Optional[List[ROUGEType]] = None
    rouge_use_stemmer: Optional[bool] = None


@dataclass
class EvaluationRequest:
    """
    An evaluation request.

    Defines important attributes of the API's `EvaluationRequest`.

    Attributes
    ----------
    dataset_names : List[str]
        The list of datasets we want to evaluate by name.
    model_names : List[str]
        The list of models we want to evaluate by name.
    filters : dict
        The filter object used to define what the model(s) is evaluating against.
    parameters : EvaluationParameters
        Any parameters that are used to modify an evaluation method.
    """

    dataset_names: Union[str, List[str]]
    model_names: Union[str, List[str]]
    parameters: EvaluationParameters
    filters: Filter = field(default_factory=Filter)

    def __post_init__(self):
        if isinstance(self.filters, dict):
            self.filters = Filter(**self.filters)
        elif self.filters is None:
            self.filters = Filter()

        if isinstance(self.parameters, dict):
            self.parameters = EvaluationParameters(**self.parameters)

    def to_dict(self) -> dict:
        """
        Converts the request into a JSON-compatible dictionary.
        """
        return {
            "dataset_names": self.dataset_names,
            "model_names": self.model_names,
            "parameters": asdict(self.parameters),
            "filters": self.filters.to_dict(),
        }
