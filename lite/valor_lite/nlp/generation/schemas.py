import json
import math
from dataclasses import dataclass

from valor_lite.nlp.generation import enums


@dataclass
class Datum:
    """
    A class used to store information about a datum for either a 'GroundTruth' or a 'Prediction'.

    Attributes
    ----------
    uid : str
        The UID of the datum.
    text : str, optional
        If the datum is a piece of text, then this field should contain the text.
    metadata : dict[str, Any]
        A dictionary of metadata that describes the datum.

    Examples
    --------
    >>> Datum(uid="uid1")
    >>> Datum(uid="uid1", metadata={})
    >>> Datum(uid="uid1", metadata={"foo": "bar", "pi": 3.14})
    >>> Datum(uid="uid2", text="What is the capital of Kenya?")
    """

    uid: str | None = None
    text: str | None = None
    metadata: dict | None = None

    def __post_init__(
        self,
    ):
        """Validate instantiated class."""

        if not isinstance(self.uid, (str, type(None))):
            raise TypeError(
                f"Expected 'uid' to be of type 'str' or 'None', got {type(self.uid).__name__}"
            )
        if not isinstance(self.text, (str, type(None))):
            raise TypeError(
                f"Expected 'text' to be of type 'str' or 'None', got {type(self.text).__name__}"
            )
        if not isinstance(self.metadata, (dict, type(None))):
            raise TypeError(
                f"Expected 'metadata' to be of type 'dict' or 'None', got {type(self.metadata).__name__}"
            )


@dataclass
class Label:
    """
    An object for labeling datasets, models, and annotations.

    Attributes
    ----------
    key : str
        The label key. (e.g. 'class', 'category')
    value : str
        The label's value. (e.g. 'dog', 'cat')
    score : float, optional
        A score assigned to the label in the case of a prediction.
    """

    key: str
    value: str
    score: float | None = None

    def __post_init__(self):
        """Validate instantiated class."""

        if not isinstance(self.key, str):
            raise TypeError(
                f"Expected 'key' to be of type 'str', got {type(self.key).__name__}"
            )

        if not isinstance(self.value, str):
            raise TypeError(
                f"Expected 'value' to be of type 'str', got {type(self.value).__name__}"
            )

        if self.score is not None and not isinstance(
            self.score,
            (
                float,
                int,
            ),
        ):
            raise TypeError(
                f"Expected 'score' to be of type 'float' or 'int' or 'None', got {type(self.score).__name__}"
            )

        # Ensure score is a float if provided as int
        if isinstance(self.score, int):
            self.score = float(self.score)

    def __eq__(self, other):
        """
        Defines how labels are compared to one another.

        Parameters
        ----------
        other : Label
            The object to compare with the label.

        Returns
        ----------
        bool
            A boolean describing whether the two objects are equal.
        """
        if (
            not hasattr(other, "key")
            or not hasattr(other, "key")
            or not hasattr(other, "score")
        ):
            return False

        # if the scores aren't the same type return False
        if (other.score is None) != (self.score is None):
            return False

        if self.score is None or other.score is None:
            scores_equal = other.score is None and self.score is None
        else:
            scores_equal = math.isclose(self.score, other.score)

        return (
            scores_equal
            and self.key == other.key
            and self.value == other.value
        )

    def __hash__(self) -> int:
        """
        Defines how a 'Label' is hashed.

        Returns
        ----------
        int
            The hashed 'Label'.
        """
        return hash(f"key:{self.key},value:{self.value},score:{self.score}")


@dataclass
class Annotation:
    """
    A class used to annotate `GroundTruths` and `Predictions`.

    Attributes
    ----------
    metadata: dict[str, Any]
        A dictionary of metadata that describes the `Annotation`.
    labels: list[Label], optional
        A list of labels to use for the `Annotation`.
    is_instance: bool, optional
        A boolean describing whether we should treat the Raster attached to an annotation as an instance segmentation or not. If set to true, then the Annotation will be validated for use in object detection tasks. If set to false, then the Annotation will be validated for use in semantic segmentation tasks.
    implied_task_types: list[str], optional
        The validated task types that are applicable to each Annotation. Doesn't need to bet set by the user.
    text: str, optional
        A piece of text to assign to the 'Annotation'.
    context_list: list[str], optional
        A list of contexts to assign to the 'Annotation'.

    Examples
    --------

    Text Generation Annotation with text and context_list. Not all text generation tasks require both text and context.
    >>> annotation = Annotation(
    ...     text="Abraham Lincoln was the 16th President of the United States.",
    ...     context_list=["Lincoln was elected the 16th president of the United States in 1860.", "Abraham Lincoln was born on February 12, 1809, in a one-room log cabin on the Sinking Spring Farm in Hardin County, Kentucky."],
    ... )
    """

    labels: list[Label] | None = None
    metadata: dict | None = None
    is_instance: bool | None = None
    implied_task_types: list[str] | None = None
    text: str | None = None
    context_list: list[str] | None = None

    def __post_init__(self):
        """Validate instantiated class."""

        if self.labels is not None:
            if not isinstance(self.labels, list):
                raise TypeError(
                    f"Expected 'labels' to be of type 'list' or 'None', got {type(self.labels).__name__}"
                )
            if not all(isinstance(label, Label) for label in self.labels):
                raise TypeError(
                    "All items in 'labels' must be of type 'Label'"
                )

        if not isinstance(self.metadata, (dict, type(None))):
            raise TypeError(
                f"Expected 'metadata' to be of type 'dict' or 'None', got {type(self.metadata).__name__}"
            )

        if not isinstance(self.is_instance, (bool, type(None))):
            raise TypeError(
                f"Expected 'is_instance' to be of type 'bool' or 'None', got {type(self.is_instance).__name__}"
            )

        if not isinstance(self.implied_task_types, (list, type(None))):
            raise TypeError(
                f"Expected 'implied_task_types' to be of type 'list' or 'None', got {type(self.implied_task_types).__name__}"
            )
        if self.implied_task_types is not None and not all(
            isinstance(task_type, str) for task_type in self.implied_task_types
        ):
            raise TypeError(
                "All items in 'implied_task_types' must be of type 'str'"
            )

        if not isinstance(self.text, (str, type(None))):
            raise TypeError(
                f"Expected 'text' to be of type 'str' or 'None', got {type(self.text).__name__}"
            )

        if self.context_list is not None:
            if not isinstance(self.context_list, list):
                raise TypeError(
                    f"Expected 'context_list' to be of type 'list' or 'None', got {type(self.context_list).__name__}"
                )

            if not all(
                isinstance(context, str) for context in self.context_list
            ):
                raise TypeError(
                    "All items in 'context_list' must be of type 'str'"
                )


@dataclass
class EvaluationParameters:
    """
    Defines optional parameters for evaluation methods.

    Attributes
    ----------
    label_map: list[list[list[str]]], optional
        Optional mapping of individual labels to a grouper label. Useful when you need to evaluate performance using labels that differ across datasets and models.
    metrics: list[str], optional
        The list of metrics to compute, store, and return to the user.
    iou_thresholds_to_compute: list[float], optional
        A list of floats describing which Intersection over Unions (IoUs) to use when calculating metrics (i.e., mAP).
    iou_thresholds_to_return: list[float], optional
        A list of floats describing which Intersection over Union (IoUs) thresholds to calculate a metric for. Must be a subset of `iou_thresholds_to_compute`.
    recall_score_threshold: float, default=0
        The confidence score threshold for use when determining whether to count a prediction as a true positive or not while calculating Average Recall.
    pr_curve_iou_threshold: float, optional
            The IOU threshold to use when calculating precision-recall curves for object detection tasks. Defaults to 0.5.
    pr_curve_max_examples: int
        The maximum number of datum examples to store when calculating PR curves.
    llm_api_params: dict[str, str | dict], optional
        A dictionary of parameters for the LLM API. Only required by some text generation metrics.
    metric_params: dict[str, dict], optional
        A dictionary of optional parameters to pass in to specific metrics.
    """

    label_map: dict[Label, Label] | None = None
    metrics_to_return: list[enums.MetricType] | None = None
    iou_thresholds_to_compute: list[float] | None = None
    iou_thresholds_to_return: list[float] | None = None
    convert_annotations_to_type: enums.AnnotationType | None = None
    recall_score_threshold: float = 0.0
    pr_curve_iou_threshold: float = 0.5
    pr_curve_max_examples: int = 1
    llm_api_params: dict[str, str | dict] | None = None
    metric_params: dict[str, dict] | None = None

    def __post_init__(self):
        """Validate instantiated class."""

        if not isinstance(self.label_map, (dict, type(None))):
            raise TypeError(
                f"Expected 'label_map' to be of type 'dict' or 'None', got {type(self.label_map).__name__}"
            )
        if self.label_map and not isinstance(self.label_map, dict):
            raise TypeError("label_map should be a dictionary of Labels.")

        if self.label_map is not None and not all(
            isinstance(k, Label) and isinstance(v, Label)
            for k, v in self.label_map.items()
        ):
            raise TypeError(
                "All keys and values in 'label_map' must be of type 'Label'"
            )

        if not isinstance(self.metrics_to_return, (list, type(None))):
            raise TypeError(
                f"Expected 'metrics_to_return' to be of type 'list' or 'None', got {type(self.metrics_to_return).__name__}"
            )
        if self.metrics_to_return is not None and not all(
            isinstance(metric, enums.MetricType)
            for metric in self.metrics_to_return
        ):
            raise TypeError(
                "All items in 'metrics_to_return' must be of type 'enums.MetricType'"
            )

        if not isinstance(self.iou_thresholds_to_compute, (list, type(None))):
            raise TypeError(
                f"Expected 'iou_thresholds_to_compute' to be of type 'list' or 'None', got {type(self.iou_thresholds_to_compute).__name__}"
            )
        if self.iou_thresholds_to_compute is not None and not all(
            isinstance(threshold, float)
            for threshold in self.iou_thresholds_to_compute
        ):
            raise TypeError(
                "All items in 'iou_thresholds_to_compute' must be of type 'float'"
            )

        if not isinstance(self.iou_thresholds_to_return, (list, type(None))):
            raise TypeError(
                f"Expected 'iou_thresholds_to_return' to be of type 'list' or 'None', got {type(self.iou_thresholds_to_return).__name__}"
            )
        if self.iou_thresholds_to_return is not None and not all(
            isinstance(threshold, float)
            for threshold in self.iou_thresholds_to_return
        ):
            raise TypeError(
                "All items in 'iou_thresholds_to_return' must be of type 'float'"
            )

        if not isinstance(self.recall_score_threshold, float):
            raise TypeError(
                f"Expected 'recall_score_threshold' to be of type 'float', got {type(self.recall_score_threshold).__name__}"
            )

        if not isinstance(self.pr_curve_iou_threshold, float):
            raise TypeError(
                f"Expected 'pr_curve_iou_threshold' to be of type 'float', got {type(self.pr_curve_iou_threshold).__name__}"
            )

        if not isinstance(self.pr_curve_max_examples, int):
            raise TypeError(
                f"Expected 'pr_curve_max_examples' to be of type 'int', got {type(self.pr_curve_max_examples).__name__}"
            )

        if self.llm_api_params is not None:
            if not isinstance(self.llm_api_params, dict):
                raise TypeError(
                    f"Expected 'llm_api_params' to be of type 'dict' or 'None', got {type(self.llm_api_params).__name__}"
                )
            if not all(
                isinstance(key, str) for key in self.llm_api_params.keys()
            ):
                raise TypeError(
                    "All keys in 'llm_api_params' must be of type 'str'"
                )

            if not all(
                isinstance(value, (str, dict))
                for value in self.llm_api_params.values()
            ):
                raise TypeError(
                    "All values in 'llm_api_params' must be of type 'str' or 'dict'"
                )

        if self.metric_params is not None:
            if not isinstance(self.metric_params, dict):
                raise TypeError(
                    f"Expected 'metric_params' to be of type 'dict' or 'None', got {type(self.llm_api_params).__name__}"
                )
            if not all(
                isinstance(key, str) for key in self.metric_params.keys()
            ):
                raise TypeError(
                    "All keys in 'metric_params' must be of type 'str'"
                )

            if not all(
                isinstance(value, dict)
                for value in self.metric_params.values()
            ):
                raise TypeError(
                    "All values in 'metric_params' must be of type 'dict'"
                )


@dataclass
class Evaluation:
    parameters: EvaluationParameters
    metrics: list[dict]
    confusion_matrices: list[dict] | None = None
    ignored_pred_labels: list[tuple[str, str]] | None = None
    missing_pred_labels: list[tuple[str, str]] | None = None
    meta: dict | None = None

    def __str__(self) -> str:
        """Dumps the object into a JSON formatted string."""
        return json.dumps(self.__dict__, indent=4)

    def __post_init__(self):
        """Validate instantiated class."""

        if not isinstance(self.parameters, EvaluationParameters):
            raise TypeError(
                f"Expected 'parameters' to be of type 'EvaluationParameters', got {type(self.parameters).__name__}"
            )

        if not isinstance(self.metrics, list):
            raise TypeError(
                f"Expected 'metrics' to be of type 'list', got {type(self.metrics).__name__}"
            )
        if not all(isinstance(metric, dict) for metric in self.metrics):
            raise TypeError("All items in 'metrics' must be of type 'dict'")

        if not isinstance(self.confusion_matrices, (list, type(None))):
            raise TypeError(
                f"Expected 'confusion_matrices' to be of type 'list' or 'None', got {type(self.confusion_matrices).__name__}"
            )
        if self.confusion_matrices is not None and not all(
            isinstance(cm, dict) for cm in self.confusion_matrices
        ):
            raise TypeError(
                "All items in 'confusion_matrices' must be of type 'dict'"
            )

        if not isinstance(self.meta, (dict, type(None))):
            raise TypeError(
                f"Expected 'meta' to be of type 'dict' or 'None', got {type(self.meta).__name__}"
            )

    def to_dict(self) -> dict:
        """
        Defines how a `valor.Evaluation` object is serialized into a dictionary.

        Returns
        ----------
        dict
            A dictionary describing an evaluation.
        """
        return {
            "parameters": self.parameters.__dict__,
            "metrics": self.metrics,
            "confusion_matrices": self.confusion_matrices,
            "ignored_pred_labels": self.ignored_pred_labels,
            "missing_pred_labels": self.missing_pred_labels,
            "meta": self.meta,
        }


@dataclass
class GroundTruth:
    """
    An object describing a ground truth (e.g., a human-drawn bounding box on an image).

    Attributes
    ----------
    datum : Datum
        The datum associated with the groundtruth.
    annotations : list[Annotation]
        The list of annotations associated with the groundtruth.

    Examples
    --------
    >>> GroundTruth(
    ...     datum=Datum(uid="uid1"),
    ...     annotations=[
    ...         Annotation(
    ...             labels=[Label(key="k1", value="v1")],
    ...         )
    ...     ]
    ... )
    """

    datum: Datum
    annotations: list[Annotation]

    def __post_init__(
        self,
    ):
        """Validate instantiated class."""

        if not isinstance(self.datum, Datum):
            raise TypeError(
                f"Expected 'datum' to be of type 'Datum', got {type(self.datum).__name__}"
            )

        if not isinstance(self.annotations, list):
            raise TypeError(
                f"Expected 'annotations' to be of type 'list', got {type(self.annotations).__name__}"
            )
        if not all(
            isinstance(annotation, Annotation)
            for annotation in self.annotations
        ):
            raise TypeError(
                "All items in 'annotations' must be of type 'Annotation'"
            )


@dataclass
class Prediction:
    """
    An object describing a prediction (e.g., a machine-drawn bounding box on an image).

    Attributes
    ----------
    datum : Datum
        The datum associated with the prediction.
    annotations : list[Annotation]
        The list of annotations associated with the prediction.

    Examples
    --------
    >>> Prediction(
    ...     datum=Datum(uid="uid1"),
    ...     annotations=[
    ...         Annotation(
    ...             labels=[
    ...                 Label(key="k1", value="v1", score=0.9),
    ...                 Label(key="k1", value="v1", score=0.1)
    ...             ],
    ...         )
    ...     ]
    ... )
    """

    datum: Datum
    annotations: list[Annotation]

    def __post_init__(self):
        """Validate instantiated class."""

        if not isinstance(self.datum, Datum):
            raise TypeError(
                f"Expected 'datum' to be of type 'Datum', got {type(self.datum).__name__}"
            )

        if not isinstance(self.annotations, list):
            raise TypeError(
                f"Expected 'annotations' to be of type 'list', got {type(self.annotations).__name__}"
            )
        if not all(
            isinstance(annotation, Annotation)
            for annotation in self.annotations
        ):
            raise TypeError(
                "All items in 'annotations' must be of type 'Annotation'"
            )
