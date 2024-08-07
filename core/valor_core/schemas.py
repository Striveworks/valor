import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from valor_core import enums, geometry


@dataclass
class Datum:
    """
    A class used to store information about a datum for either a 'GroundTruth' or a 'Prediction'.

    Attributes
    ----------
    uid : String
        The UID of the datum.
    metadata : Dictionary
        A dictionary of metadata that describes the datum.

    Examples
    --------
    >>> Datum(uid="uid1")
    >>> Datum(uid="uid1", metadata={})
    >>> Datum(uid="uid1", metadata={"foo": "bar", "pi": 3.14})
    """

    uid: Optional[str] = None
    metadata: Optional[dict] = None

    def __post_init__(
        self,
    ):

        if not isinstance(self.uid, (str, type(None))):
            raise TypeError(
                f"Expected 'uid' to be of type 'str' or 'None', got {type(self.uid).__name__}"
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
    score: Optional[float] = None

    def __post_init__(self):
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
    metadata: Dictionary
        A dictionary of metadata that describes the `Annotation`.
    labels: List[Label], optional
        A list of labels to use for the `Annotation`.
    bounding_box: geometry.Box
        A bounding box to assign to the `Annotation`.
    polygon: BoundingPolygon
        A polygon to assign to the `Annotation`.
    raster: Raster
        A raster to assign to the `Annotation`.
    embedding: List[float]
        An embedding, described by a list of values with type float and a maximum length of 16,000.
    is_instance: bool, optional
        A boolean describing whether we should treat the Raster attached to an annotation as an instance segmentation or not. If set to true, then the Annotation will be validated for use in object detection tasks. If set to false, then the Annotation will be validated for use in semantic segmentation tasks.
    implied_task_types: list[str], optional
        The validated task types that are applicable to each Annotation. Doesn't need to bet set by the user.

    Examples
    --------

    Classification
    >>> Annotation.create(
    ...     labels=[
    ...         Label(key="class", value="dog"),
    ...         Label(key="category", value="animal"),
    ...     ]
    ... )

    Object-Detection geometry.Box
    >>> annotation = Annotation(
    ...     labels=[Label(key="k1", value="v1")],
    ...     bounding_box=box2,
    ... )

    Object-Detection geometry.Polygon
    >>> annotation = Annotation(
    ...     labels=[Label(key="k1", value="v1")],
    ...     polygon=BoundingPolygon(...),
    ... )

     Raster
    >>> annotation = Annotation(
    ...     labels=[Label(key="k1", value="v1")],
    ...     raster=Raster(...),
    ...     is_instance=True
    ... )

    Object-Detection with all supported Geometries defined.
    >>> Annotation(
    ...     labels=[Label(key="k1", value="v1")],
    ...     bounding_box=geometry.Box(...),
    ...     polygon=BoundingPolygon(...),
    ...     raster=Raster(...),
    ...     is_instance=True,
    ... )

    Semantic-Segmentation Raster
    >>> annotation = Annotation(
    ...     labels=[Label(key="k1", value="v1")],
    ...     raster=Raster(...),
    ...     is_instance=False # or None
    ... )
    """

    labels: List[Label]
    metadata: Optional[dict] = None
    bounding_box: Optional[geometry.Box] = None
    polygon: Optional[Union[geometry.Polygon, geometry.Box]] = None
    raster: Optional[geometry.Raster] = None
    embedding: Optional[geometry.Embedding] = None
    is_instance: Optional[bool] = None
    implied_task_types: Optional[List[str]] = None

    def __post_init__(self):
        if not isinstance(self.labels, list):
            raise TypeError(
                f"Expected 'labels' to be of type 'list', got {type(self.labels).__name__}"
            )
        if not all(isinstance(label, Label) for label in self.labels):
            raise TypeError("All items in 'labels' must be of type 'Label'")

        if not isinstance(self.metadata, (dict, type(None))):
            raise TypeError(
                f"Expected 'metadata' to be of type 'dict' or 'None', got {type(self.metadata).__name__}"
            )

        if not isinstance(self.bounding_box, (geometry.Box, type(None))):
            raise TypeError(
                f"Expected 'bounding_box' to be of type 'geometry.Box' or 'None', got {type(self.bounding_box).__name__}"
            )

        if not isinstance(
            self.polygon, (geometry.Polygon, geometry.Box, type(None))
        ):
            raise TypeError(
                f"Expected 'polygon' to be of type 'geometry.Polygon' or 'None', got {type(self.polygon).__name__}"
            )

        if not isinstance(self.raster, (geometry.Raster, type(None))):
            raise TypeError(
                f"Expected 'raster' to be of type 'geometry.Raster' or 'None', got {type(self.raster).__name__}"
            )

        if not isinstance(self.embedding, (geometry.Embedding, type(None))):
            raise TypeError(
                f"Expected 'embedding' to be of type 'Embedding' or 'None', got {type(self.embedding).__name__}"
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


@dataclass
class EvaluationParameters:
    """
    Defines optional parameters for evaluation methods.

    Attributes
    ----------
    label_map: Optional[List[List[List[str]]]]
        Optional mapping of individual labels to a grouper label. Useful when you need to evaluate performance using labels that differ across datasets and models.
    metrics: List[str], optional
        The list of metrics to compute, store, and return to the user.
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

    label_map: Optional[Dict[Label, Label]] = None
    metrics_to_return: Optional[List[enums.MetricType]] = None
    iou_thresholds_to_compute: Optional[List[float]] = None
    iou_thresholds_to_return: Optional[List[float]] = None
    recall_score_threshold: float = 0.0
    pr_curve_iou_threshold: float = 0.5
    pr_curve_max_examples: int = 1


def __post_init__(self):
    if not isinstance(self.label_map, (dict, type(None))):
        raise TypeError(
            f"Expected 'label_map' to be of type 'dict' or 'None', got {type(self.label_map).__name__}"
        )
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


@dataclass
class Evaluation:
    parameters: EvaluationParameters
    metrics: List[Dict]
    confusion_matrices: Optional[List[Dict]]
    ignored_pred_labels: Optional[List[Label]]
    missing_pred_labels: Optional[List[Label]]
    meta: Optional[Dict] = None

    def __str__(self) -> str:
        """Dumps the object into a JSON formatted string."""
        return json.dumps(self.__dict__, indent=4)

    def __post_init__(self):
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
    annotations : List[Annotation]
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
        """
        Creates a ground truth.

        Parameters
        ----------
        datum : Datum
            The datum that the ground truth is operating over.
        annotations : List[Annotation]
            The list of ground truth annotations.
        """
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
    annotations : List[Annotation]
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


LabelMapType = Dict[Label, Label]
