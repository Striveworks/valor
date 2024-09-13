import json
from dataclasses import dataclass

from valor_core import enums, schemas


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
    """

    label_map: dict[schemas.Label, schemas.Label] | None = None
    metrics_to_return: list[enums.MetricType] | None = None
    iou_thresholds_to_compute: list[float] | None = None
    iou_thresholds_to_return: list[float] | None = None
    convert_annotations_to_type: enums.AnnotationType | None = None
    recall_score_threshold: float = 0.0
    pr_curve_iou_threshold: float = 0.5
    pr_curve_max_examples: int = 1

    def __post_init__(self):
        """Validate instantiated class."""

        if not isinstance(self.label_map, (dict, type(None))):
            raise TypeError(
                f"Expected 'label_map' to be of type 'dict' or 'None', got {type(self.label_map).__name__}"
            )
        if self.label_map and not isinstance(self.label_map, dict):
            raise TypeError("label_map should be a dictionary of Labels.")

        if self.label_map is not None and not all(
            isinstance(k, schemas.Label) and isinstance(v, schemas.Label)
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
            raise TypeError("All items in 'metrics' must be of type 'dict'.")

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
