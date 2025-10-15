from dataclasses import asdict, dataclass

import numpy as np
import pyarrow.compute as pc
import pyarrow.dataset as ds
from numpy.typing import NDArray

from valor_lite.object_detection.computation import (
    compute_pair_classifications,
)
from valor_lite.object_detection.evaluator import Evaluator as CachedEvaluator
from valor_lite.object_detection.loader import Loader as CachedLoader
from valor_lite.object_detection.metric import Metric, MetricType
from valor_lite.object_detection.utilities import (
    unpack_confusion_matrix_into_metric_list_legacy,
)

"""
Usage
-----

loader = DataLoader()
loader.add_bounding_boxes(
    groundtruths=groundtruths,
    predictions=predictions,
)
evaluator = loader.finalize()

metrics = evaluator.evaluate(iou_thresholds=[0.5])

ap_metrics = metrics[MetricType.AP]
ar_metrics = metrics[MetricType.AR]

filter_mask = evaluator.create_filter(datum_uids=["uid1", "uid2"])
filtered_metrics = evaluator.evaluate(iou_thresholds=[0.5], filter_mask=filter_mask)
"""


@dataclass
class Metadata:
    number_of_datums: int = 0
    number_of_ground_truths: int = 0
    number_of_predictions: int = 0
    number_of_labels: int = 0

    def to_dict(self) -> dict[str, int | bool]:
        return asdict(self)


@dataclass
class Filter:
    expr: pc.Expression


class Evaluator:
    """
    Legacy Object Detection Evaluator
    """

    def __init__(self):
        self._evaluator = CachedEvaluator(".valor")

    @property
    def metadata(self) -> Metadata:
        """
        Evaluation metadata.
        """
        return Metadata(
            number_of_datums=self._evaluator.info.number_of_datums,
            number_of_labels=self._evaluator.info.number_of_labels,
            number_of_ground_truths=self._evaluator.info.number_of_groundtruth_annotations,
            number_of_predictions=self._evaluator.info.number_of_prediction_annotations,
        )

    def create_filter(
        self,
        datums: list[str] | NDArray[np.int32] | None = None,
        groundtruths: list[str] | NDArray[np.int32] | None = None,
        predictions: list[str] | NDArray[np.int32] | None = None,
        labels: list[str] | NDArray[np.int32] | None = None,
    ) -> Filter:
        """
        Creates a filter object.

        Parameters
        ----------
        datum : list[str] | NDArray[int32], optional
            An optional list of string ids or indices representing datums to keep.
        groundtruths : list[str] | NDArray[int32], optional
            An optional list of string ids or indices representing ground truth annotations to keep.
        predictions : list[str] | NDArray[int32], optional
            An optional list of string ids or indices representing prediction annotations to keep.
        labels : list[str] | NDArray[int32], optional
            An optional list of labels or indices to keep.
        """
        filter_expr = pc.scalar(True)
        if datums is not None:
            if isinstance(datums, list):
                filter_expr &= ds.field("datum_uid").isin(datums + [None])
            else:
                filter_expr &= ds.field("datum_id").isin(
                    datums.tolist() + [-1]
                )
        if groundtruths is not None:
            if isinstance(groundtruths, list):
                filter_expr &= ds.field("gt_uid").isin(groundtruths + [None])
            else:
                filter_expr &= ds.field("gt_id").isin(
                    groundtruths.tolist() + [-1]
                )
        if predictions is not None:
            if isinstance(predictions, list):
                filter_expr &= ds.field("pd_uid").isin(predictions + [None])
            else:
                filter_expr &= ds.field("pd_id").isin(
                    predictions.tolist() + [-1]
                )
        if labels is not None:
            if isinstance(labels, list):
                filter_expr &= ds.field("pd_uid").isin(labels + [None])
            else:
                filter_expr &= ds.field("pd_id").isin(labels.tolist() + [-1])
        return Filter(expr=filter_expr)

    def compute_precision_recall(
        self,
        iou_thresholds: list[float],
        score_thresholds: list[float],
        filter_: Filter | None = None,
    ) -> dict[MetricType, list[Metric]]:
        """
        Computes all metrics except for ConfusionMatrix

        Parameters
        ----------
        iou_thresholds : list[float]
            A list of IOU thresholds to compute metrics over.
        score_thresholds : list[float]
            A list of score thresholds to compute metrics over.
        filter_ : Filter, optional
            A collection of filter parameters and masks.

        Returns
        -------
        dict[MetricType, list]
            A dictionary mapping MetricType enumerations to lists of computed metrics.
        """
        filter_expr = filter_.expr if filter_ else None
        return self._evaluator.compute_precision_recall(
            iou_thresholds=iou_thresholds,
            score_thresholds=score_thresholds,
            filter_expr=filter_expr,
        )

    def compute_confusion_matrix(
        self,
        iou_thresholds: list[float],
        score_thresholds: list[float],
        filter_: Filter | None = None,
    ) -> list[Metric]:
        """
        Computes confusion matrices at various thresholds.

        Parameters
        ----------
        iou_thresholds : list[float]
            A list of IOU thresholds to compute metrics over.
        score_thresholds : list[float]
            A list of score thresholds to compute metrics over.
        filter_ : Filter, optional
            A collection of filter parameters and masks.

        Returns
        -------
        list[Metric]
            List of confusion matrices per threshold pair.
        """
        if not iou_thresholds:
            raise ValueError("At least one IOU threshold must be passed.")
        elif not score_thresholds:
            raise ValueError("At least one score threshold must be passed.")

        # TODO - implement
        return []

        if filter_ is not None:
            detailed_pairs, _, _ = self.filter(filter_=filter_)
        else:
            detailed_pairs = self._detailed_pairs

        if detailed_pairs.size == 0:
            return []

        (
            mask_tp,
            mask_fp_fn_misclf,
            mask_fp_unmatched,
            mask_fn_unmatched,
        ) = compute_pair_classifications(
            detailed_pairs=detailed_pairs,
            iou_thresholds=np.array(iou_thresholds),
            score_thresholds=np.array(score_thresholds),
        )
        return unpack_confusion_matrix_into_metric_list_legacy(
            detailed_pairs=detailed_pairs,
            mask_tp=mask_tp,
            mask_fp_fn_misclf=mask_fp_fn_misclf,
            mask_fp_unmatched=mask_fp_unmatched,
            mask_fn_unmatched=mask_fn_unmatched,
            iou_thresholds=iou_thresholds,
            score_thresholds=score_thresholds,
            index_to_datum_id={
                i: v for i, v in enumerate(self.index_to_datum_id)
            },
            index_to_groundtruth_id={
                i: v for i, v in enumerate(self.index_to_groundtruth_id)
            },
            index_to_prediction_id={
                i: v for i, v in enumerate(self.index_to_prediction_id)
            },
            index_to_label={i: v for i, v in enumerate(self.index_to_label)},
        )

    def evaluate(
        self,
        iou_thresholds: list[float] = [0.1, 0.5, 0.75],
        score_thresholds: list[float] = [0.5],
        filter_: Filter | None = None,
    ) -> dict[MetricType, list[Metric]]:
        """
        Computes all available metrics.

        Parameters
        ----------
        iou_thresholds : list[float], default=[0.1, 0.5, 0.75]
            A list of IOU thresholds to compute metrics over.
        score_thresholds : list[float], default=[0.5]
            A list of score thresholds to compute metrics over.
        filter_ : Filter, optional
            A collection of filter parameters and masks.

        Returns
        -------
        dict[MetricType, list[Metric]]
            Lists of metrics organized by metric type.
        """
        metrics = self.compute_precision_recall(
            iou_thresholds=iou_thresholds,
            score_thresholds=score_thresholds,
            filter_=filter_,
        )
        metrics[MetricType.ConfusionMatrix] = self.compute_confusion_matrix(
            iou_thresholds=iou_thresholds,
            score_thresholds=score_thresholds,
            filter_=filter_,
        )
        return metrics


class DataLoader(CachedLoader):
    """
    Legacy Object Detection DataLoader
    """

    def __init__(self):
        super().__init__(
            ".valor",
            batch_size=1_000,
            rows_per_file=10_000,
        )

    def finalize(self) -> Evaluator:
        _ = super().finalize()
        return Evaluator()
