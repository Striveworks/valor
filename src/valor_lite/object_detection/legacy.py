import tempfile
from dataclasses import asdict, dataclass

import numpy as np
import pyarrow.compute as pc
import pyarrow.dataset as ds
from numpy.typing import NDArray

from valor_lite.exceptions import EmptyFilterError
from valor_lite.object_detection.evaluator import Evaluator as CachedEvaluator
from valor_lite.object_detection.evaluator import Filter
from valor_lite.object_detection.loader import Loader as CachedLoader
from valor_lite.object_detection.metric import Metric, MetricType

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


class Evaluator:
    """
    Legacy Object Detection Evaluator
    """

    def __init__(self, name: str = "default"):
        self._evaluator = CachedEvaluator(name=name)

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

    @property
    def _detailed_pairs(self) -> np.ndarray:
        return np.concatenate(
            [
                pairs
                for pairs in self._evaluator.iterate_pairs(
                    self._evaluator._dataset,
                    columns=[
                        "datum_id",
                        "gt_id",
                        "pd_id",
                        "gt_label_id",
                        "pd_label_id",
                        "iou",
                        "score",
                    ],
                )
            ]
        )

    @property
    def _label_metadata(self) -> np.ndarray:
        label_metadata = np.zeros(
            (len(self._evaluator._index_to_label), 2), dtype=np.int32
        )

        # groundtruth labels
        unique_ann = np.unique(
            self._detailed_pairs[:, (0, 1, 3)].astype(np.int64), axis=0
        )
        unique_labels, label_counts = np.unique(
            unique_ann[:, 2], return_counts=True
        )
        label_counts = label_counts[unique_labels >= 0]
        unique_labels = unique_labels[unique_labels >= 0]
        label_metadata[unique_labels, 0] = label_counts

        # prediction labels
        unique_ann = np.unique(
            self._detailed_pairs[:, (0, 2, 4)].astype(np.int64), axis=0
        )
        unique_labels, label_counts = np.unique(
            unique_ann[:, 2], return_counts=True
        )
        label_counts = label_counts[unique_labels >= 0]
        unique_labels = unique_labels[unique_labels >= 0]
        label_metadata[unique_labels, 1] = label_counts

        return label_metadata

    def filter(
        self, filter_: Filter
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int32],]:
        """
        Performs filtering over the internal cache.

        Parameters
        ----------
        filter_ : Filter
            The filter parameterization.

        Returns
        -------
        NDArray[float64]
            Filtered detailed pairs.
        NDArray[float64]
            Filtered ranked pairs.
        NDArray[int32]
            Label metadata.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = Evaluator()
            evaluator._evaluator = self._evaluator.filter(
                directory=tmpdir,
                name="filtered",
                filter_expr=filter_,
            )
            detailed_pairs = evaluator._detailed_pairs
            label_metadata = evaluator._label_metadata
            return detailed_pairs, detailed_pairs, label_metadata

    def create_filter(
        self,
        datums: list[str] | NDArray[np.int32] | None = None,
        groundtruths: list[str] | NDArray[np.int32] | None = None,
        predictions: list[str] | NDArray[np.int32] | None = None,
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
        """
        datum_expr = pc.scalar(True)
        gts_expr = pc.scalar(True)
        pds_expr = pc.scalar(True)
        if datums is not None:
            if isinstance(datums, list) and len(datums) > 0:
                datum_expr &= (
                    ds.field("datum_uid").isin(datums)
                    | ds.field("datum_uid").is_null()
                )
            elif isinstance(datums, np.ndarray) and datums.size > 0:
                datum_expr &= ds.field("datum_id").isin(datums.tolist() + [-1])
            else:
                raise EmptyFilterError("datum filter contains no elements")
        if groundtruths is not None:
            if isinstance(groundtruths, list) and len(groundtruths) > 0:
                gts_expr &= ds.field("gt_uid").isin(groundtruths)
            elif (
                isinstance(groundtruths, np.ndarray) and groundtruths.size > 0
            ):
                gts_expr &= ds.field("gt_id").isin(
                    groundtruths.tolist() + [-1]
                )
            else:
                raise EmptyFilterError(
                    "groundtruth filter contains no elements"
                )
        if predictions is not None:
            if isinstance(predictions, list) and len(predictions) > 0:
                pds_expr &= ds.field("pd_uid").isin(predictions)
            elif isinstance(predictions, np.ndarray) and predictions.size > 0:
                pds_expr &= ds.field("pd_id").isin(predictions.tolist() + [-1])
            else:
                raise EmptyFilterError(
                    "prediction filter contains no elements"
                )
        return Filter(
            datums=datum_expr,
            groundtruths=gts_expr,
            predictions=pds_expr,
        )

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
        if filter_ is not None:
            with tempfile.TemporaryDirectory() as tmpdir:
                evaluator = self._evaluator.filter(
                    directory=tmpdir,
                    name="filtered",
                    filter_expr=filter_,
                )
                return evaluator.compute_precision_recall(
                    iou_thresholds=iou_thresholds,
                    score_thresholds=score_thresholds,
                )
        return self._evaluator.compute_precision_recall(
            iou_thresholds=iou_thresholds,
            score_thresholds=score_thresholds,
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
        if filter_ is not None:
            with tempfile.TemporaryDirectory() as tmpdir:
                evaluator = self._evaluator.filter(
                    directory=tmpdir,
                    name="filtered",
                    filter_expr=filter_,
                )
                metrics = evaluator.compute_confusion_matrix_with_examples(
                    iou_thresholds=iou_thresholds,
                    score_thresholds=score_thresholds,
                )
        else:
            metrics = self._evaluator.compute_confusion_matrix_with_examples(
                iou_thresholds=iou_thresholds,
                score_thresholds=score_thresholds,
            )
        # retain legacy support
        for idx in range(len(metrics)):
            metrics[idx].type = MetricType.ConfusionMatrix.value
        return metrics

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
            batch_size=1_000,
            rows_per_file=10_000,
        )

    def finalize(self) -> Evaluator:  # type: ignore - switching type
        _ = super().finalize()
        return Evaluator()
