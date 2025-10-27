import tempfile
from dataclasses import dataclass

import numpy as np
import pyarrow.compute as pc
from numpy.typing import NDArray

from valor_lite.classification.evaluator import Evaluator as CachedEvaluator
from valor_lite.classification.evaluator import Filter
from valor_lite.classification.loader import Loader as CachedLoader
from valor_lite.classification.metric import Metric, MetricType
from valor_lite.exceptions import EmptyFilterError

"""
Usage
-----

manager = DataLoader()
manager.add_data(
    groundtruths=groundtruths,
    predictions=predictions,
)
evaluator = manager.finalize()

metrics = evaluator.evaluate()

f1_metrics = metrics[MetricType.F1]
accuracy_metrics = metrics[MetricType.Accuracy]

filter_mask = evaluator.create_filter(datum_uids=["uid1", "uid2"])
filtered_metrics = evaluator.evaluate(filter_mask=filter_mask)
"""


@dataclass
class Metadata:
    number_of_datums: int = 0
    number_of_ground_truths: int = 0
    number_of_predictions: int = 0
    number_of_labels: int = 0


class Evaluator(CachedEvaluator):
    """
    Legacy Classification Evaluator
    """

    @property
    def metadata(self) -> Metadata:
        return Metadata(
            number_of_datums=self.info.number_of_datums,
            number_of_ground_truths=int(self._label_counts[:, 0].sum()),
            number_of_predictions=int(self._label_counts[:, 1].sum()),
            number_of_labels=self.info.number_of_labels,
        )

    def create_filter(
        self,
        datums: list[str] | NDArray[np.int32] | None = None,
    ) -> Filter:
        """
        Creates a filter object.

        Parameters
        ----------
        datums : list[str] | NDArray[int32], optional
            An optional list of string ids or indices representing datums to keep.
        """
        datum_expr = pc.scalar(True)
        if datums is not None:
            if isinstance(datums, list) and len(datums) > 0:
                datum_expr &= (
                    pc.field("datum_uid").isin(datums)
                    | pc.field("datum_uid").is_null()
                )
            elif isinstance(datums, np.ndarray) and datums.size > 0:
                datum_expr &= pc.field("datum_id").isin(datums.tolist() + [-1])
            else:
                raise EmptyFilterError("datum filter contains no elements")

        return Filter(
            datums=datum_expr,
            groundtruths=None,
            predictions=None,
        )

    def compute_precision_recall_rocauc(  # type: ignore[reportIncompatibleMethodOverride]
        self,
        score_thresholds: list[float] = [0.0],
        hardmax: bool = True,
        filter_: Filter | None = None,
    ) -> dict[MetricType, list]:
        """
        Performs an evaluation and returns metrics.

        Parameters
        ----------
        score_thresholds : list[float]
            A list of score thresholds to compute metrics over.
        hardmax : bool
            Toggles whether a hardmax is applied to predictions.
        filter_ : Filter, optional
            Applies a filter to the internal cache.

        Returns
        -------
        dict[MetricType, list]
            A dictionary mapping MetricType enumerations to lists of computed metrics.
        """
        if filter_ is not None:
            with tempfile.TemporaryDirectory() as tmpdir:
                evaluator = super().filter(
                    path=tmpdir,
                    filter_expr=filter_,
                )
                return evaluator.compute_precision_recall_rocauc(
                    score_thresholds=score_thresholds,
                    hardmax=hardmax,
                )
        return super().compute_precision_recall_rocauc(
            score_thresholds=score_thresholds,
            hardmax=hardmax,
        )

    def compute_confusion_matrix(
        self,
        score_thresholds: list[float] = [0.0],
        hardmax: bool = True,
        filter_: Filter | None = None,
    ) -> list[Metric]:
        """
        Computes a detailed confusion matrix..

        Parameters
        ----------
        score_thresholds : list[float]
            A list of score thresholds to compute metrics over.
        hardmax : bool
            Toggles whether a hardmax is applied to predictions.
        filter_ : Filter, optional
            Applies a filter to the internal cache.

        Returns
        -------
        list[Metric]
            A list of confusion matrices.
        """
        if filter_ is not None:
            with tempfile.TemporaryDirectory() as tmpdir:
                evaluator = super().filter(
                    path=tmpdir,
                    filter_expr=filter_,
                )
                metrics = evaluator.compute_confusion_matrix_with_examples(
                    score_thresholds=score_thresholds,
                    hardmax=hardmax,
                )
        else:
            metrics = super().compute_confusion_matrix_with_examples(
                score_thresholds=score_thresholds,
                hardmax=hardmax,
            )
        # retain legacy support by renaming metric
        for idx in range(len(metrics)):
            metrics[idx].type = MetricType.ConfusionMatrix.value
        return metrics

    def evaluate(
        self,
        score_thresholds: list[float] = [0.0],
        hardmax: bool = True,
        filter_: Filter | None = None,
    ) -> dict[MetricType, list[Metric]]:
        """
        Computes a detailed confusion matrix..

        Parameters
        ----------
        score_thresholds : list[float]
            A list of score thresholds to compute metrics over.
        hardmax : bool
            Toggles whether a hardmax is applied to predictions.
        filter_ : Filter, optional
            Applies a filter to the internal cache.

        Returns
        -------
        dict[MetricType, list[Metric]]
            Lists of metrics organized by metric type.
        """
        metrics = self.compute_precision_recall_rocauc(
            score_thresholds=score_thresholds,
            hardmax=hardmax,
            filter_=filter_,
        )
        metrics[MetricType.ConfusionMatrix] = self.compute_confusion_matrix(
            score_thresholds=score_thresholds,
            hardmax=hardmax,
            filter_=filter_,
        )
        return metrics


class DataLoader(CachedLoader):
    """
    Legacy Classification DataLoader.
    """

    def finalize(self) -> Evaluator:  # type: ignore - switching type
        evaluator = super().finalize()
        return Evaluator(
            path=evaluator.path,
            reader=evaluator.cache,
            info=evaluator.info,
            label_counts=evaluator._label_counts,
            index_to_label=evaluator._index_to_label,
        )
