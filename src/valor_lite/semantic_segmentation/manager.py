from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.dataset as ds
from numpy.typing import NDArray

from valor_lite.exceptions import EmptyFilterError
from valor_lite.semantic_segmentation.evaluator import (
    Evaluator as CachedEvaluator,
)
from valor_lite.semantic_segmentation.evaluator import Filter
from valor_lite.semantic_segmentation.loader import Loader as CachedLoader

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

filter_mask = evaluator.create_filter(datum_ids=["uid1", "uid2"])
filtered_metrics = evaluator.evaluate(filter_mask=filter_mask)
"""


@dataclass
class Metadata:
    number_of_labels: int = 0
    number_of_pixels: int = 0
    number_of_datums: int = 0
    number_of_ground_truths: int = 0
    number_of_predictions: int = 0

    @classmethod
    def create(
        cls,
        confusion_matrices: NDArray[np.int64],
    ):
        if confusion_matrices.size == 0:
            return cls()
        return cls(
            number_of_labels=confusion_matrices.shape[1] - 1,
            number_of_pixels=confusion_matrices.sum(),
            number_of_datums=confusion_matrices.shape[0],
            number_of_ground_truths=confusion_matrices[:, 1:, :].sum(),
            number_of_predictions=confusion_matrices[:, :, 1:].sum(),
        )

    def to_dict(self) -> dict[str, int | bool]:
        return asdict(self)


class Evaluator(CachedEvaluator):
    """
    Segmentation Evaluator
    """

    @property
    def metadata(self) -> Metadata:
        return Metadata(
            number_of_labels=self._info.number_of_labels,
            number_of_pixels=self._info.number_of_pixels,
            number_of_datums=self._info.number_of_datums,
            number_of_ground_truths=self._info.number_of_groundtruth_pixels,
            number_of_predictions=self._info.number_of_prediction_pixels,
        )

    def create_filter(
        self,
        datums: list[str] | NDArray[np.int64] | None = None,
    ) -> Filter:
        """
        Creates a filter for use with the evaluator.

        Parameters
        ----------
        datums : list[str] | NDArray[int64], optional
            An optional list of string ids or array of indices representing datums.
        labels : list[str] | NDArray[int64], optional
            An optional list of labels or array of indices.

        Returns
        -------
        Filter
            The filter object containing a mask and metadata.
        """
        datum_expr = pc.scalar(True)
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

        return Filter(
            datums=datum_expr,
            groundtruths=None,
            predictions=None,
        )


class DataLoader(CachedLoader):
    """
    Segmentation DataLoader.
    """

    def finalize(self):
        evaluator = super().finalize()
        return Evaluator(
            name=evaluator._name,
            directory=evaluator._directory,
        )

    @classmethod
    def filter(
        cls,
        directory: str | Path,
        name: str,
        evaluator: CachedEvaluator,
        filter_expr: Filter,
    ) -> Evaluator:
        evaluator = super().filter(
            directory=directory,
            name=name,
            evaluator=evaluator,
            filter_expr=filter_expr,
        )
        return Evaluator(
            directory=evaluator._directory,
            name=evaluator._name,
            labels_override=evaluator._index_to_label,
        )
