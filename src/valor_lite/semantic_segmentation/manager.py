import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from valor_lite.semantic_segmentation.annotation import Segmentation
from valor_lite.semantic_segmentation.computation import (
    compute_intermediate_confusion_matrices,
    compute_label_metadata,
    compute_metrics,
)
from valor_lite.semantic_segmentation.metric import Metric, MetricType
from valor_lite.semantic_segmentation.utilities import (
    unpack_precision_recall_iou_into_metric_lists,
)

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
class Filter:
    indices: NDArray[np.intp]
    label_metadata: NDArray[np.int64]
    n_pixels: int


class Evaluator:
    """
    Segmentation Evaluator
    """

    def __init__(self):

        # external references
        self.datum_id_to_index: dict[str, int] = {}
        self.label_to_index: dict[str, int] = {}

        self.index_to_datum_id: list[str] = []
        self.index_to_label: list[str] = []

        # internal caches
        self._confusion_matrices = np.array([], dtype=np.int64)
        self._label_metadata = np.array([], dtype=np.int64)
        self._n_pixels_per_datum = np.array([], dtype=np.int64)

        # filtered internal cache
        self._filtered_confusion_matrices: NDArray[np.int64] | None = None
        self._filtered_label_metadata: NDArray[np.int64] | None = None
        self._filtered_n_pixels_per_datum: NDArray[np.int64] | None = None

    @property
    def is_filtered(self) -> bool:
        return self._filtered_confusion_matrices is not None

    @property
    def label_metadata(self) -> NDArray[np.int64]:
        return (
            self._filtered_label_metadata
            if self._filtered_label_metadata is not None
            else self._label_metadata
        )

    @property
    def confusion_matrices(self) -> NDArray[np.int64]:
        return (
            self._filtered_confusion_matrices
            if self._filtered_confusion_matrices is not None
            else self._confusion_matrices
        )

    @property
    def n_labels(self) -> int:
        """Returns the total number of unique labels."""
        return len(self.index_to_label)

    @property
    def n_pixels(self) -> int:
        """Returns the total number of evaluated pixels."""
        return self.confusion_matrices.sum()

    @property
    def n_datums(self) -> int:
        """Returns the number of datums."""
        return self.confusion_matrices.shape[0]

    @property
    def n_groundtruths(self) -> int:
        """Returns the number of ground truth annotations."""
        return self.confusion_matrices[:, 1:, :].sum()

    @property
    def n_predictions(self) -> int:
        """Returns the number of prediction annotations."""
        return self.confusion_matrices[:, :, 1:].sum()

    @property
    def ignored_prediction_labels(self) -> list[str]:
        """
        Prediction labels that are not present in the ground truth set.
        """
        glabels = set(np.where(self.label_metadata[:, 0] > 0)[0])
        plabels = set(np.where(self.label_metadata[:, 1] > 0)[0])
        return [
            self.index_to_label[label_id] for label_id in (plabels - glabels)
        ]

    @property
    def missing_prediction_labels(self) -> list[str]:
        """
        Ground truth labels that are not present in the prediction set.
        """
        glabels = set(np.where(self.label_metadata[:, 0] > 0)[0])
        plabels = set(np.where(self.label_metadata[:, 1] > 0)[0])
        return [
            self.index_to_label[label_id] for label_id in (glabels - plabels)
        ]

    @property
    def metadata(self) -> dict:
        """
        Evaluation metadata.
        """
        return {
            "number_of_datums": self.n_datums,
            "number_of_groundtruths": self.n_groundtruths,
            "number_of_predictions": self.n_predictions,
            "number_of_labels": self.n_labels,
            "number_of_pixels": self.n_pixels,
            "ignored_prediction_labels": self.ignored_prediction_labels,
            "missing_prediction_labels": self.missing_prediction_labels,
            "is_filtered": self.is_filtered,
        }

    def apply_filter(
        self,
        datum_ids: list[str] | None = None,
        labels: list[str] | None = None,
    ):
        """
        Apply a filter on the evaluator.

        Can be reset by calling 'clear_filter'.

        Parameters
        ----------
        datum_ids : list[str], optional
            An optional list of string uids representing datums.
        labels : list[str], optional
            An optional list of labels.
        """
        self._filtered_confusion_matrices = self._confusion_matrices.copy()
        self._filtered_label_metadata = np.zeros(
            (self.n_labels, 2), dtype=np.int64
        )

        mask_datums = np.ones(
            self._filtered_confusion_matrices.shape[0], dtype=np.bool_
        )

        if datum_ids is not None:
            if not datum_ids:
                self._filtered_confusion_matrices = np.array(
                    [], dtype=np.int64
                )
                warnings.warn("datum filter results in empty data array")
                return
            datum_id_array = np.array(
                [self.datum_id_to_index[uid] for uid in datum_ids],
                dtype=np.int64,
            )
            datum_id_array.sort()
            mask_valid_datums = (
                np.arange(self._filtered_confusion_matrices.shape[0]).reshape(
                    -1, 1
                )
                == datum_id_array.reshape(1, -1)
            ).any(axis=1)
            mask_datums[~mask_valid_datums] = False

        if labels is not None:
            if not labels:
                self._filtered_confusion_matrices = np.array(
                    [], dtype=np.int64
                )
                warnings.warn("label filter results in empty data array")
                return
            labels_id_array = np.array(
                [self.label_to_index[label] for label in labels] + [-1],
                dtype=np.int64,
            )
            label_range = np.arange(self.n_labels + 1) - 1
            mask_valid_labels = (
                label_range.reshape(-1, 1) == labels_id_array.reshape(1, -1)
            ).any(axis=1)
            mask_invalid_labels = ~mask_valid_labels

            # add filtered labels to background
            null_predictions = self._filtered_confusion_matrices[
                :, mask_invalid_labels, :
            ].sum(axis=(1, 2))
            null_groundtruths = self._filtered_confusion_matrices[
                :, :, mask_invalid_labels
            ].sum(axis=(1, 2))
            null_intersection = (
                self._filtered_confusion_matrices[
                    :, mask_invalid_labels, mask_invalid_labels
                ]
                .reshape(self._filtered_confusion_matrices.shape[0], -1)
                .sum(axis=1)
            )
            self._filtered_confusion_matrices[:, 0, 0] += (
                null_groundtruths + null_predictions - null_intersection
            )
            self._filtered_confusion_matrices[:, mask_invalid_labels, :] = 0
            self._filtered_confusion_matrices[:, :, mask_invalid_labels] = 0

        self._filtered_confusion_matrices = self._filtered_confusion_matrices[
            mask_datums
        ]

        self._filtered_label_metadata = compute_label_metadata(
            confusion_matrices=self._filtered_confusion_matrices,
            n_labels=self.n_labels,
        )

    def clear_filter(self):
        """
        Clears any applied filters.
        """
        self._filtered_confusion_matrices = None
        self._filtered_label_metadata = None

    def compute_precision_recall_iou(self) -> dict[MetricType, list]:
        """
        Performs an evaluation and returns metrics.

        Returns
        -------
        dict[MetricType, list]
            A dictionary mapping MetricType enumerations to lists of computed metrics.
        """
        label_metadata = self.label_metadata
        results = compute_metrics(
            confusion_matrices=self.confusion_matrices,
            label_metadata=label_metadata,
            n_pixels=self.n_pixels,
        )
        return unpack_precision_recall_iou_into_metric_lists(
            results=results,
            label_metadata=label_metadata,
            index_to_label=self.index_to_label,
        )

    def evaluate(self) -> dict[MetricType, list[Metric]]:
        """
        Computes all available metrics.

        Returns
        -------
        dict[MetricType, list[Metric]]
            Lists of metrics organized by metric type.
        """
        return self.compute_precision_recall_iou()


class DataLoader:
    """
    Segmentation DataLoader.
    """

    def __init__(self):
        self._evaluator = Evaluator()
        self.matrices = list()

    def _add_datum(self, uid: str) -> int:
        """
        Helper function for adding a datum to the cache.

        Parameters
        ----------
        uid : str
            The datum uid.

        Returns
        -------
        int
            The datum index.
        """
        if uid in self._evaluator.datum_id_to_index:
            raise ValueError(f"Datum with uid `{uid}` has already been added.")
        index = len(self._evaluator.datum_id_to_index)
        self._evaluator.datum_id_to_index[uid] = index
        self._evaluator.index_to_datum_id.append(uid)
        return index

    def _add_label(self, label: str) -> int:
        """
        Helper function for adding a label to the cache.

        Parameters
        ----------
        label : str
            A string label.

        Returns
        -------
        int
            The label's index.
        """
        if label not in self._evaluator.label_to_index:
            label_id = len(self._evaluator.index_to_label)
            self._evaluator.label_to_index[label] = label_id
            self._evaluator.index_to_label.append(label)
        return self._evaluator.label_to_index[label]

    def add_data(
        self,
        segmentations: list[Segmentation],
        show_progress: bool = False,
    ):
        """
        Adds segmentations to the cache.

        Parameters
        ----------
        segmentations : list[Segmentation]
            A list of Segmentation objects.
        show_progress : bool, default=False
            Toggle for tqdm progress bar.
        """

        disable_tqdm = not show_progress
        for segmentation in tqdm(segmentations, disable=disable_tqdm):
            # update datum cache
            if segmentation.size == 0:
                warnings.warn(
                    f"skipping datum '{segmentation.uid}' as it contains no mask information."
                )
                continue

            self._add_datum(segmentation.uid)

            groundtruth_labels = -1 * np.ones(
                len(segmentation.groundtruths), dtype=np.int64
            )
            for idx, groundtruth in enumerate(segmentation.groundtruths):
                label_idx = self._add_label(groundtruth.label)
                groundtruth_labels[idx] = label_idx

            prediction_labels = -1 * np.ones(
                len(segmentation.predictions), dtype=np.int64
            )
            for idx, prediction in enumerate(segmentation.predictions):
                label_idx = self._add_label(prediction.label)
                prediction_labels[idx] = label_idx

            combined_groundtruths = np.stack(
                [
                    groundtruth.mask.flatten()
                    for groundtruth in segmentation.groundtruths
                ],
                axis=0,
            )
            combined_predictions = np.stack(
                [
                    prediction.mask.flatten()
                    for prediction in segmentation.predictions
                ],
                axis=0,
            )

            self.matrices.append(
                compute_intermediate_confusion_matrices(
                    groundtruths=combined_groundtruths,
                    predictions=combined_predictions,
                    groundtruth_labels=groundtruth_labels,
                    prediction_labels=prediction_labels,
                    n_labels=len(self._evaluator.index_to_label),
                )
            )

    def finalize(self) -> Evaluator:
        """
        Performs data finalization and some preprocessing steps.

        Returns
        -------
        Evaluator
            A ready-to-use evaluator object.
        """

        if len(self.matrices) == 0:
            raise ValueError("No data available to create evaluator.")

        n_labels = len(self._evaluator.index_to_label)
        n_datums = len(self._evaluator.index_to_datum_id)
        self._evaluator._confusion_matrices = np.zeros(
            (n_datums, n_labels + 1, n_labels + 1), dtype=np.int64
        )
        for idx, matrix in enumerate(self.matrices):
            h, w = matrix.shape
            self._evaluator._confusion_matrices[idx, :h, :w] = matrix
        self._evaluator._label_metadata = compute_label_metadata(
            confusion_matrices=self._evaluator._confusion_matrices,
            n_labels=n_labels,
        )

        return self._evaluator
