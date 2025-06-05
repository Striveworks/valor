from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from valor_lite.exceptions import EmptyEvaluatorError, EmptyFilterError
from valor_lite.semantic_segmentation.annotation import Segmentation
from valor_lite.semantic_segmentation.computation import (
    compute_intermediate_confusion_matrices,
    compute_label_metadata,
    compute_metrics,
    filter_cache,
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


@dataclass
class Filter:
    datum_mask: NDArray[np.bool_]
    label_mask: NDArray[np.bool_]
    metadata: Metadata

    def __post_init__(self):
        # validate datum mask
        if not self.datum_mask.any():
            raise EmptyFilterError("filter removes all datums")

        # validate label mask
        if self.label_mask.all():
            raise EmptyFilterError("filter removes all labels")


class Evaluator:
    """
    Segmentation Evaluator
    """

    def __init__(self):
        """Initializes evaluator caches."""
        # external references
        self.datum_id_to_index: dict[str, int] = {}
        self.index_to_datum_id: list[str] = []
        self.label_to_index: dict[str, int] = {}
        self.index_to_label: list[str] = []

        # internal caches
        self._confusion_matrices = np.array([], dtype=np.int64)
        self._label_metadata = np.array([], dtype=np.int64)
        self._metadata = Metadata()

    @property
    def metadata(self) -> Metadata:
        return self._metadata

    @property
    def ignored_prediction_labels(self) -> list[str]:
        """
        Prediction labels that are not present in the ground truth set.
        """
        glabels = set(np.where(self._label_metadata[:, 0] > 0)[0])
        plabels = set(np.where(self._label_metadata[:, 1] > 0)[0])
        return [
            self.index_to_label[label_id] for label_id in (plabels - glabels)
        ]

    @property
    def missing_prediction_labels(self) -> list[str]:
        """
        Ground truth labels that are not present in the prediction set.
        """
        glabels = set(np.where(self._label_metadata[:, 0] > 0)[0])
        plabels = set(np.where(self._label_metadata[:, 1] > 0)[0])
        return [
            self.index_to_label[label_id] for label_id in (glabels - plabels)
        ]

    def create_filter(
        self,
        datums: list[str] | NDArray[np.int64] | None = None,
        labels: list[str] | NDArray[np.int64] | None = None,
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
        datum_mask = np.ones(self._confusion_matrices.shape[0], dtype=np.bool_)
        label_mask = np.zeros(
            self.metadata.number_of_labels + 1, dtype=np.bool_
        )

        if datums is not None:
            # convert to indices
            if isinstance(datums, list):
                datums = np.array(
                    [self.datum_id_to_index[uid] for uid in datums],
                    dtype=np.int64,
                )

            # validate indices
            if datums.size == 0:
                raise EmptyFilterError(
                    "filter removes all datums"
                )  # validate indices
            elif datums.min() < 0:
                raise ValueError(
                    f"datum index cannot be negative '{datums.min()}'"
                )
            elif datums.max() >= len(self.index_to_datum_id):
                raise ValueError(
                    f"datum index cannot exceed total number of datums '{datums.max()}'"
                )

            # apply to mask
            datums.sort()
            mask_valid_datums = (
                np.arange(self._confusion_matrices.shape[0]).reshape(-1, 1)
                == datums.reshape(1, -1)
            ).any(axis=1)
            datum_mask[~mask_valid_datums] = False

        if labels is not None:
            # convert to indices
            if isinstance(labels, list):
                labels = np.array(
                    [self.label_to_index[label] for label in labels],
                    dtype=np.int64,
                )

            # validate indices
            if labels.size == 0:
                raise EmptyFilterError("filter removes all labels")
            elif labels.min() < 0:
                raise ValueError(
                    f"label index cannot be negative '{labels.min()}'"
                )
            elif labels.max() >= len(self.index_to_label):
                raise ValueError(
                    f"label index cannot exceed total number of labels '{labels.max()}'"
                )

            # apply to mask
            labels = np.concatenate([labels, np.array([-1])])
            label_range = np.arange(self.metadata.number_of_labels + 1) - 1
            mask_valid_labels = (
                label_range.reshape(-1, 1) == labels.reshape(1, -1)
            ).any(axis=1)
            label_mask[~mask_valid_labels] = True

        filtered_confusion_matrices, _ = filter_cache(
            confusion_matrices=self._confusion_matrices.copy(),
            datum_mask=datum_mask,
            label_mask=label_mask,
            number_of_labels=self.metadata.number_of_labels,
        )

        return Filter(
            datum_mask=datum_mask,
            label_mask=label_mask,
            metadata=Metadata.create(
                confusion_matrices=filtered_confusion_matrices,
            ),
        )

    def filter(
        self, filter_: Filter
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """
        Performs the filter operation over the internal cache.

        Parameters
        ----------
        filter_ : Filter
            An object describing the filter operation.

        Returns
        -------
        NDArray[int64]
            Filtered confusion matrices.
        NDArray[int64]
            Filtered label metadata
        """
        return filter_cache(
            confusion_matrices=self._confusion_matrices.copy(),
            datum_mask=filter_.datum_mask,
            label_mask=filter_.label_mask,
            number_of_labels=self.metadata.number_of_labels,
        )

    def compute_precision_recall_iou(
        self, filter_: Filter | None = None
    ) -> dict[MetricType, list]:
        """
        Performs an evaluation and returns metrics.

        Returns
        -------
        dict[MetricType, list]
            A dictionary mapping MetricType enumerations to lists of computed metrics.
        """
        if filter_ is not None:
            confusion_matrices, label_metadata = self.filter(filter_)
            n_pixels = filter_.metadata.number_of_pixels
        else:
            confusion_matrices = self._confusion_matrices
            label_metadata = self._label_metadata
            n_pixels = self.metadata.number_of_pixels

        results = compute_metrics(
            confusion_matrices=confusion_matrices,
            label_metadata=label_metadata,
            n_pixels=n_pixels,
        )
        return unpack_precision_recall_iou_into_metric_lists(
            results=results,
            label_metadata=label_metadata,
            index_to_label=self.index_to_label,
        )

    def evaluate(
        self, filter_: Filter | None = None
    ) -> dict[MetricType, list[Metric]]:
        """
        Computes all available metrics.

        Returns
        -------
        dict[MetricType, list[Metric]]
            Lists of metrics organized by metric type.
        """
        return self.compute_precision_recall_iou(filter_=filter_)


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
            raise ValueError(f"Datum with uid `{uid}` already exists.")
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

            if segmentation.groundtruths:
                combined_groundtruths = np.stack(
                    [
                        groundtruth.mask.flatten()
                        for groundtruth in segmentation.groundtruths
                    ],
                    axis=0,
                )
            else:
                combined_groundtruths = np.zeros(
                    (1, segmentation.shape[0] * segmentation.shape[1]),
                    dtype=np.bool_,
                )

            if segmentation.predictions:
                combined_predictions = np.stack(
                    [
                        prediction.mask.flatten()
                        for prediction in segmentation.predictions
                    ],
                    axis=0,
                )
            else:
                combined_predictions = np.zeros(
                    (1, segmentation.shape[0] * segmentation.shape[1]),
                    dtype=np.bool_,
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
            raise EmptyEvaluatorError()

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
        self._evaluator._metadata = Metadata.create(
            confusion_matrices=self._evaluator._confusion_matrices,
        )
        return self._evaluator
