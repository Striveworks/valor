from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from valor_lite.semantic_segmentation.annotation import Segmentation
from valor_lite.semantic_segmentation.computation import (
    compute_intermediate_confusion_matrices,
    compute_metrics,
)
from valor_lite.semantic_segmentation.metric import (
    F1,
    Accuracy,
    ConfusionMatrix,
    IoU,
    MetricType,
    Precision,
    Recall,
    mIoU,
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

filter_mask = evaluator.create_filter(datum_uids=["uid1", "uid2"])
filtered_metrics = evaluator.evaluate(filter_mask=filter_mask)
"""


@dataclass
class Filter:
    indices: NDArray[np.int32]
    label_metadata: NDArray[np.int32]
    n_pixels: int


class Evaluator:
    """
    Segmentation Evaluator
    """

    def __init__(self):

        # metadata
        self.n_datums = 0
        self.n_groundtruths = 0
        self.n_predictions = 0
        self.n_pixels = 0
        self.n_groundtruth_pixels = 0
        self.n_prediction_pixels = 0
        self.n_labels = 0

        # datum reference
        self.uid_to_index: dict[str, int] = dict()
        self.index_to_uid: dict[int, str] = dict()

        # label reference
        self.label_to_index: dict[str, int] = dict()
        self.index_to_label: dict[int, str] = dict()

        # computation caches
        self._confusion_matrices = np.array([])
        self._label_metadata = np.array([], dtype=np.int32)
        self._label_metadata_per_datum = np.array([], dtype=np.int32)
        self._n_pixels_per_datum = np.array([], dtype=np.int32)

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

    @property
    def metadata(self) -> dict:
        """
        Evaluation metadata.
        """
        return {
            "number_of_datums": self.n_datums,
            "number_of_groundtruths": self.n_groundtruths,
            "number_of_predictions": self.n_predictions,
            "number_of_groundtruth_pixels": self.n_groundtruth_pixels,
            "number_of_prediction_pixels": self.n_prediction_pixels,
            "number_of_labels": self.n_labels,
            "ignored_prediction_labels": self.ignored_prediction_labels,
            "missing_prediction_labels": self.missing_prediction_labels,
        }

    def create_filter(
        self,
        datum_uids: list[str] | NDArray[np.int32] | None = None,
        labels: list[str] | NDArray[np.int32] | None = None,
    ) -> Filter:
        """
        Creates a boolean mask that can be passed to an evaluation.

        Parameters
        ----------
        datum_uids : list[str] | NDArray[np.int32], optional
            An optional list of string uids or a numpy array of uid indices.
        labels : list[tuple[str, str]] | NDArray[np.int32], optional
            An optional list of labels or a numpy array of label indices.

        Returns
        -------
        Filter
            A filter object that can be passed to the `evaluate` method.
        """
        n_datums = self._label_metadata_per_datum.shape[1]
        n_labels = self._label_metadata_per_datum.shape[2]

        mask_datums = np.ones(n_datums, dtype=np.bool_)
        mask_labels = np.ones(n_labels, dtype=np.bool_)

        if datum_uids is not None:
            if isinstance(datum_uids, list):
                datum_uids = np.array(
                    [self.uid_to_index[uid] for uid in datum_uids],
                    dtype=np.int32,
                )
            if datum_uids.size == 0:
                mask_datums[mask_datums] = False
            else:
                mask = (
                    np.arange(n_datums).reshape(-1, 1)
                    == datum_uids.reshape(1, -1)
                ).any(axis=1)
                mask_datums[~mask] = False

        if labels is not None:
            if isinstance(labels, list):
                labels = np.array(
                    [self.label_to_index[label] for label in labels],
                    dtype=np.int32,
                )
            if labels.size == 0:
                mask_labels[mask_labels] = False
            else:
                mask = (
                    np.arange(n_labels).reshape(-1, 1) == labels.reshape(1, -1)
                ).any(axis=1)
                mask_labels[~mask] = False

        mask = mask_datums[:, np.newaxis] & mask_labels[np.newaxis, :]
        label_metadata_per_datum = self._label_metadata_per_datum.copy()
        label_metadata_per_datum[:, ~mask] = 0

        label_metadata = np.zeros_like(self._label_metadata, dtype=np.int32)
        label_metadata = np.transpose(
            np.sum(
                label_metadata_per_datum,
                axis=1,
            )
        )
        n_datums = int(np.sum(label_metadata[:, 0]))

        return Filter(
            indices=np.where(mask_datums)[0],
            label_metadata=label_metadata,
            n_pixels=self._n_pixels_per_datum[mask_datums].sum(),
        )

    def compute_precision_recall_iou(
        self,
        filter_: Filter | None = None,
        as_dict: bool = False,
    ) -> dict[MetricType, list]:
        """
        Performs an evaluation and returns metrics.

        Parameters
        ----------
        filter_ : Filter, optional
            An optional filter object.
        as_dict : bool, default=False
            An option to return metrics as dictionaries.

        Returns
        -------
        dict[MetricType, list]
            A dictionary mapping MetricType enumerations to lists of computed metrics.
        """

        # apply filters
        data = self._confusion_matrices
        label_metadata = self._label_metadata
        n_pixels = self.n_pixels
        if filter_ is not None:
            data = data[filter_.indices]
            label_metadata = filter_.label_metadata
            n_pixels = filter_.n_pixels

        (
            precision,
            recall,
            f1_score,
            accuracy,
            ious,
            hallucination_ratios,
            missing_prediction_ratios,
        ) = compute_metrics(
            data=data,
            label_metadata=label_metadata,
            n_pixels=n_pixels,
        )

        metrics = defaultdict(list)

        metrics[MetricType.Accuracy] = [
            Accuracy(
                value=float(accuracy),
            )
        ]

        metrics[MetricType.ConfusionMatrix] = [
            ConfusionMatrix(
                confusion_matrix={
                    self.index_to_label[gt_label_idx]: {
                        self.index_to_label[pd_label_idx]: {
                            "iou": float(ious[gt_label_idx, pd_label_idx])
                        }
                        for pd_label_idx in range(self.n_labels)
                        if label_metadata[pd_label_idx, 0] > 0
                    }
                    for gt_label_idx in range(self.n_labels)
                    if label_metadata[gt_label_idx, 0] > 0
                },
                hallucinations={
                    self.index_to_label[pd_label_idx]: {
                        "ratio": float(hallucination_ratios[pd_label_idx])
                    }
                    for pd_label_idx in range(self.n_labels)
                    if label_metadata[pd_label_idx, 0] > 0
                },
                missing_predictions={
                    self.index_to_label[gt_label_idx]: {
                        "ratio": float(missing_prediction_ratios[gt_label_idx])
                    }
                    for gt_label_idx in range(self.n_labels)
                    if label_metadata[gt_label_idx, 0] > 0
                },
            )
        ]

        metrics[MetricType.mIoU] = [
            mIoU(
                value=float(ious.diagonal().mean()),
            )
        ]

        for label_idx, label in self.index_to_label.items():

            kwargs = {
                "label": label,
            }

            # if no groundtruths exists for a label, skip it.
            if label_metadata[label_idx, 0] == 0:
                continue

            metrics[MetricType.Precision].append(
                Precision(
                    value=float(precision[label_idx]),
                    **kwargs,
                )
            )
            metrics[MetricType.Recall].append(
                Recall(
                    value=float(recall[label_idx]),
                    **kwargs,
                )
            )
            metrics[MetricType.F1].append(
                F1(
                    value=float(f1_score[label_idx]),
                    **kwargs,
                )
            )
            metrics[MetricType.IoU].append(
                IoU(
                    value=float(ious[label_idx, label_idx]),
                    **kwargs,
                )
            )

        if as_dict:
            return {
                mtype: [metric.to_dict() for metric in mvalues]
                for mtype, mvalues in metrics.items()
            }

        return metrics

    def evaluate(
        self,
        filter_: Filter | None = None,
        as_dict: bool = False,
    ) -> dict[MetricType, list]:
        """
        Computes all available metrics.

        Parameters
        ----------
        filter_ : Filter, optional
            An optional filter object.
        as_dict : bool, default=False
            An option to return metrics as dictionaries.

        Returns
        -------
        dict[MetricType, list]
            A dictionary mapping metric type to lists of metrics.
        """
        return self.compute_precision_recall_iou(
            filter_=filter_, as_dict=as_dict
        )


class DataLoader:
    """
    Segmentation DataLoader.
    """

    def __init__(self):
        self._evaluator = Evaluator()
        self.groundtruth_count = defaultdict(lambda: defaultdict(int))
        self.prediction_count = defaultdict(lambda: defaultdict(int))
        self.matrices = list()
        self.pixel_count = list()

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
        if uid in self._evaluator.uid_to_index:
            raise ValueError(f"Datum with uid `{uid}` has already been added.")
        index = len(self._evaluator.uid_to_index)
        self._evaluator.uid_to_index[uid] = index
        self._evaluator.index_to_uid[index] = uid
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
            self._evaluator.index_to_label[label_id] = label
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

            # update metadata
            self._evaluator.n_datums += 1
            self._evaluator.n_groundtruths += len(segmentation.groundtruths)
            self._evaluator.n_predictions += len(segmentation.predictions)
            self._evaluator.n_pixels += segmentation.size
            self._evaluator.n_groundtruth_pixels += segmentation.size * len(
                segmentation.groundtruths
            )
            self._evaluator.n_prediction_pixels += segmentation.size * len(
                segmentation.predictions
            )

            # update datum cache
            uid_index = self._add_datum(segmentation.uid)

            groundtruth_labels = np.full(
                len(segmentation.groundtruths), fill_value=-1
            )
            for idx, groundtruth in enumerate(segmentation.groundtruths):
                label_idx = self._add_label(groundtruth.label)
                groundtruth_labels[idx] = label_idx
                self.groundtruth_count[label_idx][
                    uid_index
                ] += groundtruth.mask.sum()

            prediction_labels = np.full(
                len(segmentation.predictions), fill_value=-1
            )
            for idx, prediction in enumerate(segmentation.predictions):
                label_idx = self._add_label(prediction.label)
                prediction_labels[idx] = label_idx
                self.prediction_count[label_idx][
                    uid_index
                ] += prediction.mask.sum()

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
            self.pixel_count.append(segmentation.size)

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

        n_datums = self._evaluator.n_datums
        n_labels = len(self._evaluator.index_to_label)

        self._evaluator.n_labels = n_labels

        self._evaluator._label_metadata_per_datum = np.zeros(
            (2, n_datums, n_labels), dtype=np.int32
        )
        for datum_idx in range(n_datums):
            for label_idx in range(n_labels):
                gt_count = (
                    self.groundtruth_count[label_idx].get(datum_idx, 0)
                    if label_idx in self.groundtruth_count
                    else 0
                )
                pd_count = (
                    self.prediction_count[label_idx].get(datum_idx, 0)
                    if label_idx in self.prediction_count
                    else 0
                )
                self._evaluator._label_metadata_per_datum[
                    :, datum_idx, label_idx
                ] = np.array([gt_count, pd_count])

        self._evaluator._label_metadata = np.array(
            [
                [
                    np.sum(
                        self._evaluator._label_metadata_per_datum[
                            0, :, label_idx
                        ]
                    ),
                    np.sum(
                        self._evaluator._label_metadata_per_datum[
                            1, :, label_idx
                        ]
                    ),
                ]
                for label_idx in range(n_labels)
            ],
            dtype=np.int32,
        )

        self._evaluator._n_pixels_per_datum = np.array(
            self.pixel_count, dtype=np.int32
        )

        self._evaluator._confusion_matrices = np.zeros(
            (n_datums, n_labels + 1, n_labels + 1), dtype=np.int32
        )
        for idx, matrix in enumerate(self.matrices):
            h, w = matrix.shape
            self._evaluator._confusion_matrices[idx, :h, :w] = matrix

        return self._evaluator
