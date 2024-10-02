from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from valor_lite.segmentation.annotation import Segmentation
from valor_lite.segmentation.computation import compute_metrics
from valor_lite.segmentation.metric import (
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
    n_datums: int


class Evaluator:
    """
    Segmentation Evaluator
    """

    def __init__(self):

        # metadata
        self.n_datums = 0
        self.n_groundtruths = 0
        self.n_predictions = 0
        self.n_labels = 0

        # datum reference
        self.uid_to_index: dict[str, int] = dict()
        self.index_to_uid: dict[int, str] = dict()

        # label reference
        self.label_to_index: dict[str, int] = dict()
        self.index_to_label: dict[int, str] = dict()

        # computation caches
        self._classifications = np.array([])
        self._label_metadata = np.array([], dtype=np.int32)
        self._label_metadata_per_datum = np.array([], dtype=np.int32)

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
            "n_datums": self.n_datums,
            "n_groundtruths": self.n_groundtruths,
            "n_predictions": self.n_predictions,
            "n_labels": self.n_labels,
            "ignored_prediction_labels": self.ignored_prediction_labels,
            "missing_prediction_labels": self.missing_prediction_labels,
        }

    def create_filter(
        self,
        datum_uids: list[str] | NDArray[np.int32] | None = None,
    ) -> Filter:
        """
        Creates a boolean mask that can be passed to an evaluation.

        Parameters
        ----------
        datum_uids : list[str] | NDArray[np.int32], optional
            An optional list of string uids or a numpy array of uid indices.
        labels : list[tuple[str, str]] | NDArray[np.int32], optional
            An optional list of labels or a numpy array of label indices.
        label_keys : list[str] | NDArray[np.int32], optional
            An optional list of label keys or a numpy array of label key indices.

        Returns
        -------
        Filter
            A filter object that can be passed to the `evaluate` method.
        """
        n_rows = self._classifications.shape[0]

        n_datums = self._label_metadata_per_datum.shape[1]
        n_labels = self._label_metadata_per_datum.shape[2]

        mask_pairs = np.ones((n_rows, 1), dtype=np.bool_)
        mask_datums = np.ones(n_datums, dtype=np.bool_)
        mask_labels = np.ones(n_labels, dtype=np.bool_)

        if datum_uids is not None:
            if isinstance(datum_uids, list):
                datum_uids = np.array(
                    [self.uid_to_index[uid] for uid in datum_uids],
                    dtype=np.int32,
                )
            mask = np.zeros_like(mask_pairs, dtype=np.bool_)
            mask[datum_uids] = True
            mask_pairs &= mask

            mask = np.zeros_like(mask_datums, dtype=np.bool_)
            mask[datum_uids] = True
            mask_datums &= mask

        mask = mask_datums[:, np.newaxis] & mask_labels[np.newaxis, :]
        label_metadata_per_datum = self._label_metadata_per_datum.copy()
        label_metadata_per_datum[:, ~mask] = 0

        label_metadata = np.zeros_like(self._label_metadata, dtype=np.int32)
        label_metadata[:, :2] = np.transpose(
            np.sum(
                label_metadata_per_datum,
                axis=1,
            )
        )
        label_metadata[:, 2] = self._label_metadata[:, 2]
        n_datums = int(np.sum(label_metadata[:, 0]))

        return Filter(
            indices=np.where(mask_pairs)[0],
            label_metadata=label_metadata,
            n_datums=n_datums,
        )

    def evaluate(
        self,
        metrics_to_return: list[MetricType] = MetricType.base(),
        score_thresholds: list[float] = [0.0],
        number_of_examples: int = 0,
        filter_: Filter | None = None,
    ) -> dict[MetricType, list]:
        """
        Performs an evaluation and returns metrics.

        Parameters
        ----------
        metrics_to_return : list[MetricType]
            A list of metrics to return in the results.
        score_thresholds : list[float]
            A list of score thresholds to compute metrics over.
        number_of_examples : int, default=0
            Maximum number of annotation examples to return in ConfusionMatrix.
        filter_ : Filter, optional
            An optional filter object.

        Returns
        -------
        dict[MetricType, list]
            A dictionary mapping MetricType enumerations to lists of computed metrics.
        """

        # apply filters
        data = self._classifications
        label_metadata = self._label_metadata
        if filter_ is not None:
            data = data[filter_.indices]
            label_metadata = filter_.label_metadata

        (
            precision,
            recall,
            f1_score,
            accuracy,
            ious,
            missing_predictions,
        ) = compute_metrics(
            data=data,
            label_metadata=label_metadata,
            score_thresholds=np.array(score_thresholds),
        )

        metrics = defaultdict(list)

        metrics[MetricType.Accuracy].append(
            Accuracy(
                value=accuracy.tolist(),
                score_thresholds=score_thresholds,
            )
        )

        for label_idx, label in self.index_to_label.items():

            kwargs = {
                "label": label,
                "score_thresholds": score_thresholds,
            }

            # if no groundtruths exists for a label, skip it.
            if label_metadata[label_idx, 0] == 0:
                continue

            metrics[MetricType.Precision].append(
                Precision(
                    value=precision[:, label_idx].tolist(),
                    **kwargs,
                )
            )
            metrics[MetricType.Recall].append(
                Recall(
                    value=recall[:, label_idx].tolist(),
                    **kwargs,
                )
            )
            metrics[MetricType.F1].append(
                F1(
                    value=f1_score[:, label_idx].tolist(),
                    **kwargs,
                )
            )

        for metric in set(metrics.keys()):
            if metric not in metrics_to_return:
                del metrics[metric]

        return metrics


class DataLoader:
    """
    Segmentation DataLoader.
    """

    def __init__(self):
        self._evaluator = Evaluator()
        self.groundtruth_count = defaultdict(lambda: defaultdict(int))
        self.prediction_count = defaultdict(lambda: defaultdict(int))

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
        label_id = len(self._evaluator.index_to_label)
        if label not in self._evaluator.label_to_index:
            self._evaluator.label_to_index[label] = label_id
            self._evaluator.index_to_label[label_id] = label
            label_id += 1
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

            uid_index = self._add_datum(segmentation.uid)

            combined_groundtruths = np.stack(
                [
                    groundtruth.mask
                    for groundtruth in segmentation.groundtruths
                ],
                axis=0,
            )
            combined_predictions = np.stack(
                [prediction.mask for prediction in segmentation.predictions],
                axis=0,
            )

            classification = (
                np.ones_like((1, 3, segmentation.size), dtype=np.floating) * -1
            )
            groundtruth_indices = np.argmax(combined_groundtruths, axis=0)
            prediction_indices = np.argmax(combined_predictions, axis=0)

            for idx, groundtruth in enumerate(segmentation.groundtruths):
                label_idx = self._add_label(groundtruth.label)
                mask_pixels = groundtruth_indices == idx
                self.groundtruth_count[label_idx][
                    uid_index
                ] += mask_pixels.sum()
                classification[0, 0, :][mask_pixels.flatten()] = label_idx
            for idx, prediction in enumerate(segmentation.predictions):
                label_idx = self._add_label(prediction.label)
                mask_pixels = prediction_indices == idx
                self.prediction_count[label_idx][
                    uid_index
                ] += mask_pixels.sum()
                classification[0, 1, :][mask_pixels.flatten()] = label_idx
            classification[0, 2, :] = combined_predictions[prediction_indices]

            self._evaluator._classifications = np.concatenate(
                [
                    self._evaluator._classifications,
                    classification,
                ],
                axis=0,
            )

    def finalize(self) -> Evaluator:
        """
        Performs data finalization and some preprocessing steps.

        Returns
        -------
        Evaluator
            A ready-to-use evaluator object.
        """

        if self._evaluator._classifications.size == 0:
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

        return self._evaluator
