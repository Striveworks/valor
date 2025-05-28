from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from valor_lite.classification.annotation import Classification
from valor_lite.classification.computation import (
    compute_confusion_matrix,
    compute_precision_recall_rocauc,
)
from valor_lite.classification.metric import Metric, MetricType
from valor_lite.classification.utilities import (
    unpack_confusion_matrix_into_metric_list,
    unpack_precision_recall_rocauc_into_metric_lists,
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
    indices: NDArray[np.intp]
    label_metadata: NDArray[np.int32]
    n_datums: int


class Evaluator:
    """
    Classification Evaluator
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
        self._detailed_pairs = np.array([])
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
        labels: list[str] | NDArray[np.int32] | None = None,
    ) -> Filter:
        """
        Creates a boolean mask that can be passed to an evaluation.

        Parameters
        ----------
        datum_uids : list[str] | NDArray[np.int32], optional
            An optional list of string uids or a numpy array of uid indices.
        labels : list[str] | NDArray[np.int32], optional
            An optional list of labels or a numpy array of label indices.

        Returns
        -------
        Filter
            A filter object that can be passed to the `evaluate` method.
        """
        n_rows = self._detailed_pairs.shape[0]

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
            mask[
                np.isin(self._detailed_pairs[:, 0].astype(int), datum_uids)
            ] = True
            mask_pairs &= mask

            mask = np.zeros_like(mask_datums, dtype=np.bool_)
            mask[datum_uids] = True
            mask_datums &= mask

        if labels is not None:
            if isinstance(labels, list):
                labels = np.array(
                    [self.label_to_index[label] for label in labels]
                )
            mask = np.zeros_like(mask_pairs, dtype=np.bool_)
            mask[
                np.isin(self._detailed_pairs[:, 1].astype(int), labels)
            ] = True
            mask_pairs &= mask

            mask = np.zeros_like(mask_labels, dtype=np.bool_)
            mask[labels] = True
            mask_labels &= mask

        mask = mask_datums[:, np.newaxis] & mask_labels[np.newaxis, :]
        label_metadata_per_datum = self._label_metadata_per_datum.copy()
        label_metadata_per_datum[:, ~mask] = 0

        label_metadata: NDArray[np.int32] = np.transpose(
            np.sum(
                label_metadata_per_datum,
                axis=1,
            )
        )

        n_datums = int(np.sum(label_metadata[:, 0]))

        return Filter(
            indices=np.where(mask_pairs)[0],
            label_metadata=label_metadata,
            n_datums=n_datums,
        )

    def compute_precision_recall_rocauc(
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
            An optional filter object.

        Returns
        -------
        dict[MetricType, list]
            A dictionary mapping MetricType enumerations to lists of computed metrics.
        """

        # apply filters
        data = self._detailed_pairs
        label_metadata = self._label_metadata
        n_datums = self.n_datums
        if filter_ is not None:
            data = data[filter_.indices]
            label_metadata = filter_.label_metadata
            n_datums = filter_.n_datums

        results = compute_precision_recall_rocauc(
            data=data,
            label_metadata=label_metadata,
            score_thresholds=np.array(score_thresholds),
            hardmax=hardmax,
            n_datums=n_datums,
        )

        return unpack_precision_recall_rocauc_into_metric_lists(
            results=results,
            score_thresholds=score_thresholds,
            hardmax=hardmax,
            label_metadata=label_metadata,
            index_to_label=self.index_to_label,
        )

    def compute_confusion_matrix(
        self,
        score_thresholds: list[float] = [0.0],
        hardmax: bool = True,
        number_of_examples: int = 0,
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
        number_of_examples : int, default=0
            The number of examples to return per count.
        filter_ : Filter, optional
            An optional filter object.

        Returns
        -------
        list[Metric]
            A list of confusion matrices.
        """

        # apply filters
        data = self._detailed_pairs
        label_metadata = self._label_metadata
        if filter_ is not None:
            data = data[filter_.indices]
            label_metadata = filter_.label_metadata

        if data.size == 0:
            return list()

        results = compute_confusion_matrix(
            data=data,
            label_metadata=label_metadata,
            score_thresholds=np.array(score_thresholds),
            hardmax=hardmax,
            n_examples=number_of_examples,
        )

        return unpack_confusion_matrix_into_metric_list(
            results=results,
            score_thresholds=score_thresholds,
            number_of_examples=number_of_examples,
            index_to_uid=self.index_to_uid,
            index_to_label=self.index_to_label,
        )

    def evaluate(
        self,
        score_thresholds: list[float] = [0.0],
        hardmax: bool = True,
        number_of_examples: int = 0,
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
        number_of_examples : int, default=0
            The number of examples to return per count.
        filter_ : Filter, optional
            An optional filter object.

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
            number_of_examples=number_of_examples,
            filter_=filter_,
        )

        return metrics


class DataLoader:
    """
    Classification DataLoader.
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
        if uid not in self._evaluator.uid_to_index:
            index = len(self._evaluator.uid_to_index)
            self._evaluator.uid_to_index[uid] = index
            self._evaluator.index_to_uid[index] = uid
        return self._evaluator.uid_to_index[uid]

    def _add_label(self, label: str) -> int:
        """
        Helper function for adding a label to the cache.

        Parameters
        ----------
        label : str
            A string representing a label.

        Returns
        -------
        int
            Label index.
        """
        label_id = len(self._evaluator.index_to_label)
        if label not in self._evaluator.label_to_index:
            self._evaluator.label_to_index[label] = label_id
            self._evaluator.index_to_label[label_id] = label

            label_id += 1

        return self._evaluator.label_to_index[label]

    def _add_data(
        self,
        uid_index: int,
        groundtruth: int,
        predictions: list[tuple[int, float]],
    ):

        pairs = list()
        scores = np.array([score for _, score in predictions])
        max_score_idx = np.argmax(scores)

        for idx, (plabel, score) in enumerate(predictions):
            pairs.append(
                (
                    float(uid_index),
                    float(groundtruth),
                    float(plabel),
                    float(score),
                    float(max_score_idx == idx),
                )
            )

        if self._evaluator._detailed_pairs.size == 0:
            self._evaluator._detailed_pairs = np.array(pairs)
        else:
            self._evaluator._detailed_pairs = np.concatenate(
                [
                    self._evaluator._detailed_pairs,
                    np.array(pairs),
                ],
                axis=0,
            )

    def add_data(
        self,
        classifications: list[Classification],
        show_progress: bool = False,
    ):
        """
        Adds classifications to the cache.

        Parameters
        ----------
        classifications : list[Classification]
            A list of Classification objects.
        show_progress : bool, default=False
            Toggle for tqdm progress bar.
        """

        disable_tqdm = not show_progress
        for classification in tqdm(classifications, disable=disable_tqdm):

            if len(classification.predictions) == 0:
                raise ValueError(
                    "Classifications must contain at least one prediction."
                )
            # update metadata
            self._evaluator.n_datums += 1
            self._evaluator.n_groundtruths += 1
            self._evaluator.n_predictions += len(classification.predictions)

            # update datum uid index
            uid_index = self._add_datum(uid=classification.uid)

            # cache labels and annotations
            groundtruth = self._add_label(classification.groundtruth)
            self.groundtruth_count[groundtruth][uid_index] += 1

            predictions = list()
            for plabel, pscore in zip(
                classification.predictions, classification.scores
            ):
                label_idx = self._add_label(plabel)
                self.prediction_count[label_idx][uid_index] += 1
                predictions.append(
                    (
                        label_idx,
                        pscore,
                    )
                )

            self._add_data(
                uid_index=uid_index,
                groundtruth=groundtruth,
                predictions=predictions,
            )

    def finalize(self) -> Evaluator:
        """
        Performs data finalization and some preprocessing steps.

        Returns
        -------
        Evaluator
            A ready-to-use evaluator object.
        """

        if self._evaluator._detailed_pairs.size == 0:
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

        # sort pairs by groundtruth, prediction, score
        indices = np.lexsort(
            (
                self._evaluator._detailed_pairs[:, 1],
                self._evaluator._detailed_pairs[:, 2],
                -self._evaluator._detailed_pairs[:, 3],
            )
        )
        self._evaluator._detailed_pairs = self._evaluator._detailed_pairs[
            indices
        ]

        return self._evaluator
