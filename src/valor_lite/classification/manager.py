import warnings

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from valor_lite.classification.annotation import Classification
from valor_lite.classification.computation import (
    compute_confusion_matrix,
    compute_label_metadata,
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


class Evaluator:
    """
    Classification Evaluator
    """

    def __init__(self):
        # external references
        self.datum_id_to_index: dict[str, int] = {}
        self.label_to_index: dict[str, int] = {}

        self.index_to_datum_id: list[str] = []
        self.index_to_label: list[str] = []

        # computation caches
        self._detailed_pairs = np.array([])
        self._label_metadata = np.array([], dtype=np.int32)

        # filtered internal cache
        self._filtered_detailed_pairs: NDArray[np.float64] | None = None
        self._filtered_label_metadata: NDArray[np.int32] | None = None

    @property
    def is_filtered(self) -> bool:
        return self._filtered_detailed_pairs is not None

    @property
    def label_metadata(self) -> NDArray[np.int32]:
        return (
            self._filtered_label_metadata
            if self._filtered_label_metadata is not None
            else self._label_metadata
        )

    @property
    def detailed_pairs(self) -> NDArray[np.float64]:
        return (
            self._filtered_detailed_pairs
            if self._filtered_detailed_pairs is not None
            else self._detailed_pairs
        )

    @property
    def n_labels(self) -> int:
        """Returns the total number of unique labels."""
        return len(self.index_to_label)

    @property
    def n_datums(self) -> int:
        """Returns the number of datums."""
        return np.unique(self.detailed_pairs[:, 0]).size

    @property
    def n_groundtruths(self) -> int:
        """Returns the number of ground truth annotations."""
        mask_valid_gts = self.detailed_pairs[:, 1] >= 0
        unique_ids = np.unique(
            self.detailed_pairs[np.ix_(mask_valid_gts, (0, 1))], axis=0  # type: ignore - np.ix_ typing
        )
        return int(unique_ids.shape[0])

    @property
    def n_predictions(self) -> int:
        """Returns the number of prediction annotations."""
        mask_valid_pds = self.detailed_pairs[:, 2] >= 0
        unique_ids = np.unique(
            self.detailed_pairs[np.ix_(mask_valid_pds, (0, 2))], axis=0  # type: ignore - np.ix_ typing
        )
        return int(unique_ids.shape[0])

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
            "n_datums": self.n_datums,
            "n_groundtruths": self.n_groundtruths,
            "n_predictions": self.n_predictions,
            "n_labels": self.n_labels,
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
        datum_uids : list[str], optional
            An optional list of string uids representing datums.
        labels : list[str], optional
            An optional list of labels.
        """
        self._filtered_detailed_pairs = self._detailed_pairs.copy()
        self._filtered_label_metadata = np.zeros(
            (self.n_labels, 2), dtype=np.int32
        )

        valid_datum_indices = None
        if datum_ids is not None:
            if not datum_ids:
                self._filtered_detailed_pairs = np.array([], dtype=np.float64)
                warnings.warn("no valid filtered pairs")
                return
            valid_datum_indices = np.array(
                [self.datum_id_to_index[uid] for uid in datum_ids],
                dtype=np.int32,
            )

        valid_label_indices = None
        if labels is not None:
            if not labels:
                self._filtered_detailed_pairs = np.array([], dtype=np.float64)
                warnings.warn("no valid filtered pairs")
                return
            valid_label_indices = np.array(
                [self.label_to_index[label] for label in labels] + [-1]
            )

        # filter datums
        if valid_datum_indices is not None:
            mask_valid_datums = np.isin(
                self._filtered_detailed_pairs[:, 0], valid_datum_indices
            )
            self._filtered_detailed_pairs = self._filtered_detailed_pairs[
                mask_valid_datums
            ]

        n_rows = self._filtered_detailed_pairs.shape[0]
        mask_invalid_groundtruths = np.zeros(n_rows, dtype=np.bool_)
        mask_invalid_predictions = np.zeros_like(mask_invalid_groundtruths)

        # filter labels
        if valid_label_indices is not None:
            mask_invalid_groundtruths[
                ~np.isin(
                    self._filtered_detailed_pairs[:, 1], valid_label_indices
                )
            ] = True
            mask_invalid_predictions[
                ~np.isin(
                    self._filtered_detailed_pairs[:, 2], valid_label_indices
                )
            ] = True

        # filter cache
        if mask_invalid_groundtruths.any():
            invalid_groundtruth_indices = np.where(mask_invalid_groundtruths)[
                0
            ]
            self._filtered_detailed_pairs[
                invalid_groundtruth_indices[:, None], 1
            ] = np.array([[-1.0]])

        if mask_invalid_predictions.any():
            invalid_prediction_indices = np.where(mask_invalid_predictions)[0]
            self._filtered_detailed_pairs[
                invalid_prediction_indices[:, None], (2, 3, 4)
            ] = np.array([[-1.0, -1.0, -1.0]])

        # filter null pairs
        mask_null_pairs = np.all(
            np.isclose(
                self._filtered_detailed_pairs[:, 1:5],
                np.array([-1.0, -1.0, -1.0, -1.0]),
            ),
            axis=1,
        )
        self._filtered_detailed_pairs = self._filtered_detailed_pairs[
            ~mask_null_pairs
        ]

        if self._filtered_detailed_pairs.size == 0:
            warnings.warn("no valid filtered pairs")
            return

        self._filtered_detailed_pairs = np.unique(
            self._filtered_detailed_pairs, axis=0
        )
        indices = np.lexsort(
            (
                self._filtered_detailed_pairs[:, 1],  # ground truth
                self._filtered_detailed_pairs[:, 2],  # prediction
                -self._filtered_detailed_pairs[:, 3],  # score
            )
        )
        self._filtered_detailed_pairs = self._filtered_detailed_pairs[indices]
        self._filtered_label_metadata = compute_label_metadata(
            ids=self._filtered_detailed_pairs[:, :3].astype(np.int32),
            n_labels=self.n_labels,
        )

    def compute_precision_recall_rocauc(
        self,
        score_thresholds: list[float] = [0.0],
        hardmax: bool = True,
    ) -> dict[MetricType, list]:
        """
        Performs an evaluation and returns metrics.

        Parameters
        ----------
        score_thresholds : list[float]
            A list of score thresholds to compute metrics over.
        hardmax : bool
            Toggles whether a hardmax is applied to predictions.

        Returns
        -------
        dict[MetricType, list]
            A dictionary mapping MetricType enumerations to lists of computed metrics.
        """

        # apply filters
        data = self.detailed_pairs
        label_metadata = self.label_metadata
        n_datums = self.n_datums

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

        Returns
        -------
        list[Metric]
            A list of confusion matrices.
        """

        # apply filters
        data = self.detailed_pairs
        label_metadata = self.label_metadata

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
            index_to_datum_id=self.index_to_datum_id,
            index_to_label=self.index_to_label,
        )

    def evaluate(
        self,
        score_thresholds: list[float] = [0.0],
        hardmax: bool = True,
        number_of_examples: int = 0,
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

        Returns
        -------
        dict[MetricType, list[Metric]]
            Lists of metrics organized by metric type.
        """

        metrics = self.compute_precision_recall_rocauc(
            score_thresholds=score_thresholds,
            hardmax=hardmax,
        )

        metrics[MetricType.ConfusionMatrix] = self.compute_confusion_matrix(
            score_thresholds=score_thresholds,
            hardmax=hardmax,
            number_of_examples=number_of_examples,
        )

        return metrics

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
        if uid not in self.datum_id_to_index:
            index = len(self.datum_id_to_index)
            self.datum_id_to_index[uid] = index
            self.index_to_datum_id.append(uid)
        return self.datum_id_to_index[uid]

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
        label_id = len(self.index_to_label)
        if label not in self.label_to_index:
            self.label_to_index[label] = label_id
            self.index_to_label.append(label)
            label_id += 1
        return self.label_to_index[label]

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

            # update datum uid index
            uid_index = self._add_datum(uid=classification.uid)

            # cache labels and annotations
            groundtruth = self._add_label(classification.groundtruth)

            predictions = list()
            for plabel, pscore in zip(
                classification.predictions, classification.scores
            ):
                label_idx = self._add_label(plabel)
                predictions.append(
                    (
                        label_idx,
                        pscore,
                    )
                )

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

            if self._detailed_pairs.size == 0:
                self._detailed_pairs = np.array(pairs)
            else:
                self._detailed_pairs = np.concatenate(
                    [
                        self._detailed_pairs,
                        np.array(pairs),
                    ],
                    axis=0,
                )

    def finalize(self):
        """
        Performs data finalization and some preprocessing steps.

        Returns
        -------
        Evaluator
            A ready-to-use evaluator object.
        """
        if self._detailed_pairs.size == 0:
            self._label_metadata = np.array([], dtype=np.int32)
            warnings.warn("evaluator is empty")
            return self

        self._label_metadata = compute_label_metadata(
            ids=self._detailed_pairs[:, :3].astype(np.int32),
            n_labels=self.n_labels,
        )
        indices = np.lexsort(
            (
                self._detailed_pairs[:, 1],  # ground truth
                self._detailed_pairs[:, 2],  # prediction
                -self._detailed_pairs[:, 3],  # score
            )
        )
        self._detailed_pairs = self._detailed_pairs[indices]
        return self


class DataLoader(Evaluator):
    """
    Used for backwards compatibility as the Evaluator now handles ingestion.
    """

    pass
