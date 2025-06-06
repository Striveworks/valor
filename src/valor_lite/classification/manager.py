from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from valor_lite.classification.annotation import Classification
from valor_lite.classification.computation import (
    compute_confusion_matrix,
    compute_label_metadata,
    compute_precision_recall_rocauc,
    filter_cache,
)
from valor_lite.classification.metric import Metric, MetricType
from valor_lite.classification.utilities import (
    unpack_confusion_matrix_into_metric_list,
    unpack_precision_recall_rocauc_into_metric_lists,
)
from valor_lite.exceptions import EmptyEvaluatorError, EmptyFilterError

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

    @classmethod
    def create(
        cls,
        detailed_pairs: NDArray[np.float64],
        number_of_datums: int,
        number_of_labels: int,
    ):
        # count number of unique ground truths
        mask_valid_gts = detailed_pairs[:, 1] >= 0
        unique_ids = np.unique(
            detailed_pairs[np.ix_(mask_valid_gts, (0, 1))],  # type: ignore - np.ix_ typing
            axis=0,
        )
        number_of_ground_truths = int(unique_ids.shape[0])

        # count number of unqiue predictions
        mask_valid_pds = detailed_pairs[:, 2] >= 0
        unique_ids = np.unique(
            detailed_pairs[np.ix_(mask_valid_pds, (0, 2))], axis=0  # type: ignore - np.ix_ typing
        )
        number_of_predictions = int(unique_ids.shape[0])

        return cls(
            number_of_datums=number_of_datums,
            number_of_ground_truths=number_of_ground_truths,
            number_of_predictions=number_of_predictions,
            number_of_labels=number_of_labels,
        )

    def to_dict(self) -> dict[str, int | bool]:
        return asdict(self)


@dataclass
class Filter:
    datum_mask: NDArray[np.bool_]
    valid_label_indices: NDArray[np.int32] | None
    metadata: Metadata

    def __post_init__(self):
        # validate datum mask
        if not self.datum_mask.any():
            raise EmptyFilterError("filter removes all datums")

        # validate label indices
        if (
            self.valid_label_indices is not None
            and self.valid_label_indices.size == 0
        ):
            raise EmptyFilterError("filter removes all labels")


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

        # internal caches
        self._detailed_pairs = np.array([])
        self._label_metadata = np.array([], dtype=np.int32)
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
        datums: list[str] | NDArray[np.int32] | None = None,
        labels: list[str] | NDArray[np.int32] | None = None,
    ) -> Filter:
        """
        Creates a filter object.

        Parameters
        ----------
        datums : list[str] | NDArray[int32], optional
            An optional list of string uids or integer indices representing datums.
        labels : list[str] | NDArray[int32], optional
            An optional list of strings or integer indices representing labels.

        Returns
        -------
        Filter
            The filter object representing the input parameters.
        """
        # create datum mask
        n_pairs = self._detailed_pairs.shape[0]
        datum_mask = np.ones(n_pairs, dtype=np.bool_)
        if datums is not None:
            # convert to array of valid datum indices
            if isinstance(datums, list):
                datums = np.array(
                    [self.datum_id_to_index[uid] for uid in datums],
                    dtype=np.int32,
                )

            # return early if all data removed
            if datums.size == 0:
                raise EmptyFilterError("filter removes all datums")

            # validate indices
            if datums.max() >= len(self.index_to_datum_id):
                raise ValueError(
                    f"datum index '{datums.max()}' exceeds total number of datums"
                )
            elif datums.min() < 0:
                raise ValueError(
                    f"datum index '{datums.min()}' is a negative value"
                )

            # create datum mask
            datum_mask = np.isin(self._detailed_pairs[:, 0], datums)

        # collect valid label indices
        if labels is not None:
            # convert to array of valid label indices
            if isinstance(labels, list):
                labels = np.array(
                    [self.label_to_index[label] for label in labels]
                )

            # return early if all data removed
            if labels.size == 0:
                raise EmptyFilterError("filter removes all labels")

            # validate indices
            if labels.max() >= len(self.index_to_label):
                raise ValueError(
                    f"label index '{labels.max()}' exceeds total number of labels"
                )
            elif labels.min() < 0:
                raise ValueError(
                    f"label index '{labels.min()}' is a negative value"
                )

            # add -1 to represent null labels which should not be filtered
            labels = np.concatenate([labels, np.array([-1])])

        filtered_detailed_pairs, _ = filter_cache(
            detailed_pairs=self._detailed_pairs,
            datum_mask=datum_mask,
            valid_label_indices=labels,
            n_labels=self.metadata.number_of_labels,
        )

        number_of_datums = (
            datums.size
            if datums is not None
            else self.metadata.number_of_datums
        )

        return Filter(
            datum_mask=datum_mask,
            valid_label_indices=labels,
            metadata=Metadata.create(
                detailed_pairs=filtered_detailed_pairs,
                number_of_datums=number_of_datums,
                number_of_labels=self.metadata.number_of_labels,
            ),
        )

    def filter(
        self, filter_: Filter
    ) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
        """
        Performs filtering over the internal cache.

        Parameters
        ----------
        filter_ : Filter
            The filter object representation.

        Returns
        -------
        NDArray[float64]
            The filtered detailed pairs.
        NDArray[int32]
            The filtered label metadata.
        """
        return filter_cache(
            detailed_pairs=self._detailed_pairs,
            datum_mask=filter_.datum_mask,
            valid_label_indices=filter_.valid_label_indices,
            n_labels=self.metadata.number_of_labels,
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
            Applies a filter to the internal cache.

        Returns
        -------
        dict[MetricType, list]
            A dictionary mapping MetricType enumerations to lists of computed metrics.
        """
        # apply filters
        if filter_ is not None:
            detailed_pairs, label_metadata = self.filter(filter_=filter_)
            n_datums = filter_.metadata.number_of_datums
        else:
            detailed_pairs = self._detailed_pairs
            label_metadata = self._label_metadata
            n_datums = self.metadata.number_of_datums

        results = compute_precision_recall_rocauc(
            detailed_pairs=detailed_pairs,
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
        # apply filters
        if filter_ is not None:
            detailed_pairs, _ = self.filter(filter_=filter_)
        else:
            detailed_pairs = self._detailed_pairs

        if detailed_pairs.size == 0:
            return list()

        result = compute_confusion_matrix(
            detailed_pairs=detailed_pairs,
            score_thresholds=np.array(score_thresholds),
            hardmax=hardmax,
        )
        return unpack_confusion_matrix_into_metric_list(
            detailed_pairs=detailed_pairs,
            result=result,
            score_thresholds=score_thresholds,
            index_to_datum_id=self.index_to_datum_id,
            index_to_label=self.index_to_label,
        )

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

        Raises
        ------
        ValueError
            If datum id already exists.
        """
        if uid in self.datum_id_to_index:
            raise ValueError("datum with id '{uid}' already exists")
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
            raise EmptyEvaluatorError()

        self._label_metadata = compute_label_metadata(
            ids=self._detailed_pairs[:, :3].astype(np.int32),
            n_labels=len(self.index_to_label),
        )
        indices = np.lexsort(
            (
                self._detailed_pairs[:, 1],  # ground truth
                self._detailed_pairs[:, 2],  # prediction
                -self._detailed_pairs[:, 3],  # score
            )
        )
        self._detailed_pairs = self._detailed_pairs[indices]
        self._metadata = Metadata.create(
            detailed_pairs=self._detailed_pairs,
            number_of_datums=len(self.index_to_datum_id),
            number_of_labels=len(self.index_to_label),
        )
        return self


class DataLoader(Evaluator):
    """
    Used for backwards compatibility as the Evaluator now handles ingestion.
    """

    pass
