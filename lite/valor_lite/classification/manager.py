from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from valor_lite.classification.annotation import Classification
from valor_lite.classification.computation import (
    compute_confusion_matrix,
    compute_metrics,
)
from valor_lite.classification.metric import (
    F1,
    ROCAUC,
    Accuracy,
    ConfusionMatrix,
    Counts,
    MetricType,
    Precision,
    Recall,
    mROCAUC,
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
        self.label_to_index: dict[tuple[str, str], int] = dict()
        self.index_to_label: dict[int, tuple[str, str]] = dict()

        # label key reference
        self.index_to_label_key: dict[int, str] = dict()
        self.label_key_to_index: dict[str, int] = dict()
        self.label_index_to_label_key_index: dict[int, int] = dict()

        # computation caches
        self._detailed_pairs = np.array([])
        self._label_metadata = np.array([], dtype=np.int32)
        self._label_metadata_per_datum = np.array([], dtype=np.int32)

    @property
    def ignored_prediction_labels(self) -> list[tuple[str, str]]:
        """
        Prediction labels that are not present in the ground truth set.
        """
        glabels = set(np.where(self._label_metadata[:, 0] > 0)[0])
        plabels = set(np.where(self._label_metadata[:, 1] > 0)[0])
        return [
            self.index_to_label[label_id] for label_id in (plabels - glabels)
        ]

    @property
    def missing_prediction_labels(self) -> list[tuple[str, str]]:
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
        labels: list[tuple[str, str]] | NDArray[np.int32] | None = None,
        label_keys: list[str] | NDArray[np.int32] | None = None,
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

        if label_keys is not None:
            if isinstance(label_keys, list):
                label_keys = np.array(
                    [self.label_key_to_index[key] for key in label_keys]
                )
            label_indices = np.where(
                np.isclose(self._label_metadata[:, 2], label_keys)
            )[0]
            mask = np.zeros_like(mask_pairs, dtype=np.bool_)
            mask[
                np.isin(self._detailed_pairs[:, 1].astype(int), label_indices)
            ] = True
            mask_pairs &= mask

            mask = np.zeros_like(mask_labels, dtype=np.bool_)
            mask[label_indices] = True
            mask_labels &= mask

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
        hardmax: bool = True,
        number_of_examples: int = 0,
        filter_: Filter | None = None,
        as_dict: bool = False,
    ) -> dict[MetricType, list]:
        """
        Performs an evaluation and returns metrics.

        Parameters
        ----------
        metrics_to_return : list[MetricType]
            A list of metrics to return in the results.
        score_thresholds : list[float]
            A list of score thresholds to compute metrics over.
        hardmax : bool
            Toggles whether a hardmax is applied to predictions.
        number_of_examples : int, default=0
            Maximum number of annotation examples to return in ConfusionMatrix.
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
        data = self._detailed_pairs
        label_metadata = self._label_metadata
        n_datums = self.n_datums
        if filter_ is not None:
            data = data[filter_.indices]
            label_metadata = filter_.label_metadata
            n_datums = filter_.n_datums

        (
            counts,
            precision,
            recall,
            accuracy,
            f1_score,
            rocauc,
            mean_rocauc,
        ) = compute_metrics(
            data=data,
            label_metadata=label_metadata,
            score_thresholds=np.array(score_thresholds),
            hardmax=hardmax,
            n_datums=n_datums,
        )

        metrics = defaultdict(list)

        metrics[MetricType.ROCAUC] = [
            ROCAUC(
                value=rocauc[label_idx],
                label=self.index_to_label[label_idx],
            )
            for label_idx in range(label_metadata.shape[0])
            if label_metadata[label_idx, 0] > 0
        ]

        metrics[MetricType.mROCAUC] = [
            mROCAUC(
                value=mean_rocauc[label_key_idx],
                label_key=self.index_to_label_key[label_key_idx],
            )
            for label_key_idx in range(len(self.label_key_to_index))
        ]

        for label_idx, label in self.index_to_label.items():

            kwargs = {
                "label": label,
                "score_thresholds": score_thresholds,
                "hardmax": hardmax,
            }
            row = counts[:, label_idx]
            metrics[MetricType.Counts].append(
                Counts(
                    tp=row[:, 0].tolist(),
                    fp=row[:, 1].tolist(),
                    fn=row[:, 2].tolist(),
                    tn=row[:, 3].tolist(),
                    **kwargs,
                )
            )

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
            metrics[MetricType.Accuracy].append(
                Accuracy(
                    value=accuracy[:, label_idx].tolist(),
                    **kwargs,
                )
            )
            metrics[MetricType.F1].append(
                F1(
                    value=f1_score[:, label_idx].tolist(),
                    **kwargs,
                )
            )

        if MetricType.ConfusionMatrix in metrics_to_return:
            metrics[
                MetricType.ConfusionMatrix
            ] = self._compute_confusion_matrix(
                data=data,
                label_metadata=label_metadata,
                score_thresholds=score_thresholds,
                hardmax=hardmax,
                number_of_examples=number_of_examples,
            )

        for metric in set(metrics.keys()):
            if metric not in metrics_to_return:
                del metrics[metric]

        if as_dict:
            return {
                mtype: [metric.to_dict() for metric in mvalues]
                for mtype, mvalues in metrics.items()
            }

        return metrics

    def _unpack_confusion_matrix(
        self,
        confusion_matrix: NDArray[np.floating],
        label_key_idx: int,
        number_of_labels: int,
        number_of_examples: int,
    ) -> dict[
        str,
        dict[
            str,
            dict[
                str,
                int
                | list[
                    dict[
                        str,
                        str | float,
                    ]
                ],
            ],
        ],
    ]:
        """
        Unpacks a numpy array of confusion matrix counts and examples.
        """

        datum_idx = lambda gt_label_idx, pd_label_idx, example_idx: int(  # noqa: E731 - lambda fn
            confusion_matrix[
                gt_label_idx,
                pd_label_idx,
                example_idx * 2 + 1,
            ]
        )

        score_idx = lambda gt_label_idx, pd_label_idx, example_idx: float(  # noqa: E731 - lambda fn
            confusion_matrix[
                gt_label_idx,
                pd_label_idx,
                example_idx * 2 + 2,
            ]
        )

        return {
            self.index_to_label[gt_label_idx][1]: {
                self.index_to_label[pd_label_idx][1]: {
                    "count": max(
                        int(confusion_matrix[gt_label_idx, pd_label_idx, 0]),
                        0,
                    ),
                    "examples": [
                        {
                            "datum": self.index_to_uid[
                                datum_idx(
                                    gt_label_idx, pd_label_idx, example_idx
                                )
                            ],
                            "score": score_idx(
                                gt_label_idx, pd_label_idx, example_idx
                            ),
                        }
                        for example_idx in range(number_of_examples)
                        if datum_idx(gt_label_idx, pd_label_idx, example_idx)
                        >= 0
                    ],
                }
                for pd_label_idx in range(number_of_labels)
                if (
                    self.label_index_to_label_key_index[pd_label_idx]
                    == label_key_idx
                )
            }
            for gt_label_idx in range(number_of_labels)
            if (
                self.label_index_to_label_key_index[gt_label_idx]
                == label_key_idx
            )
        }

    def _unpack_missing_predictions(
        self,
        missing_predictions: NDArray[np.int32],
        label_key_idx: int,
        number_of_labels: int,
        number_of_examples: int,
    ) -> dict[str, dict[str, int | list[dict[str, str]]]]:
        """
        Unpacks a numpy array of missing prediction counts and examples.
        """

        datum_idx = (
            lambda gt_label_idx, example_idx: int(  # noqa: E731 - lambda fn
                missing_predictions[
                    gt_label_idx,
                    example_idx + 1,
                ]
            )
        )

        return {
            self.index_to_label[gt_label_idx][1]: {
                "count": max(
                    int(missing_predictions[gt_label_idx, 0]),
                    0,
                ),
                "examples": [
                    {
                        "datum": self.index_to_uid[
                            datum_idx(gt_label_idx, example_idx)
                        ]
                    }
                    for example_idx in range(number_of_examples)
                    if datum_idx(gt_label_idx, example_idx) >= 0
                ],
            }
            for gt_label_idx in range(number_of_labels)
            if (
                self.label_index_to_label_key_index[gt_label_idx]
                == label_key_idx
            )
        }

    def _compute_confusion_matrix(
        self,
        data: NDArray[np.floating],
        label_metadata: NDArray[np.int32],
        score_thresholds: list[float],
        hardmax: bool,
        number_of_examples: int,
    ) -> list[ConfusionMatrix]:
        """
        Computes a detailed confusion matrix..

        Parameters
        ----------
        data : NDArray[np.floating]
            A data array containing classification pairs.
        label_metadata : NDArray[np.int32]
            An integer array containing label metadata.
        score_thresholds : list[float]
            A list of score thresholds to compute metrics over.
        hardmax : bool
            Toggles whether a hardmax is applied to predictions.
        number_of_examples : int, default=0
            The number of examples to return per count.

        Returns
        -------
        list[ConfusionMatrix]
            A list of ConfusionMatrix per label key.
        """

        if data.size == 0:
            return list()

        confusion_matrix, missing_predictions = compute_confusion_matrix(
            data=data,
            label_metadata=label_metadata,
            score_thresholds=np.array(score_thresholds),
            hardmax=hardmax,
            n_examples=number_of_examples,
        )

        n_scores, n_labels, _, _ = confusion_matrix.shape
        return [
            ConfusionMatrix(
                score_threshold=score_thresholds[score_idx],
                label_key=label_key,
                number_of_examples=number_of_examples,
                confusion_matrix=self._unpack_confusion_matrix(
                    confusion_matrix=confusion_matrix[score_idx, :, :, :],
                    label_key_idx=label_key_idx,
                    number_of_labels=n_labels,
                    number_of_examples=number_of_examples,
                ),
                missing_predictions=self._unpack_missing_predictions(
                    missing_predictions=missing_predictions[score_idx, :, :],
                    label_key_idx=label_key_idx,
                    number_of_labels=n_labels,
                    number_of_examples=number_of_examples,
                ),
            )
            for label_key_idx, label_key in self.index_to_label_key.items()
            for score_idx in range(n_scores)
        ]


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

    def _add_label(self, label: tuple[str, str]) -> tuple[int, int]:
        """
        Helper function for adding a label to the cache.

        Parameters
        ----------
        label : tuple[str, str]
            The label as a tuple in format (key, value).

        Returns
        -------
        int
            Label index.
        int
            Label key index.
        """
        label_id = len(self._evaluator.index_to_label)
        label_key_id = len(self._evaluator.index_to_label_key)
        if label not in self._evaluator.label_to_index:
            self._evaluator.label_to_index[label] = label_id
            self._evaluator.index_to_label[label_id] = label

            # update label key index
            if label[0] not in self._evaluator.label_key_to_index:
                self._evaluator.label_key_to_index[label[0]] = label_key_id
                self._evaluator.index_to_label_key[label_key_id] = label[0]
                label_key_id += 1

            self._evaluator.label_index_to_label_key_index[
                label_id
            ] = self._evaluator.label_key_to_index[label[0]]
            label_id += 1

        return (
            self._evaluator.label_to_index[label],
            self._evaluator.label_key_to_index[label[0]],
        )

    def _add_data(
        self,
        uid_index: int,
        keyed_groundtruths: dict[int, int],
        keyed_predictions: dict[int, list[tuple[int, float]]],
    ):
        gt_keys = set(keyed_groundtruths.keys())
        pd_keys = set(keyed_predictions.keys())
        joint_keys = gt_keys.intersection(pd_keys)

        gt_unique_keys = gt_keys - pd_keys
        pd_unique_keys = pd_keys - gt_keys
        if gt_unique_keys or pd_unique_keys:
            raise ValueError(
                "Label keys must match between ground truths and predictions."
            )

        pairs = list()
        for key in joint_keys:
            scores = np.array([score for _, score in keyed_predictions[key]])
            max_score_idx = np.argmax(scores)

            glabel = keyed_groundtruths[key]
            for idx, (plabel, score) in enumerate(keyed_predictions[key]):
                pairs.append(
                    (
                        float(uid_index),
                        float(glabel),
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

            # update metadata
            self._evaluator.n_datums += 1
            self._evaluator.n_groundtruths += len(classification.groundtruths)
            self._evaluator.n_predictions += len(classification.predictions)

            # update datum uid index
            uid_index = self._add_datum(uid=classification.uid)

            # cache labels and annotations
            keyed_groundtruths = defaultdict(int)
            keyed_predictions = defaultdict(list)
            for glabel in classification.groundtruths:
                label_idx, label_key_idx = self._add_label(glabel)
                self.groundtruth_count[label_idx][uid_index] += 1
                keyed_groundtruths[label_key_idx] = label_idx
            for idx, (plabel, pscore) in enumerate(
                zip(classification.predictions, classification.scores)
            ):
                label_idx, label_key_idx = self._add_label(plabel)
                self.prediction_count[label_idx][uid_index] += 1
                keyed_predictions[label_key_idx].append(
                    (
                        label_idx,
                        pscore,
                    )
                )

            self._add_data(
                uid_index=uid_index,
                keyed_groundtruths=keyed_groundtruths,
                keyed_predictions=keyed_predictions,
            )

    def add_data_from_valor_dict(
        self,
        classifications: list[tuple[dict, dict]],
        show_progress: bool = False,
    ):
        """
        Adds Valor-format classifications to the cache.

        Parameters
        ----------
        classifications : list[tuple[dict, dict]]
            A list of groundtruth, prediction pairs in Valor-format dictionaries.
        show_progress : bool, default=False
            Toggle for tqdm progress bar.
        """

        disable_tqdm = not show_progress
        for groundtruth, prediction in tqdm(
            classifications, disable=disable_tqdm
        ):

            # update metadata
            self._evaluator.n_datums += 1
            self._evaluator.n_groundtruths += len(groundtruth["annotations"])
            self._evaluator.n_predictions += len(prediction["annotations"])

            # update datum uid index
            uid_index = self._add_datum(uid=groundtruth["datum"]["uid"])

            # cache labels and annotations
            keyed_groundtruths = defaultdict(int)
            keyed_predictions = defaultdict(list)
            for gann in groundtruth["annotations"]:
                for valor_label in gann["labels"]:
                    glabel = (valor_label["key"], valor_label["value"])
                    label_idx, label_key_idx = self._add_label(glabel)
                    self.groundtruth_count[label_idx][uid_index] += 1
                    keyed_groundtruths[label_key_idx] = label_idx
            for pann in prediction["annotations"]:
                for valor_label in pann["labels"]:
                    plabel = (valor_label["key"], valor_label["value"])
                    pscore = valor_label["score"]
                    label_idx, label_key_idx = self._add_label(plabel)
                    self.prediction_count[label_idx][uid_index] += 1
                    keyed_predictions[label_key_idx].append(
                        (
                            label_idx,
                            pscore,
                        )
                    )

            self._add_data(
                uid_index=uid_index,
                keyed_groundtruths=keyed_groundtruths,
                keyed_predictions=keyed_predictions,
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
                    self._evaluator.label_index_to_label_key_index[label_idx],
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
