from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from valor_lite.classification.annotation import Classification
from valor_lite.classification.computation import compute_metrics
from valor_lite.classification.metric import (
    F1,
    ROCAUC,
    Accuracy,
    Counts,
    DetailedPrecisionRecallCurve,
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


class Evaluator:
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
        self._compact_pairs = np.array([])
        self._label_metadata = np.array([], dtype=np.int32)
        self._label_metadata_per_datum = np.array([], dtype=np.int32)

    @property
    def ignored_prediction_labels(self) -> list[tuple[str, str]]:
        glabels = set(np.where(self._label_metadata[:, 0] > 0)[0])
        plabels = set(np.where(self._label_metadata[:, 1] > 0)[0])
        return [
            self.index_to_label[label_id] for label_id in (plabels - glabels)
        ]

    @property
    def missing_prediction_labels(self) -> list[tuple[str, str]]:
        glabels = set(np.where(self._label_metadata[:, 0] > 0)[0])
        plabels = set(np.where(self._label_metadata[:, 1] > 0)[0])
        return [
            self.index_to_label[label_id] for label_id in (glabels - plabels)
        ]

    @property
    def metadata(self) -> dict:
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

        return Filter(
            indices=np.where(mask_pairs)[0],
            label_metadata=label_metadata,
        )

    def evaluate(
        self,
        score_thresholds: list[float] = [0.5],
        filter_: Filter | None = None,
    ) -> dict[MetricType, list]:
        """
        Runs evaluation over cached data.

        Parameters
        ----------
        score_thresholds : list[float]
            A list of score thresholds to compute over.
        filter_mask : NDArray[bool], optional
            A boolean mask that filters the cached data.
        """

        data = self._compact_pairs
        label_metadata = self._label_metadata
        if filter_ is not None:
            data = data[filter_.indices]
            label_metadata = filter_.label_metadata

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
            n_datums=self.n_datums,  # FIXME - This currently breaks filtering
        )

        metrics = defaultdict(list)

        metrics[MetricType.ROCAUC] = [
            ROCAUC(
                value=rocauc[label_idx],
                label=self.index_to_label[label_idx],
            )
            for label_idx in range(label_metadata.shape[0])
        ]

        metrics[MetricType.mROCAUC] = [
            mROCAUC(
                value=mean_rocauc[label_key_idx],
                label_key=self.index_to_label_key[label_key_idx],
            )
            for label_key_idx in range(len(self.label_key_to_index))
        ]

        for score_idx, score_threshold in enumerate(score_thresholds):
            for label_idx, label in self.index_to_label.items():
                row = counts[score_idx][label_idx]
                kwargs = {
                    "label": label,
                    "score_threshold": score_threshold,
                }
                metrics[MetricType.Counts].append(
                    Counts(
                        tp=int(row[0]),
                        fp=int(row[1]),
                        fn=int(row[2]),
                        tn=int(row[3]),
                        **kwargs,
                    )
                )
                metrics[MetricType.Precision].append(
                    Precision(
                        value=precision[score_idx][label_idx],
                        **kwargs,
                    )
                )
                metrics[MetricType.Recall].append(
                    Recall(
                        value=recall[score_idx][label_idx],
                        **kwargs,
                    )
                )
                metrics[MetricType.Accuracy].append(
                    Accuracy(
                        value=accuracy[score_idx][label_idx],
                        **kwargs,
                    )
                )
                metrics[MetricType.F1].append(
                    F1(
                        value=f1_score[score_idx][label_idx],
                        **kwargs,
                    )
                )

        return metrics

    def compute_detailed_pr_curve(
        self,
        score_thresholds: list[float] = [
            score / 10.0 for score in range(1, 11)
        ],
        n_samples: int = 0,
    ) -> list[DetailedPrecisionRecallCurve]:
        return list()

    #     if self._detailed_pairs.size == 0:
    #         return list()

    #     metrics = compute_detailed_pr_curve(
    #         self._detailed_pairs,
    #         label_counts=self._label_metadata,
    #         score_thresholds=np.array(score_thresholds),
    #         n_samples=n_samples,
    #     )

    #     tp_idx = 0
    #     fp_misclf_idx = tp_idx + n_samples + 1
    #     fp_halluc_idx = fp_misclf_idx + n_samples + 1
    #     fn_misclf_idx = fp_halluc_idx + n_samples + 1
    #     fn_misprd_idx = fn_misclf_idx + n_samples + 1

    #     results = list()
    #     for label_idx in range(len(metrics)):
    #         n_scores, _, _ = metrics.shape
    #         curve = DetailedPrecisionRecallCurve(
    #             value=list(),
    #             label=self.index_to_label[label_idx],
    #         )
    #         for score_idx in range(n_scores):
    #             curve.value.append(
    #                 DetailedPrecisionRecallPoint(
    #                     score=score_thresholds[score_idx],
    #                     tp=metrics[score_idx][label_idx][tp_idx],
    #                     tp_examples=[
    #                         self.index_to_uid[int(datum_idx)]
    #                         for datum_idx in metrics[score_idx][label_idx][
    #                             tp_idx + 1 : fp_misclf_idx
    #                         ]
    #                         if int(datum_idx) >= 0
    #                     ],
    #                     fp_misclassification=metrics[score_idx][label_idx][
    #                         fp_misclf_idx
    #                     ],
    #                     fp_misclassification_examples=[
    #                         self.index_to_uid[int(datum_idx)]
    #                         for datum_idx in metrics[score_idx][label_idx][
    #                             fp_misclf_idx + 1 : fp_halluc_idx
    #                         ]
    #                         if int(datum_idx) >= 0
    #                     ],
    #                     fp_hallucination=metrics[score_idx][label_idx][
    #                         fp_halluc_idx
    #                     ],
    #                     fp_hallucination_examples=[
    #                         self.index_to_uid[int(datum_idx)]
    #                         for datum_idx in metrics[score_idx][label_idx][
    #                             fp_halluc_idx + 1 : fn_misclf_idx
    #                         ]
    #                         if int(datum_idx) >= 0
    #                     ],
    #                     fn_misclassification=metrics[score_idx][label_idx][
    #                         fn_misclf_idx
    #                     ],
    #                     fn_misclassification_examples=[
    #                         self.index_to_uid[int(datum_idx)]
    #                         for datum_idx in metrics[score_idx][label_idx][
    #                             fn_misclf_idx + 1 : fn_misprd_idx
    #                         ]
    #                         if int(datum_idx) >= 0
    #                     ],
    #                     fn_missing_prediction=metrics[score_idx][label_idx][
    #                         fn_misprd_idx
    #                     ],
    #                     fn_missing_prediction_examples=[
    #                         self.index_to_uid[int(datum_idx)]
    #                         for datum_idx in metrics[score_idx][label_idx][
    #                             fn_misprd_idx + 1 :
    #                         ]
    #                         if int(datum_idx) >= 0
    #                     ],
    #                 )
    #             )
    #         results.append(curve)
    #     return results


class DataLoader:
    def __init__(self):
        self._evaluator = Evaluator()
        self.groundtruth_count = defaultdict(lambda: defaultdict(int))
        self.prediction_count = defaultdict(lambda: defaultdict(int))

    def _add_datum(self, uid: str) -> int:
        if uid not in self._evaluator.uid_to_index:
            index = len(self._evaluator.uid_to_index)
            self._evaluator.uid_to_index[uid] = index
            self._evaluator.index_to_uid[index] = uid
        return self._evaluator.uid_to_index[uid]

    def _add_label(self, label: tuple[str, str]) -> tuple[int, int]:
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

    def add_data(
        self,
        classifications: list[Classification],
        show_progress: bool = False,
    ):
        disable_tqdm = not show_progress
        for classification in tqdm(classifications, disable=disable_tqdm):

            # update metadata
            self._evaluator.n_datums += 1
            self._evaluator.n_groundtruths += len(classification.groundtruths)
            self._evaluator.n_predictions += len(classification.predictions)

            # update datum uid index
            uid_index = self._add_datum(uid=classification.uid)

            # cache labels and annotations
            keyed_groundtruths = defaultdict(list)
            keyed_predictions = defaultdict(list)
            for glabel in classification.groundtruths:
                label_idx, label_key_idx = self._add_label(glabel)
                self.groundtruth_count[label_idx][uid_index] += 1
                keyed_groundtruths[label_key_idx].append(label_idx)
            for plabel, pscore in zip(
                classification.predictions, classification.scores
            ):
                label_idx, label_key_idx = self._add_label(plabel)
                self.prediction_count[label_idx][uid_index] += 1
                keyed_predictions[label_key_idx].append(
                    (
                        label_idx,
                        pscore,
                    )
                )

            gt_keys = set(keyed_groundtruths.keys())
            pd_keys = set(keyed_predictions.keys())
            joint_keys = gt_keys.intersection(pd_keys)
            gt_unique_keys = gt_keys - pd_keys
            pd_unique_keys = pd_keys - gt_keys

            pairs = list()
            for key in joint_keys:
                pairs.extend(
                    [
                        np.array(
                            [
                                float(uid_index),
                                float(glabel),
                                float(plabel),
                                float(score),
                            ]
                        )
                        for plabel, score in keyed_predictions[key]
                        for glabel in keyed_groundtruths[key]
                    ]
                )
            for key in gt_unique_keys:
                pairs.extend(
                    [
                        np.array(
                            [
                                float(uid_index),
                                float(glabel),
                                -1.0,
                                -1.0,
                            ]
                        )
                        for glabel in keyed_groundtruths[key]
                    ]
                )
            for key in pd_unique_keys:
                pairs.extend(
                    [
                        np.array(
                            [
                                float(uid_index),
                                -1.0,
                                float(plabel),
                                float(score),
                            ]
                        )
                        for plabel, score in keyed_predictions[key]
                    ]
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

    def add_data_from_valor_dict(
        self,
        classifications: list[tuple[dict, dict]],
        show_progress: bool = False,
    ):

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
            keyed_groundtruths = defaultdict(list)
            keyed_predictions = defaultdict(list)
            for gann in groundtruth["annotations"]:
                for valor_label in gann["labels"]:
                    glabel = (valor_label["key"], valor_label["value"])
                    label_idx, label_key_idx = self._add_label(glabel)
                    self.groundtruth_count[label_idx][uid_index] += 1
                    keyed_groundtruths[label_key_idx].append(label_idx)
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

            gt_keys = set(keyed_groundtruths.keys())
            pd_keys = set(keyed_predictions.keys())
            joint_keys = gt_keys.intersection(pd_keys)
            gt_unique_keys = gt_keys - pd_keys
            pd_unique_keys = pd_keys - gt_keys

            pairs = list()
            for key in joint_keys:
                pairs.extend(
                    [
                        np.array(
                            [
                                float(uid_index),
                                float(glabel),
                                float(plabel),
                                float(score),
                            ]
                        )
                        for plabel, score in keyed_predictions[key]
                        for glabel in keyed_groundtruths[key]
                    ]
                )
            for key in gt_unique_keys:
                pairs.extend(
                    [
                        np.array(
                            [
                                float(uid_index),
                                float(glabel),
                                -1.0,
                                -1.0,
                            ]
                        )
                        for glabel in keyed_groundtruths[key]
                    ]
                )
            for key in pd_unique_keys:
                pairs.extend(
                    [
                        np.array(
                            [
                                float(uid_index),
                                -1.0,
                                float(plabel),
                                float(score),
                            ]
                        )
                        for plabel, score in keyed_predictions[key]
                    ]
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

    def finalize(self) -> Evaluator:

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

        # verify that all predictions contain all labels
        if not np.isclose(
            self._evaluator._label_metadata[0, 1],
            self._evaluator._label_metadata[0, 1],
        ).all():
            raise ValueError

        # remove datums for compact representation
        self._evaluator._compact_pairs = self._evaluator._detailed_pairs[
            :, 1:
        ].copy()

        # sort compact pairs by groundtruth, prediction, score
        indices = np.lexsort(
            (
                self._evaluator._compact_pairs[:, 0],
                self._evaluator._compact_pairs[:, 1],
                -self._evaluator._compact_pairs[:, 2],
            )
        )
        self._evaluator._compact_pairs = self._evaluator._compact_pairs[
            indices
        ]

        return self._evaluator
