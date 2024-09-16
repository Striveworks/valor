from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from valor_lite.detection.annotation import Detection
from valor_lite.detection.computation import (
    compute_detailed_pr_curve,
    compute_iou,
    compute_metrics,
    compute_ranked_pairs,
)
from valor_lite.detection.metric import (
    AP,
    AR,
    F1,
    Accuracy,
    APAveragedOverIOUs,
    ARAveragedOverScores,
    Counts,
    DetailedPrecisionRecallCurve,
    DetailedPrecisionRecallPoint,
    MetricType,
    Precision,
    PrecisionRecallCurve,
    Recall,
    mAP,
    mAPAveragedOverIOUs,
    mAR,
    mARAveragedOverScores,
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

metrics = evaluator.evaluate(iou_thresholds=[0.5])

ap_metrics = metrics[MetricType.AP]
ar_metrics = metrics[MetricType.AR]

filter_mask = evaluator.create_filter(datum_uids=["uid1", "uid2"])
filtered_metrics = evaluator.evaluate(iou_thresholds=[0.5], filter_mask=filter_mask)
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
        self._ranked_pairs = np.array([])
        self._label_metadata = np.array([])
        self._label_metadata_per_datum = np.array([])

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
        n_rows = self._ranked_pairs.shape[0]

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
                np.isin(self._ranked_pairs[:, 0].astype(int), datum_uids)
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
            mask[np.isin(self._ranked_pairs[:, 4].astype(int), labels)] = True
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
                np.isin(self._ranked_pairs[:, 4].astype(int), label_indices)
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
            # uids=datum_uids,
            # labels=labels,
            # label_keys=label_keys,
        )

    def evaluate(
        self,
        iou_thresholds: list[float] = [0.5, 0.75, 0.9],
        score_thresholds: list[float] = [0.5],
        filter_: Filter | None = None,
    ) -> dict[MetricType, list]:
        """
        Runs evaluation over cached data.

        Parameters
        ----------
        iou_thresholds : list[float]
            A list of iou thresholds to compute over.
        score_thresholds : list[float]
            A list of score thresholds to compute over.
        filter_mask : NDArray[bool], optional
            A boolean mask that filters the cached data.
        """

        data = self._ranked_pairs
        label_metadata = self._label_metadata
        if filter_ is not None:
            data = data[filter_.indices]
            label_metadata = filter_.label_metadata

        (
            (
                average_precision,
                mean_average_precision,
                average_precision_average_over_ious,
                mean_average_precision_average_over_ious,
            ),
            (
                average_recall,
                mean_average_recall,
                average_recall_averaged_over_scores,
                mean_average_recall_averaged_over_scores,
            ),
            precision_recall,
            pr_curves,
        ) = compute_metrics(
            data=data,
            label_counts=label_metadata,
            iou_thresholds=np.array(iou_thresholds),
            score_thresholds=np.array(score_thresholds),
        )

        metrics = defaultdict(list)

        metrics[MetricType.AP] = [
            AP(
                value=average_precision[iou_idx][label_idx],
                iou=iou_thresholds[iou_idx],
                label=self.index_to_label[label_idx],
            )
            for iou_idx in range(average_precision.shape[0])
            for label_idx in range(average_precision.shape[1])
            if int(label_metadata[label_idx][0]) > 0
        ]

        metrics[MetricType.mAP] = [
            mAP(
                value=mean_average_precision[iou_idx][label_key_idx],
                iou=iou_thresholds[iou_idx],
                label_key=self.index_to_label_key[label_key_idx],
            )
            for iou_idx in range(mean_average_precision.shape[0])
            for label_key_idx in range(mean_average_precision.shape[1])
        ]

        metrics[MetricType.APAveragedOverIOUs] = [
            APAveragedOverIOUs(
                value=average_precision_average_over_ious[label_idx],
                ious=iou_thresholds,
                label=self.index_to_label[label_idx],
            )
            for label_idx in range(self.n_labels)
            if int(label_metadata[label_idx][0]) > 0
        ]

        metrics[MetricType.mAPAveragedOverIOUs] = [
            mAPAveragedOverIOUs(
                value=mean_average_precision_average_over_ious[label_key_idx],
                ious=iou_thresholds,
                label_key=self.index_to_label_key[label_key_idx],
            )
            for label_key_idx in range(
                mean_average_precision_average_over_ious.shape[0]
            )
        ]

        metrics[MetricType.AR] = [
            AR(
                value=average_recall[score_idx][label_idx],
                ious=iou_thresholds,
                score=score_thresholds[score_idx],
                label=self.index_to_label[label_idx],
            )
            for score_idx in range(average_recall.shape[0])
            for label_idx in range(average_recall.shape[1])
            if int(label_metadata[label_idx][0]) > 0
        ]

        metrics[MetricType.mAR] = [
            mAR(
                value=mean_average_recall[score_idx][label_key_idx],
                ious=iou_thresholds,
                score=score_thresholds[score_idx],
                label_key=self.index_to_label_key[label_key_idx],
            )
            for score_idx in range(mean_average_recall.shape[0])
            for label_key_idx in range(mean_average_recall.shape[1])
        ]

        metrics[MetricType.ARAveragedOverScores] = [
            ARAveragedOverScores(
                value=average_recall_averaged_over_scores[label_idx],
                scores=score_thresholds,
                ious=iou_thresholds,
                label=self.index_to_label[label_idx],
            )
            for label_idx in range(self.n_labels)
            if int(label_metadata[label_idx][0]) > 0
        ]

        metrics[MetricType.mARAveragedOverScores] = [
            mARAveragedOverScores(
                value=mean_average_recall_averaged_over_scores[label_key_idx],
                scores=score_thresholds,
                ious=iou_thresholds,
                label_key=self.index_to_label_key[label_key_idx],
            )
            for label_key_idx in range(
                mean_average_recall_averaged_over_scores.shape[0]
            )
        ]

        metrics[MetricType.PrecisionRecallCurve] = [
            PrecisionRecallCurve(
                precision=list(pr_curves[iou_idx][label_idx]),
                iou=iou_threshold,
                label=label,
            )
            for iou_idx, iou_threshold in enumerate(iou_thresholds)
            for label_idx, label in self.index_to_label.items()
            if int(label_metadata[label_idx][0]) > 0
        ]

        for iou_idx, iou_threshold in enumerate(iou_thresholds):
            for score_idx, score_threshold in enumerate(score_thresholds):
                for label_idx, label in self.index_to_label.items():
                    row = precision_recall[iou_idx][score_idx][label_idx]
                    kwargs = {
                        "label": label,
                        "iou": iou_threshold,
                        "score": score_threshold,
                    }
                    metrics[MetricType.Counts].append(
                        Counts(
                            tp=int(row[0]),
                            fp=int(row[1]),
                            fn=int(row[2]),
                            **kwargs,
                        )
                    )
                    metrics[MetricType.Precision].append(
                        Precision(
                            value=row[3],
                            **kwargs,
                        )
                    )
                    metrics[MetricType.Recall].append(
                        Recall(
                            value=row[4],
                            **kwargs,
                        )
                    )
                    metrics[MetricType.F1].append(
                        F1(
                            value=row[5],
                            **kwargs,
                        )
                    )
                    metrics[MetricType.Accuracy].append(
                        Accuracy(
                            value=row[6],
                            **kwargs,
                        )
                    )

        return metrics

    def compute_detailed_pr_curve(
        self,
        iou_thresholds: list[float] = [0.5],
        score_thresholds: list[float] = [
            score / 10.0 for score in range(1, 11)
        ],
        n_samples: int = 0,
    ) -> list[DetailedPrecisionRecallCurve]:

        if self._detailed_pairs.size == 0:
            return list()

        metrics = compute_detailed_pr_curve(
            self._detailed_pairs,
            label_counts=self._label_metadata,
            iou_thresholds=np.array(iou_thresholds),
            score_thresholds=np.array(score_thresholds),
            n_samples=n_samples,
        )

        tp_idx = 0
        fp_misclf_idx = tp_idx + n_samples + 1
        fp_halluc_idx = fp_misclf_idx + n_samples + 1
        fn_misclf_idx = fp_halluc_idx + n_samples + 1
        fn_misprd_idx = fn_misclf_idx + n_samples + 1

        results = list()
        for label_idx in range(len(metrics)):
            n_ious, n_scores, _, _ = metrics.shape
            for iou_idx in range(n_ious):
                curve = DetailedPrecisionRecallCurve(
                    iou=iou_thresholds[iou_idx],
                    value=list(),
                    label=self.index_to_label[label_idx],
                )
                for score_idx in range(n_scores):
                    curve.value.append(
                        DetailedPrecisionRecallPoint(
                            score=score_thresholds[score_idx],
                            tp=metrics[iou_idx][score_idx][label_idx][tp_idx],
                            tp_examples=[
                                self.index_to_uid[int(datum_idx)]
                                for datum_idx in metrics[iou_idx][score_idx][
                                    label_idx
                                ][tp_idx + 1 : fp_misclf_idx]
                                if int(datum_idx) >= 0
                            ],
                            fp_misclassification=metrics[iou_idx][score_idx][
                                label_idx
                            ][fp_misclf_idx],
                            fp_misclassification_examples=[
                                self.index_to_uid[int(datum_idx)]
                                for datum_idx in metrics[iou_idx][score_idx][
                                    label_idx
                                ][fp_misclf_idx + 1 : fp_halluc_idx]
                                if int(datum_idx) >= 0
                            ],
                            fp_hallucination=metrics[iou_idx][score_idx][
                                label_idx
                            ][fp_halluc_idx],
                            fp_hallucination_examples=[
                                self.index_to_uid[int(datum_idx)]
                                for datum_idx in metrics[iou_idx][score_idx][
                                    label_idx
                                ][fp_halluc_idx + 1 : fn_misclf_idx]
                                if int(datum_idx) >= 0
                            ],
                            fn_misclassification=metrics[iou_idx][score_idx][
                                label_idx
                            ][fn_misclf_idx],
                            fn_misclassification_examples=[
                                self.index_to_uid[int(datum_idx)]
                                for datum_idx in metrics[iou_idx][score_idx][
                                    label_idx
                                ][fn_misclf_idx + 1 : fn_misprd_idx]
                                if int(datum_idx) >= 0
                            ],
                            fn_missing_prediction=metrics[iou_idx][score_idx][
                                label_idx
                            ][fn_misprd_idx],
                            fn_missing_prediction_examples=[
                                self.index_to_uid[int(datum_idx)]
                                for datum_idx in metrics[iou_idx][score_idx][
                                    label_idx
                                ][fn_misprd_idx + 1 :]
                                if int(datum_idx) >= 0
                            ],
                        )
                    )
                results.append(curve)
        return results


class DataLoader:
    def __init__(self):
        self._evaluator = Evaluator()
        self.pairs = list()
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
        detections: list[Detection],
        show_progress: bool = False,
    ):
        disable_tqdm = not show_progress
        for detection in tqdm(detections, disable=disable_tqdm):

            # update metadata
            self._evaluator.n_datums += 1
            self._evaluator.n_groundtruths += len(detection.groundtruths)
            self._evaluator.n_predictions += len(detection.predictions)

            # update datum uid index
            uid_index = self._add_datum(uid=detection.uid)

            # cache labels and annotations
            keyed_groundtruths = defaultdict(list)
            keyed_predictions = defaultdict(list)
            for gidx, gann in enumerate(detection.groundtruths):
                for glabel in gann.labels:
                    label_idx, label_key_idx = self._add_label(glabel)
                    self.groundtruth_count[label_idx][uid_index] += 1
                    keyed_groundtruths[label_key_idx].append(
                        (
                            gidx,
                            label_idx,
                            gann.extrema,
                        )
                    )
            for pidx, pann in enumerate(detection.predictions):
                for plabel, pscore in zip(pann.labels, pann.scores):
                    label_idx, label_key_idx = self._add_label(plabel)
                    self.prediction_count[label_idx][uid_index] += 1
                    keyed_predictions[label_key_idx].append(
                        (
                            pidx,
                            label_idx,
                            pscore,
                            pann.extrema,
                        )
                    )

            gt_keys = set(keyed_groundtruths.keys())
            pd_keys = set(keyed_predictions.keys())
            joint_keys = gt_keys.intersection(pd_keys)
            gt_unique_keys = gt_keys - pd_keys
            pd_unique_keys = pd_keys - gt_keys

            pairs = list()
            for key in joint_keys:
                boxes = np.array(
                    [
                        np.array([*gextrema, *pextrema])
                        for _, _, _, pextrema in keyed_predictions[key]
                        for _, _, gextrema in keyed_groundtruths[key]
                    ]
                )
                ious = compute_iou(boxes)
                pairs.extend(
                    [
                        np.array(
                            [
                                float(uid_index),
                                float(gidx),
                                float(pidx),
                                ious[
                                    pidx * len(keyed_groundtruths[key]) + gidx
                                ],
                                float(glabel),
                                float(plabel),
                                float(score),
                            ]
                        )
                        for pidx, plabel, score, _ in keyed_predictions[key]
                        for gidx, glabel, _ in keyed_groundtruths[key]
                    ]
                )
            for key in gt_unique_keys:
                pairs.extend(
                    [
                        np.array(
                            [
                                float(uid_index),
                                float(gidx),
                                -1.0,
                                0.0,
                                float(glabel),
                                -1.0,
                                -1.0,
                            ]
                        )
                        for gidx, glabel, _ in keyed_groundtruths[key]
                    ]
                )
            for key in pd_unique_keys:
                pairs.extend(
                    [
                        np.array(
                            [
                                float(uid_index),
                                -1.0,
                                float(pidx),
                                0.0,
                                -1.0,
                                float(plabel),
                                float(score),
                            ]
                        )
                        for pidx, plabel, score, _ in keyed_predictions[key]
                    ]
                )

            self.pairs.append(np.array(pairs))

    def add_data_from_valor_dict(
        self,
        detections: list[tuple[dict, dict]],
        show_progress: bool = False,
    ):
        def _get_bbox_extrema(
            data: list[list[list[float]]],
        ) -> tuple[float, float, float, float]:
            x = [point[0] for shape in data for point in shape]
            y = [point[1] for shape in data for point in shape]
            return (min(x), max(x), min(y), max(y))

        disable_tqdm = not show_progress
        for groundtruth, prediction in tqdm(detections, disable=disable_tqdm):

            # update metadata
            self._evaluator.n_datums += 1
            self._evaluator.n_groundtruths += len(groundtruth["annotations"])
            self._evaluator.n_predictions += len(prediction["annotations"])

            # update datum uid index
            uid_index = self._add_datum(uid=groundtruth["datum"]["uid"])

            # cache labels and annotations
            keyed_groundtruths = defaultdict(list)
            keyed_predictions = defaultdict(list)
            for gidx, gann in enumerate(groundtruth["annotations"]):
                for valor_label in gann["labels"]:
                    glabel = (valor_label["key"], valor_label["value"])
                    label_idx, label_key_idx = self._add_label(glabel)
                    self.groundtruth_count[label_idx][uid_index] += 1
                    keyed_groundtruths[label_key_idx].append(
                        (
                            gidx,
                            label_idx,
                            _get_bbox_extrema(gann["bounding_box"]),
                        )
                    )
            for pidx, pann in enumerate(prediction["annotations"]):
                for valor_label in pann["labels"]:
                    plabel = (valor_label["key"], valor_label["value"])
                    pscore = valor_label["score"]
                    label_idx, label_key_idx = self._add_label(plabel)
                    self.prediction_count[label_idx][uid_index] += 1
                    keyed_predictions[label_key_idx].append(
                        (
                            pidx,
                            label_idx,
                            pscore,
                            _get_bbox_extrema(pann["bounding_box"]),
                        )
                    )

            gt_keys = set(keyed_groundtruths.keys())
            pd_keys = set(keyed_predictions.keys())
            joint_keys = gt_keys.intersection(pd_keys)
            gt_unique_keys = gt_keys - pd_keys
            pd_unique_keys = pd_keys - gt_keys

            pairs = list()
            for key in joint_keys:
                boxes = np.array(
                    [
                        np.array([*gextrema, *pextrema])
                        for _, _, _, pextrema in keyed_predictions[key]
                        for _, _, gextrema in keyed_groundtruths[key]
                    ]
                )
                ious = compute_iou(boxes)
                pairs.extend(
                    [
                        np.array(
                            [
                                float(uid_index),
                                float(gidx),
                                float(pidx),
                                ious[
                                    pidx * len(keyed_groundtruths[key]) + gidx
                                ],
                                float(glabel),
                                float(plabel),
                                float(score),
                            ]
                        )
                        for pidx, plabel, score, _ in keyed_predictions[key]
                        for gidx, glabel, _ in keyed_groundtruths[key]
                    ]
                )
            for key in gt_unique_keys:
                pairs.extend(
                    [
                        np.array(
                            [
                                float(uid_index),
                                float(gidx),
                                -1.0,
                                0.0,
                                float(glabel),
                                -1.0,
                                -1.0,
                            ]
                        )
                        for gidx, glabel, _ in keyed_groundtruths[key]
                    ]
                )
            for key in pd_unique_keys:
                pairs.extend(
                    [
                        np.array(
                            [
                                float(uid_index),
                                -1.0,
                                float(pidx),
                                0.0,
                                -1.0,
                                float(plabel),
                                float(score),
                            ]
                        )
                        for pidx, plabel, score, _ in keyed_predictions[key]
                    ]
                )

            self.pairs.append(np.array(pairs))

    def finalize(self) -> Evaluator:

        self.pairs = [pair for pair in self.pairs if pair.size > 0]
        if len(self.pairs) == 0:
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
                    float(
                        np.sum(
                            self._evaluator._label_metadata_per_datum[
                                0, :, label_idx
                            ]
                        )
                    ),
                    float(
                        np.sum(
                            self._evaluator._label_metadata_per_datum[
                                1, :, label_idx
                            ]
                        )
                    ),
                    float(
                        self._evaluator.label_index_to_label_key_index[
                            label_idx
                        ]
                    ),
                ]
                for label_idx in range(n_labels)
            ]
        )

        self._evaluator._detailed_pairs = np.concatenate(
            self.pairs,
            axis=0,
        )

        self._evaluator._ranked_pairs = compute_ranked_pairs(
            self.pairs,
            label_counts=self._evaluator._label_metadata,
        )

        return self._evaluator
