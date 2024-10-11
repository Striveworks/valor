from collections import defaultdict
from dataclasses import dataclass
from typing import Type

import numpy as np
import valor_lite.object_detection.annotation as annotation
from numpy.typing import NDArray
from tqdm import tqdm
from valor_lite.object_detection.annotation import (
    Bitmask,
    BoundingBox,
    Detection,
    Polygon,
)
from valor_lite.object_detection.computation import (
    compute_bbox_iou,
    compute_bitmask_iou,
    compute_confusion_matrix,
    compute_metrics,
    compute_polygon_iou,
    compute_ranked_pairs,
)
from valor_lite.object_detection.metric import (
    AP,
    AR,
    F1,
    Accuracy,
    APAveragedOverIOUs,
    ARAveragedOverScores,
    ConfusionMatrix,
    Counts,
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

loader = DataLoader()
loader.add_bounding_boxes(
    groundtruths=groundtruths,
    predictions=predictions,
)
evaluator = loader.finalize()

metrics = evaluator.evaluate(iou_thresholds=[0.5])

ap_metrics = metrics[MetricType.AP]
ar_metrics = metrics[MetricType.AR]

filter_mask = evaluator.create_filter(datum_uids=["uid1", "uid2"])
filtered_metrics = evaluator.evaluate(iou_thresholds=[0.5], filter_mask=filter_mask)
"""


@dataclass
class Filter:
    ranked_indices: NDArray[np.int32]
    detailed_indices: NDArray[np.int32]
    label_metadata: NDArray[np.int32]


class Evaluator:
    """
    Object Detection Evaluator
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

        # annotation reference
        self.groundtruth_examples: dict[int, NDArray[np.float16]] = dict()
        self.prediction_examples: dict[int, NDArray[np.float16]] = dict()

        # label reference
        self.label_to_index: dict[str, int] = dict()
        self.index_to_label: dict[int, str] = dict()

        # computation caches
        self._detailed_pairs: NDArray[np.float64] = np.array([])
        self._ranked_pairs: NDArray[np.float64] = np.array([])
        self._label_metadata: NDArray[np.int32] = np.array([])
        self._label_metadata_per_datum: NDArray[np.int32] = np.array([])

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
        Creates a filter that can be passed to an evaluation.

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

        n_datums = self._label_metadata_per_datum.shape[1]
        n_labels = self._label_metadata_per_datum.shape[2]

        mask_ranked = np.ones((self._ranked_pairs.shape[0], 1), dtype=np.bool_)
        mask_detailed = np.ones(
            (self._detailed_pairs.shape[0], 1), dtype=np.bool_
        )
        mask_datums = np.ones(n_datums, dtype=np.bool_)
        mask_labels = np.ones(n_labels, dtype=np.bool_)

        if datum_uids is not None:
            if isinstance(datum_uids, list):
                datum_uids = np.array(
                    [self.uid_to_index[uid] for uid in datum_uids],
                    dtype=np.int32,
                )
            mask_ranked[
                ~np.isin(self._ranked_pairs[:, 0].astype(int), datum_uids)
            ] = False
            mask_detailed[
                ~np.isin(self._detailed_pairs[:, 0].astype(int), datum_uids)
            ] = False
            mask_datums[~np.isin(np.arange(n_datums), datum_uids)] = False

        if labels is not None:
            if isinstance(labels, list):
                labels = np.array(
                    [self.label_to_index[label] for label in labels]
                )
            mask_ranked[
                ~np.isin(self._ranked_pairs[:, 4].astype(int), labels)
            ] = False
            mask_detailed[
                ~np.isin(self._detailed_pairs[:, 4].astype(int), labels)
            ] = False
            mask_labels[~np.isin(np.arange(n_labels), labels)] = False

        mask_label_metadata = (
            mask_datums[:, np.newaxis] & mask_labels[np.newaxis, :]
        )
        label_metadata_per_datum = self._label_metadata_per_datum.copy()
        label_metadata_per_datum[:, ~mask_label_metadata] = 0

        label_metadata = np.zeros_like(self._label_metadata, dtype=np.int32)
        label_metadata = np.transpose(
            np.sum(
                label_metadata_per_datum,
                axis=1,
            )
        )

        return Filter(
            ranked_indices=np.where(mask_ranked)[0],
            detailed_indices=np.where(mask_detailed)[0],
            label_metadata=label_metadata,
        )

    def _convert_example_to_dict(
        self, box: NDArray[np.float16]
    ) -> dict[str, float]:
        """
        Converts a cached bounding box example to dictionary format.
        """
        return {
            "xmin": float(box[0]),
            "xmax": float(box[1]),
            "ymin": float(box[2]),
            "ymax": float(box[3]),
        }

    def _unpack_confusion_matrix(
        self,
        confusion_matrix: NDArray[np.float64],
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
                        str | dict[str, float] | float,
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
                example_idx * 4 + 1,
            ]
        )

        groundtruth_idx = lambda gt_label_idx, pd_label_idx, example_idx: int(  # noqa: E731 - lambda fn
            confusion_matrix[
                gt_label_idx,
                pd_label_idx,
                example_idx * 4 + 2,
            ]
        )

        prediction_idx = lambda gt_label_idx, pd_label_idx, example_idx: int(  # noqa: E731 - lambda fn
            confusion_matrix[
                gt_label_idx,
                pd_label_idx,
                example_idx * 4 + 3,
            ]
        )

        score_idx = lambda gt_label_idx, pd_label_idx, example_idx: float(  # noqa: E731 - lambda fn
            confusion_matrix[
                gt_label_idx,
                pd_label_idx,
                example_idx * 4 + 4,
            ]
        )

        return {
            self.index_to_label[gt_label_idx]: {
                self.index_to_label[pd_label_idx]: {
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
                            "groundtruth": self._convert_example_to_dict(
                                self.groundtruth_examples[
                                    datum_idx(
                                        gt_label_idx,
                                        pd_label_idx,
                                        example_idx,
                                    )
                                ][
                                    groundtruth_idx(
                                        gt_label_idx,
                                        pd_label_idx,
                                        example_idx,
                                    )
                                ]
                            ),
                            "prediction": self._convert_example_to_dict(
                                self.prediction_examples[
                                    datum_idx(
                                        gt_label_idx,
                                        pd_label_idx,
                                        example_idx,
                                    )
                                ][
                                    prediction_idx(
                                        gt_label_idx,
                                        pd_label_idx,
                                        example_idx,
                                    )
                                ]
                            ),
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
            }
            for gt_label_idx in range(number_of_labels)
        }

    def _unpack_hallucinations(
        self,
        hallucinations: NDArray[np.float64],
        number_of_labels: int,
        number_of_examples: int,
    ) -> dict[
        str,
        dict[
            str,
            int | list[dict[str, str | float | dict[str, float]]],
        ],
    ]:
        """
        Unpacks a numpy array of hallucination counts and examples.
        """

        datum_idx = (
            lambda pd_label_idx, example_idx: int(  # noqa: E731 - lambda fn
                hallucinations[
                    pd_label_idx,
                    example_idx * 3 + 1,
                ]
            )
        )

        prediction_idx = (
            lambda pd_label_idx, example_idx: int(  # noqa: E731 - lambda fn
                hallucinations[
                    pd_label_idx,
                    example_idx * 3 + 2,
                ]
            )
        )

        score_idx = (
            lambda pd_label_idx, example_idx: float(  # noqa: E731 - lambda fn
                hallucinations[
                    pd_label_idx,
                    example_idx * 3 + 3,
                ]
            )
        )

        return {
            self.index_to_label[pd_label_idx]: {
                "count": max(
                    int(hallucinations[pd_label_idx, 0]),
                    0,
                ),
                "examples": [
                    {
                        "datum": self.index_to_uid[
                            datum_idx(pd_label_idx, example_idx)
                        ],
                        "prediction": self._convert_example_to_dict(
                            self.prediction_examples[
                                datum_idx(pd_label_idx, example_idx)
                            ][prediction_idx(pd_label_idx, example_idx)]
                        ),
                        "score": score_idx(pd_label_idx, example_idx),
                    }
                    for example_idx in range(number_of_examples)
                    if datum_idx(pd_label_idx, example_idx) >= 0
                ],
            }
            for pd_label_idx in range(number_of_labels)
        }

    def _unpack_missing_predictions(
        self,
        missing_predictions: NDArray[np.int32],
        number_of_labels: int,
        number_of_examples: int,
    ) -> dict[str, dict[str, int | list[dict[str, str | dict[str, float]]]]]:
        """
        Unpacks a numpy array of missing prediction counts and examples.
        """

        datum_idx = (
            lambda gt_label_idx, example_idx: int(  # noqa: E731 - lambda fn
                missing_predictions[
                    gt_label_idx,
                    example_idx * 2 + 1,
                ]
            )
        )

        groundtruth_idx = (
            lambda gt_label_idx, example_idx: int(  # noqa: E731 - lambda fn
                missing_predictions[
                    gt_label_idx,
                    example_idx * 2 + 2,
                ]
            )
        )

        return {
            self.index_to_label[gt_label_idx]: {
                "count": max(
                    int(missing_predictions[gt_label_idx, 0]),
                    0,
                ),
                "examples": [
                    {
                        "datum": self.index_to_uid[
                            datum_idx(gt_label_idx, example_idx)
                        ],
                        "groundtruth": self._convert_example_to_dict(
                            self.groundtruth_examples[
                                datum_idx(gt_label_idx, example_idx)
                            ][groundtruth_idx(gt_label_idx, example_idx)]
                        ),
                    }
                    for example_idx in range(number_of_examples)
                    if datum_idx(gt_label_idx, example_idx) >= 0
                ],
            }
            for gt_label_idx in range(number_of_labels)
        }

    def compute_precision_recall(
        self,
        iou_thresholds: list[float] = [0.5, 0.75, 0.9],
        score_thresholds: list[float] = [0.5],
        filter_: Filter | None = None,
        as_dict: bool = False,
    ) -> dict[MetricType, list]:
        """
        Computes all metrics except for ConfusionMatrix

        Parameters
        ----------
        iou_thresholds : list[float]
            A list of IoU thresholds to compute metrics over.
        score_thresholds : list[float]
            A list of score thresholds to compute metrics over.
        filter_ : Filter, optional
            An optional filter object.
        as_dict : bool, default=False
            An option to return metrics as dictionaries.

        Returns
        -------
        dict[MetricType, list]
            A dictionary mapping MetricType enumerations to lists of computed metrics.
        """

        ranked_pairs = self._ranked_pairs
        label_metadata = self._label_metadata
        if filter_ is not None:
            ranked_pairs = ranked_pairs[filter_.ranked_indices]
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
            data=ranked_pairs,
            label_metadata=label_metadata,
            iou_thresholds=np.array(iou_thresholds),
            score_thresholds=np.array(score_thresholds),
        )

        metrics = defaultdict(list)

        metrics[MetricType.AP] = [
            AP(
                value=float(average_precision[iou_idx][label_idx]),
                iou_threshold=iou_thresholds[iou_idx],
                label=self.index_to_label[label_idx],
            )
            for iou_idx in range(average_precision.shape[0])
            for label_idx in range(average_precision.shape[1])
            if int(label_metadata[label_idx, 0]) > 0
        ]

        metrics[MetricType.mAP] = [
            mAP(
                value=float(mean_average_precision[iou_idx]),
                iou_threshold=iou_thresholds[iou_idx],
            )
            for iou_idx in range(mean_average_precision.shape[0])
        ]

        metrics[MetricType.APAveragedOverIOUs] = [
            APAveragedOverIOUs(
                value=float(average_precision_average_over_ious[label_idx]),
                iou_thresholds=iou_thresholds,
                label=self.index_to_label[label_idx],
            )
            for label_idx in range(self.n_labels)
            if int(label_metadata[label_idx, 0]) > 0
        ]

        metrics[MetricType.mAPAveragedOverIOUs] = [
            mAPAveragedOverIOUs(
                value=float(mean_average_precision_average_over_ious),
                iou_thresholds=iou_thresholds,
            )
        ]

        metrics[MetricType.AR] = [
            AR(
                value=float(average_recall[score_idx][label_idx]),
                iou_thresholds=iou_thresholds,
                score_threshold=score_thresholds[score_idx],
                label=self.index_to_label[label_idx],
            )
            for score_idx in range(average_recall.shape[0])
            for label_idx in range(average_recall.shape[1])
            if int(label_metadata[label_idx, 0]) > 0
        ]

        metrics[MetricType.mAR] = [
            mAR(
                value=float(mean_average_recall[score_idx]),
                iou_thresholds=iou_thresholds,
                score_threshold=score_thresholds[score_idx],
            )
            for score_idx in range(mean_average_recall.shape[0])
        ]

        metrics[MetricType.ARAveragedOverScores] = [
            ARAveragedOverScores(
                value=float(average_recall_averaged_over_scores[label_idx]),
                score_thresholds=score_thresholds,
                iou_thresholds=iou_thresholds,
                label=self.index_to_label[label_idx],
            )
            for label_idx in range(self.n_labels)
            if int(label_metadata[label_idx, 0]) > 0
        ]

        metrics[MetricType.mARAveragedOverScores] = [
            mARAveragedOverScores(
                value=float(mean_average_recall_averaged_over_scores),
                score_thresholds=score_thresholds,
                iou_thresholds=iou_thresholds,
            )
        ]

        metrics[MetricType.PrecisionRecallCurve] = [
            PrecisionRecallCurve(
                precisions=pr_curves[iou_idx, label_idx, :, 0]
                .astype(float)
                .tolist(),
                scores=pr_curves[iou_idx, label_idx, :, 1]
                .astype(float)
                .tolist(),
                iou_threshold=iou_threshold,
                label=label,
            )
            for iou_idx, iou_threshold in enumerate(iou_thresholds)
            for label_idx, label in self.index_to_label.items()
            if int(label_metadata[label_idx, 0]) > 0
        ]

        for label_idx, label in self.index_to_label.items():

            if label_metadata[label_idx, 0] == 0:
                continue

            for score_idx, score_threshold in enumerate(score_thresholds):
                for iou_idx, iou_threshold in enumerate(iou_thresholds):

                    row = precision_recall[iou_idx][score_idx][label_idx]
                    kwargs = {
                        "label": label,
                        "iou_threshold": iou_threshold,
                        "score_threshold": score_threshold,
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
                            value=float(row[3]),
                            **kwargs,
                        )
                    )
                    metrics[MetricType.Recall].append(
                        Recall(
                            value=float(row[4]),
                            **kwargs,
                        )
                    )
                    metrics[MetricType.F1].append(
                        F1(
                            value=float(row[5]),
                            **kwargs,
                        )
                    )
                    metrics[MetricType.Accuracy].append(
                        Accuracy(
                            value=float(row[6]),
                            **kwargs,
                        )
                    )

        if as_dict:
            return {
                mtype: [metric.to_dict() for metric in mvalues]
                for mtype, mvalues in metrics.items()
            }

        return metrics

    def compute_confusion_matrix(
        self,
        iou_thresholds: list[float] = [0.5, 0.75, 0.9],
        score_thresholds: list[float] = [0.5],
        number_of_examples: int = 0,
        filter_: Filter | None = None,
        as_dict: bool = False,
    ) -> list:
        """
        Computes confusion matrices at various thresholds.

        Parameters
        ----------
        iou_thresholds : list[float]
            A list of IoU thresholds to compute metrics over.
        score_thresholds : list[float]
            A list of score thresholds to compute metrics over.
        number_of_examples : int, default=0
            Maximum number of annotation examples to return in ConfusionMatrix.
        filter_ : Filter, optional
            An optional filter object.
        as_dict : bool, default=False
            An option to return metrics as dictionaries.

        Returns
        -------
        list[ConfusionMatrix] | list[dict]
            List of confusion matrices per threshold pair.
        """

        detailed_pairs = self._detailed_pairs
        label_metadata = self._label_metadata
        if filter_ is not None:
            detailed_pairs = detailed_pairs[filter_.detailed_indices]
            label_metadata = filter_.label_metadata

        if detailed_pairs.size == 0:
            return list()

        (
            confusion_matrix,
            hallucinations,
            missing_predictions,
        ) = compute_confusion_matrix(
            data=detailed_pairs,
            label_metadata=label_metadata,
            iou_thresholds=np.array(iou_thresholds),
            score_thresholds=np.array(score_thresholds),
            n_examples=number_of_examples,
        )

        n_ious, n_scores, n_labels, _, _ = confusion_matrix.shape
        matrices = [
            ConfusionMatrix(
                iou_threshold=iou_thresholds[iou_idx],
                score_threshold=score_thresholds[score_idx],
                number_of_examples=number_of_examples,
                confusion_matrix=self._unpack_confusion_matrix(
                    confusion_matrix=confusion_matrix[
                        iou_idx, score_idx, :, :, :
                    ],
                    number_of_labels=n_labels,
                    number_of_examples=number_of_examples,
                ),
                hallucinations=self._unpack_hallucinations(
                    hallucinations=hallucinations[iou_idx, score_idx, :, :],
                    number_of_labels=n_labels,
                    number_of_examples=number_of_examples,
                ),
                missing_predictions=self._unpack_missing_predictions(
                    missing_predictions=missing_predictions[
                        iou_idx, score_idx, :, :
                    ],
                    number_of_labels=n_labels,
                    number_of_examples=number_of_examples,
                ),
            )
            for iou_idx in range(n_ious)
            for score_idx in range(n_scores)
        ]

        if as_dict:
            return [m.to_dict() for m in matrices]
        return matrices

    def evaluate(
        self,
        iou_thresholds: list[float] = [0.5, 0.75, 0.9],
        score_thresholds: list[float] = [0.5],
        number_of_examples: int = 0,
        filter_: Filter | None = None,
        as_dict: bool = False,
    ) -> dict[MetricType, list]:
        """
        Computes all avaiable metrics.

        Parameters
        ----------
        iou_thresholds : list[float]
            A list of IoU thresholds to compute metrics over.
        score_thresholds : list[float]
            A list of score thresholds to compute metrics over.
        number_of_examples : int, default=0
            Maximum number of annotation examples to return in ConfusionMatrix.
        filter_ : Filter, optional
            An optional filter object.
        as_dict : bool, default=False
            An option to return metrics as dictionaries.

        Returns
        -------
        dict[MetricType, list]
            A dictionary mapping metric type to a list of metrics.
        """
        results = self.compute_precision_recall(
            iou_thresholds=iou_thresholds,
            score_thresholds=score_thresholds,
            filter_=filter_,
            as_dict=as_dict,
        )
        results[MetricType.ConfusionMatrix] = self.compute_confusion_matrix(
            iou_thresholds=iou_thresholds,
            score_thresholds=score_thresholds,
            number_of_examples=number_of_examples,
            filter_=filter_,
            as_dict=as_dict,
        )
        return results


class DataLoader:
    """
    Object Detection DataLoader
    """

    def __init__(self):
        self._evaluator = Evaluator()
        self.pairs: list[NDArray[np.float64]] = list()
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
            The label associated with the annotation.

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

    def _compute_ious_and_cache_pairs(
        self,
        uid_index: int,
        groundtruths: list,
        predictions: list,
        annotation_type: Type[BoundingBox] | Type[Polygon] | Type[Bitmask],
    ) -> None:
        """
        Compute IOUs between groundtruths and preditions before storing as pairs.

        Parameters
        ----------
        uid_index: int
            The index of the detection.
        groundtruths: list
            A list of groundtruths.
        predictions: list
            A list of predictions.
        annotation_type: type[BoundingBox] | type[Polygon] | type[Bitmask]
            The type of annotation to compute IOUs for.
        """

        pairs = list()
        n_predictions = len(predictions)
        n_groundtruths = len(groundtruths)

        all_pairs = np.array(
            [
                np.array([gann, pann])
                for _, _, _, pann in predictions
                for _, _, gann in groundtruths
            ]
        )

        match annotation_type:
            case annotation.BoundingBox:
                ious = compute_bbox_iou(all_pairs)
            case annotation.Polygon:
                ious = compute_polygon_iou(all_pairs)
            case annotation.Bitmask:
                ious = compute_bitmask_iou(all_pairs)
            case _:
                raise ValueError(
                    f"Invalid annotation type `{annotation_type}`."
                )

        ious = ious.reshape(n_predictions, n_groundtruths)
        predictions_with_iou_of_zero = np.where((ious < 1e-9).all(axis=1))[0]
        groundtruths_with_iou_of_zero = np.where((ious < 1e-9).all(axis=0))[0]

        pairs.extend(
            [
                np.array(
                    [
                        float(uid_index),
                        float(gidx),
                        float(pidx),
                        ious[pidx, gidx],
                        float(glabel),
                        float(plabel),
                        float(score),
                    ]
                )
                for pidx, plabel, score, _ in predictions
                for gidx, glabel, _ in groundtruths
                if ious[pidx, gidx] >= 1e-9
            ]
        )
        pairs.extend(
            [
                np.array(
                    [
                        float(uid_index),
                        -1.0,
                        float(predictions[index][0]),
                        0.0,
                        -1.0,
                        float(predictions[index][1]),
                        float(predictions[index][2]),
                    ]
                )
                for index in predictions_with_iou_of_zero
            ]
        )
        pairs.extend(
            [
                np.array(
                    [
                        float(uid_index),
                        float(groundtruths[index][0]),
                        -1.0,
                        0.0,
                        float(groundtruths[index][1]),
                        -1.0,
                        -1.0,
                    ]
                )
                for index in groundtruths_with_iou_of_zero
            ]
        )

        self.pairs.append(np.array(pairs))

    def _add_data(
        self,
        detections: list[Detection],
        annotation_type: type[Bitmask] | type[BoundingBox] | type[Polygon],
        show_progress: bool = False,
    ):
        """
        Adds detections to the cache.

        Parameters
        ----------
        detections : list[Detection]
            A list of Detection objects.
        annotation_type : type[Bitmask] | type[BoundingBox] | type[Polygon]
            The annotation type to process.
        show_progress : bool, default=False
            Toggle for tqdm progress bar.
        """
        disable_tqdm = not show_progress
        for detection in tqdm(detections, disable=disable_tqdm):

            # update metadata
            self._evaluator.n_datums += 1
            self._evaluator.n_groundtruths += len(detection.groundtruths)
            self._evaluator.n_predictions += len(detection.predictions)

            # update datum uid index
            uid_index = self._add_datum(uid=detection.uid)

            # initialize bounding box examples
            self._evaluator.groundtruth_examples[uid_index] = np.zeros(
                (len(detection.groundtruths), 4), dtype=np.float16
            )
            self._evaluator.prediction_examples[uid_index] = np.zeros(
                (len(detection.predictions), 4), dtype=np.float16
            )

            # cache labels and annotations
            groundtruths = list()
            predictions = list()

            for gidx, gann in enumerate(detection.groundtruths):
                if not isinstance(gann, annotation_type):
                    raise ValueError(
                        f"Expected {annotation_type}, but annotation is of type {type(gann)}."
                    )

                self._evaluator.groundtruth_examples[uid_index][
                    gidx
                ] = gann.extrema
                for glabel in gann.labels:
                    label_idx = self._add_label(glabel)
                    self.groundtruth_count[label_idx][uid_index] += 1
                    groundtruths.append(
                        (
                            gidx,
                            label_idx,
                            gann.annotation,
                        )
                    )

            for pidx, pann in enumerate(detection.predictions):
                if not isinstance(pann, annotation_type):
                    raise ValueError(
                        f"Expected {annotation_type}, but annotation is of type {type(pann)}."
                    )

                self._evaluator.prediction_examples[uid_index][
                    pidx
                ] = pann.extrema
                for plabel, pscore in zip(pann.labels, pann.scores):
                    label_idx = self._add_label(plabel)
                    self.prediction_count[label_idx][uid_index] += 1
                    predictions.append(
                        (
                            pidx,
                            label_idx,
                            pscore,
                            pann.annotation,
                        )
                    )

            self._compute_ious_and_cache_pairs(
                uid_index=uid_index,
                groundtruths=groundtruths,
                predictions=predictions,
                annotation_type=annotation_type,
            )

    def add_bounding_boxes(
        self,
        detections: list[Detection],
        show_progress: bool = False,
    ):
        """
        Adds bounding box detections to the cache.

        Parameters
        ----------
        detections : list[Detection]
            A list of Detection objects.
        show_progress : bool, default=False
            Toggle for tqdm progress bar.
        """
        return self._add_data(
            detections=detections,
            show_progress=show_progress,
            annotation_type=BoundingBox,
        )

    def add_polygons(
        self,
        detections: list[Detection],
        show_progress: bool = False,
    ):
        """
        Adds polygon detections to the cache.

        Parameters
        ----------
        detections : list[Detection]
            A list of Detection objects.
        show_progress : bool, default=False
            Toggle for tqdm progress bar.
        """
        return self._add_data(
            detections=detections,
            show_progress=show_progress,
            annotation_type=Polygon,
        )

    def add_bitmasks(
        self,
        detections: list[Detection],
        show_progress: bool = False,
    ):
        """
        Adds bitmask detections to the cache.

        Parameters
        ----------
        detections : list[Detection]
            A list of Detection objects.
        show_progress : bool, default=False
            Toggle for tqdm progress bar.
        """
        return self._add_data(
            detections=detections,
            show_progress=show_progress,
            annotation_type=Bitmask,
        )

    def finalize(self) -> Evaluator:
        """
        Performs data finalization and some preprocessing steps.

        Returns
        -------
        Evaluator
            A ready-to-use evaluator object.
        """

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
            label_metadata=self._evaluator._label_metadata,
        )

        return self._evaluator
