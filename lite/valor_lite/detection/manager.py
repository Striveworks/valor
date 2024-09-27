from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from shapely.geometry import Polygon as ShapelyPolygon
from tqdm import tqdm
from valor_lite.detection.annotation import (
    Bitmask,
    BoundingBox,
    Detection,
    Polygon,
)
from valor_lite.detection.computation import (
    compute_bbox_iou,
    compute_bitmask_iou,
    compute_detailed_metrics,
    compute_metrics,
    compute_polygon_iou,
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
    CountsWithExamples,
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


def _get_valor_dict_annotation_key(
    annotation_type: type[BoundingBox] | type[Polygon] | type[Bitmask],
) -> str:
    """Get the correct JSON key to extract a given annotation type."""

    if issubclass(annotation_type, BoundingBox):
        return "bounding_box"
    if issubclass(annotation_type, Polygon):
        return "polygon"
    else:
        return "raster"


def _get_annotation_representation(
    annotation_type: type[BoundingBox] | type[Polygon] | type[Bitmask],
) -> str:
    """Get the correct representation of an annotation object."""

    representation = (
        "extrema"
        if issubclass(annotation_type, BoundingBox)
        else ("mask" if issubclass(annotation_type, Bitmask) else "shape")
    )

    return representation


def _get_annotation_representation_from_valor_dict(
    data: list,
    annotation_type: type[BoundingBox] | type[Polygon] | type[Bitmask],
) -> tuple[float, float, float, float] | ShapelyPolygon | NDArray[np.bool_]:
    """Get the correct representation of an annotation object from a valor dictionary."""

    if issubclass(annotation_type, BoundingBox):
        x = [point[0] for shape in data for point in shape]
        y = [point[1] for shape in data for point in shape]
        return (min(x), max(x), min(y), max(y))
    if issubclass(annotation_type, Polygon):
        return ShapelyPolygon(data)
    else:
        return np.array(data)


def _get_annotation_data(
    keyed_groundtruths: dict,
    keyed_predictions: dict,
    annotation_type: type[BoundingBox] | type[Polygon] | type[Bitmask] | None,
    key=int,
) -> np.ndarray:
    """Create an array of annotation pairs for use when calculating IOU. Needed because we unpack bounding box representations, but not bitmask or polygon representations."""
    if annotation_type == BoundingBox:
        return np.array(
            [
                np.array([*gextrema, *pextrema])
                for _, _, _, pextrema in keyed_predictions[key]
                for _, _, gextrema in keyed_groundtruths[key]
            ]
        )
    else:
        return np.array(
            [
                np.array([groundtruth_obj, prediction_obj])
                for _, _, _, prediction_obj in keyed_predictions[key]
                for _, _, groundtruth_obj in keyed_groundtruths[key]
            ]
        )


def compute_iou(
    data: NDArray[np.floating],
    annotation_type: type[BoundingBox] | type[Polygon] | type[Bitmask],
) -> NDArray[np.floating]:
    """
    Computes intersection-over-union (IoU) calculations for various annotation types.

    Parameters
    ----------
    data : NDArray[np.floating]
        A sorted array of bounding box, bitmask, or polygon pairs.
    annotation_type: type[BoundingBox] | type[Polygon] | type[Bitmask]
        The type of annotation contained in the data.


    Returns
    -------
    NDArray[np.floating]
        Computed IoU's.
    """

    if annotation_type == BoundingBox:
        return compute_bbox_iou(data=data)
    elif annotation_type == Bitmask:
        return compute_bitmask_iou(data=data)
    else:
        return compute_polygon_iou(data=data)


@dataclass
class Filter:
    indices: NDArray[np.int32]
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
        self.label_to_index: dict[tuple[str, str], int] = dict()
        self.index_to_label: dict[int, tuple[str, str]] = dict()

        # label key reference
        self.index_to_label_key: dict[int, str] = dict()
        self.label_key_to_index: dict[str, int] = dict()
        self.label_index_to_label_key_index: dict[int, int] = dict()

        # computation caches
        self._detailed_pairs: NDArray[np.floating] = np.array([])
        self._ranked_pairs: NDArray[np.floating] = np.array([])
        self._label_metadata: NDArray[np.int32] = np.array([])
        self._label_metadata_per_datum: NDArray[np.int32] = np.array([])

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
        Creates a filter that can be passed to an evaluation.

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
        )

    def evaluate(
        self,
        metrics_to_return: list[MetricType] = MetricType.base_metrics(),
        iou_thresholds: list[float] = [0.5, 0.75, 0.9],
        score_thresholds: list[float] = [0.5],
        number_of_examples: int = 0,
        filter_: Filter | None = None,
    ) -> dict[MetricType, list]:
        """
        Performs an evaluation and returns metrics.

        Parameters
        ----------
        metrics_to_return : list[MetricType]
            A list of metrics to return in the results.
        iou_thresholds : list[float]
            A list of IoU thresholds to compute metrics over.
        score_thresholds : list[float]
            A list of score thresholds to compute metrics over.
        number_of_examples : int, default=0
            Number of annotation examples to return in CountsWithExamples.
        filter_ : Filter, optional
            An optional filter object.

        Returns
        -------
        dict[MetricType, list]
            A dictionary mapping MetricType enumerations to lists of computed metrics.
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
            label_metadata=label_metadata,
            iou_thresholds=np.array(iou_thresholds),
            score_thresholds=np.array(score_thresholds),
        )

        metrics = defaultdict(list)

        metrics[MetricType.AP] = [
            AP(
                value=average_precision[iou_idx][label_idx],
                iou_threshold=iou_thresholds[iou_idx],
                label=self.index_to_label[label_idx],
            )
            for iou_idx in range(average_precision.shape[0])
            for label_idx in range(average_precision.shape[1])
            if int(label_metadata[label_idx][0]) > 0
        ]

        metrics[MetricType.mAP] = [
            mAP(
                value=mean_average_precision[iou_idx][label_key_idx],
                iou_threshold=iou_thresholds[iou_idx],
                label_key=self.index_to_label_key[label_key_idx],
            )
            for iou_idx in range(mean_average_precision.shape[0])
            for label_key_idx in range(mean_average_precision.shape[1])
        ]

        metrics[MetricType.APAveragedOverIOUs] = [
            APAveragedOverIOUs(
                value=average_precision_average_over_ious[label_idx],
                iou_thresholds=iou_thresholds,
                label=self.index_to_label[label_idx],
            )
            for label_idx in range(self.n_labels)
            if int(label_metadata[label_idx][0]) > 0
        ]

        metrics[MetricType.mAPAveragedOverIOUs] = [
            mAPAveragedOverIOUs(
                value=mean_average_precision_average_over_ious[label_key_idx],
                iou_thresholds=iou_thresholds,
                label_key=self.index_to_label_key[label_key_idx],
            )
            for label_key_idx in range(
                mean_average_precision_average_over_ious.shape[0]
            )
        ]

        metrics[MetricType.AR] = [
            AR(
                value=average_recall[score_idx][label_idx],
                iou_thresholds=iou_thresholds,
                score_threshold=score_thresholds[score_idx],
                label=self.index_to_label[label_idx],
            )
            for score_idx in range(average_recall.shape[0])
            for label_idx in range(average_recall.shape[1])
            if int(label_metadata[label_idx][0]) > 0
        ]

        metrics[MetricType.mAR] = [
            mAR(
                value=mean_average_recall[score_idx][label_key_idx],
                iou_thresholds=iou_thresholds,
                score_threshold=score_thresholds[score_idx],
                label_key=self.index_to_label_key[label_key_idx],
            )
            for score_idx in range(mean_average_recall.shape[0])
            for label_key_idx in range(mean_average_recall.shape[1])
        ]

        metrics[MetricType.ARAveragedOverScores] = [
            ARAveragedOverScores(
                value=average_recall_averaged_over_scores[label_idx],
                score_thresholds=score_thresholds,
                iou_thresholds=iou_thresholds,
                label=self.index_to_label[label_idx],
            )
            for label_idx in range(self.n_labels)
            if int(label_metadata[label_idx][0]) > 0
        ]

        metrics[MetricType.mARAveragedOverScores] = [
            mARAveragedOverScores(
                value=mean_average_recall_averaged_over_scores[label_key_idx],
                score_thresholds=score_thresholds,
                iou_thresholds=iou_thresholds,
                label_key=self.index_to_label_key[label_key_idx],
            )
            for label_key_idx in range(
                mean_average_recall_averaged_over_scores.shape[0]
            )
        ]

        metrics[MetricType.PrecisionRecallCurve] = [
            PrecisionRecallCurve(
                precision=list(pr_curves[iou_idx][label_idx]),
                iou_threshold=iou_threshold,
                label=label,
            )
            for iou_idx, iou_threshold in enumerate(iou_thresholds)
            for label_idx, label in self.index_to_label.items()
            if int(label_metadata[label_idx][0]) > 0
        ]

        for label_idx, label in self.index_to_label.items():
            for score_idx, score_threshold in enumerate(score_thresholds):
                for iou_idx, iou_threshold in enumerate(iou_thresholds):

                    if label_metadata[label_idx, 0] == 0:
                        continue

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

        if MetricType.CountsWithExamples in metrics_to_return:
            metrics[MetricType.CountsWithExamples] = self._compute_detailed_metrics(
                iou_thresholds=iou_thresholds,
                score_thresholds=score_thresholds,
                n_samples=number_of_examples,
            )

        for metric in set(metrics.keys()):
            if metric not in metrics_to_return:
                del metrics[metric]

        return metrics

    def _compute_detailed_metrics(
        self,
        iou_thresholds: list[float] = [0.5],
        score_thresholds: list[float] = [
            score / 10.0 for score in range(1, 11)
        ],
        n_samples: int = 0,
    ) -> list[CountsWithExamples]:
        """
        Computes detailed counting metrics.

        Parameters
        ----------
        iou_thresholds : list[float], default=[0.5]
            List of IoU thresholds to compute metrics for.
        score_thresholds : list[float], default=[0.1,0.2,...,1.0]
            List of confidence thresholds to compute metrics for.
        n_samples : int, default=0
            Number of datum samples to return per metric.

        Returns
        -------
        list[list[CountsWithExamples]]
            Outer list is indexed by label, inner list is by IoU.
        """

        if self._detailed_pairs.size == 0:
            return list()

        (
            confusion_matrix,
            hallucinations,
            missing_predictions,
        ) = compute_detailed_metrics(
            self._detailed_pairs,
            label_metadata=self._label_metadata,
            iou_thresholds=np.array(iou_thresholds),
            score_thresholds=np.array(score_thresholds),
            n_samples=n_samples,
        )

        def _unpack_confusion_matrix(
            iou_idx: int,
            score_idx: int,
            number_of_labels: int,
            number_of_examples: int,
        ) -> dict[str, dict[str, dict[str, int | list[dict[str, str | float | tuple[float, float, float, float]]]]]]:
            """
            Unpacks metric examples from computation.
            """
            return {
                self.index_to_label[gt_label_idx][0]: {
                    self.index_to_label[pd_label_idx][0]: {
                        "count": int(
                            confusion_matrix[
                                iou_idx,
                                score_idx,
                                gt_label_idx,
                                pd_label_idx,
                                0
                            ]
                        ),
                        "examples": [
                            {
                                "datum": self.index_to_uid[
                                    int(
                                        confusion_matrix[
                                            iou_idx,
                                            score_idx,
                                            gt_label_idx,
                                            pd_label_idx,
                                            example_idx * 4 + 1,
                                        ]
                                    )
                                ],
                                "groundtruth": tuple(
                                    *self.groundtruth_examples[
                                        int(
                                            confusion_matrix[
                                                iou_idx,
                                                score_idx,
                                                gt_label_idx,
                                                pd_label_idx,
                                                example_idx * 4 + 2,
                                            ]
                                        )
                                    ].tolist()
                                ),
                                "prediction": tuple(
                                    *self.prediction_examples[
                                        int(
                                            confusion_matrix[
                                                iou_idx,
                                                score_idx,
                                                gt_label_idx,
                                                pd_label_idx,
                                                example_idx * 4 + 3,
                                            ]
                                        )
                                    ].tolist()
                                ),
                                "score": float(
                                    confusion_matrix[
                                        iou_idx,
                                        score_idx,
                                        gt_label_idx,
                                        pd_label_idx,
                                        example_idx * 4 + 4,
                                    ]
                                )
                            }
                            for example_idx in range(number_of_examples)
                        ]
                    }
                    for pd_label_idx in range(number_of_labels)
                }
                for gt_label_idx in range(number_of_labels)
            }

        def _unpack_hallucinations_and_missing_preidctions(
            iou_idx: int,
            score_idx: int,
            number_of_labels: int,
            number_of_examples: int,
        ) -> dict[str, dict[str, dict[str, int | list[dict[str, str | float | tuple[float, float, float, float]]]]]]:
            """
            Unpacks metric examples from computation.
            """
            return {
                self.index_to_label[gt_label_idx][0]: {
                    self.index_to_label[pd_label_idx][0]: {
                        "count": int(
                            confusion_matrix[
                                iou_idx,
                                score_idx,
                                gt_label_idx,
                                pd_label_idx,
                                0
                            ]
                        ),
                        "examples": [
                            {
                                "datum": self.index_to_uid[
                                    int(
                                        confusion_matrix[
                                            iou_idx,
                                            score_idx,
                                            gt_label_idx,
                                            pd_label_idx,
                                            example_idx * 4 + 1,
                                        ]
                                    )
                                ],
                                "groundtruth": tuple(
                                    *self.groundtruth_examples[
                                        int(
                                            confusion_matrix[
                                                iou_idx,
                                                score_idx,
                                                gt_label_idx,
                                                pd_label_idx,
                                                example_idx * 4 + 2,
                                            ]
                                        )
                                    ].tolist()
                                ),
                                "prediction": tuple(
                                    *self.prediction_examples[
                                        int(
                                            confusion_matrix[
                                                iou_idx,
                                                score_idx,
                                                gt_label_idx,
                                                pd_label_idx,
                                                example_idx * 4 + 3,
                                            ]
                                        )
                                    ].tolist()
                                ),
                                "score": float(
                                    confusion_matrix[
                                        iou_idx,
                                        score_idx,
                                        gt_label_idx,
                                        pd_label_idx,
                                        example_idx * 4 + 4,
                                    ]
                                )
                            }
                            for example_idx in range(number_of_examples)
                        ]
                    }
                    for pd_label_idx in range(number_of_labels)
                }
                for gt_label_idx in range(number_of_labels)
            }


            data =

            results = list()
            for score_idx in range(n_scores):
                examples = list()
                for example_idx in range(n_samples):
                    datum_idx = metrics[
                        iou_idx,
                        score_idx,
                        label_idx,
                        type_idx + example_idx * 2,
                    ]
                    annotation_idx = metrics[
                        iou_idx,
                        score_idx,
                        label_idx,
                        type_idx + example_idx * 2 + 1,
                    ]
                    if datum_idx >= 0:
                        examples.append(
                            (
                                self.index_to_uid[datum_idx],
                                tuple(
                                    example_source[datum_idx][
                                        annotation_idx
                                    ].tolist()
                                ),
                            )
                        )
                results.append(examples)

            return results

        n_ious, n_scores, n_labels, _ = metrics.shape
        return [
            CountsWithExamples(
                iou_threshold=iou_thresholds[iou_idx],
                label=self.index_to_label[label_idx],
                score_thresholds=score_thresholds,
                confusion_matrix=
            )
            for iou_idx in range(n_ious)
            for score_idx in range(n_scores)
        ]


class DataLoader:
    """
    Object Detection DataLoader
    """

    def __init__(self):
        self._evaluator = Evaluator()
        self.pairs: list[NDArray[np.floating]] = list()
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

    def _compute_ious_and_cache_pairs(
        self,
        uid_index: int,
        keyed_groundtruths: dict,
        keyed_predictions: dict,
        annotation_type: type[BoundingBox] | type[Polygon] | type[Bitmask],
    ) -> None:
        """
        Compute IOUs between groundtruths and preditions before storing as pairs.

        Parameters
        ----------
        uid_index: int
            The index of the detection.
        keyed_groundtruths: dict
            A dictionary of groundtruths.
        keyed_predictions: dict
            A dictionary of predictions.
        annotation_type: type[BoundingBox] | type[Polygon] | type[Bitmask]
            The type of annotation to compute IOUs for.
        """
        gt_keys = set(keyed_groundtruths.keys())
        pd_keys = set(keyed_predictions.keys())
        joint_keys = gt_keys.intersection(pd_keys)
        gt_unique_keys = gt_keys - pd_keys
        pd_unique_keys = pd_keys - gt_keys

        pairs = list()
        for key in joint_keys:
            n_predictions = len(keyed_predictions[key])
            n_groundtruths = len(keyed_groundtruths[key])
            data = _get_annotation_data(
                keyed_groundtruths=keyed_groundtruths,
                keyed_predictions=keyed_predictions,
                key=key,
                annotation_type=annotation_type,
            )
            ious = compute_iou(data=data, annotation_type=annotation_type)
            mask_nonzero_iou = (ious > 1e-9).reshape(
                (n_predictions, n_groundtruths)
            )
            mask_ious_halluc = ~(mask_nonzero_iou.any(axis=1))
            mask_ious_misprd = ~(mask_nonzero_iou.any(axis=0))

            pairs.extend(
                [
                    np.array(
                        [
                            float(uid_index),
                            float(gidx),
                            float(pidx),
                            ious[pidx * len(keyed_groundtruths[key]) + gidx],
                            float(glabel),
                            float(plabel),
                            float(score),
                        ]
                    )
                    for pidx, plabel, score, _ in keyed_predictions[key]
                    for gidx, glabel, _ in keyed_groundtruths[key]
                    if ious[pidx * len(keyed_groundtruths[key]) + gidx] > 1e-9
                ]
            )
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
                    if mask_ious_halluc[pidx]
                ]
            )
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
                    if mask_ious_misprd[gidx]
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
            keyed_groundtruths = defaultdict(list)
            keyed_predictions = defaultdict(list)

            representation_property = _get_annotation_representation(
                annotation_type=annotation_type
            )

            for gidx, gann in enumerate(detection.groundtruths):
                if not isinstance(gann, annotation_type):
                    raise ValueError(
                        f"Expected {annotation_type}, but annotation is of type {type(gann)}."
                    )

                if isinstance(gann, BoundingBox):
                    self._evaluator.groundtruth_examples[uid_index][
                        gidx
                    ] = getattr(gann, representation_property)
                else:
                    converted_box = gann.to_box()
                    self._evaluator.groundtruth_examples[uid_index][gidx] = (
                        getattr(converted_box, "extrema")
                        if converted_box is not None
                        else None
                    )
                for glabel in gann.labels:
                    label_idx, label_key_idx = self._add_label(glabel)
                    self.groundtruth_count[label_idx][uid_index] += 1
                    representation = getattr(gann, representation_property)
                    keyed_groundtruths[label_key_idx].append(
                        (
                            gidx,
                            label_idx,
                            representation,
                        )
                    )

            for pidx, pann in enumerate(detection.predictions):
                if not isinstance(pann, annotation_type):
                    raise ValueError(
                        f"Expected {annotation_type}, but annotation is of type {type(pann)}."
                    )

                if isinstance(pann, BoundingBox):
                    self._evaluator.prediction_examples[uid_index][
                        pidx
                    ] = getattr(pann, representation_property)
                else:
                    converted_box = pann.to_box()
                    self._evaluator.prediction_examples[uid_index][pidx] = (
                        getattr(converted_box, "extrema")
                        if converted_box is not None
                        else None
                    )
                for plabel, pscore in zip(pann.labels, pann.scores):
                    label_idx, label_key_idx = self._add_label(plabel)
                    self.prediction_count[label_idx][uid_index] += 1
                    representation = representation = getattr(
                        pann, representation_property
                    )
                    keyed_predictions[label_key_idx].append(
                        (
                            pidx,
                            label_idx,
                            pscore,
                            representation,
                        )
                    )

            self._compute_ious_and_cache_pairs(
                uid_index=uid_index,
                keyed_groundtruths=keyed_groundtruths,
                keyed_predictions=keyed_predictions,
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

    def _add_data_from_valor_dict(
        self,
        detections: list[tuple[dict, dict]],
        annotation_type: type[Bitmask] | type[BoundingBox] | type[Polygon],
        show_progress: bool = False,
    ):
        """
        Adds Valor-format detections to the cache.

        Parameters
        ----------
        detections : list[tuple[dict, dict]]
            A list of groundtruth, prediction pairs in Valor-format dictionaries.
        annotation_type : type[Bitmask] | type[BoundingBox] | type[Polygon]
            The annotation type to process.
        show_progress : bool, default=False
            Toggle for tqdm progress bar.
        """

        disable_tqdm = not show_progress
        for groundtruth, prediction in tqdm(detections, disable=disable_tqdm):
            # update metadata
            self._evaluator.n_datums += 1
            self._evaluator.n_groundtruths += len(groundtruth["annotations"])
            self._evaluator.n_predictions += len(prediction["annotations"])

            # update datum uid index
            uid_index = self._add_datum(uid=groundtruth["datum"]["uid"])

            # initialize bounding box examples
            self._evaluator.groundtruth_examples[uid_index] = np.zeros(
                (len(groundtruth["annotations"]), 4), dtype=np.float16
            )
            self._evaluator.prediction_examples[uid_index] = np.zeros(
                (len(prediction["annotations"]), 4), dtype=np.float16
            )

            # cache labels and annotations
            keyed_groundtruths = defaultdict(list)
            keyed_predictions = defaultdict(list)

            annotation_key = _get_valor_dict_annotation_key(
                annotation_type=annotation_type
            )
            invalid_keys = list(
                filter(
                    lambda x: x != annotation_key,
                    ["bounding_box", "raster", "polygon"],
                )
            )

            for gidx, gann in enumerate(groundtruth["annotations"]):
                if (gann[annotation_key] is None) or any(
                    [gann[k] is not None for k in invalid_keys]
                ):
                    raise ValueError(
                        f"Input JSON doesn't contain {annotation_type} data, or contains data for multiple annotation types."
                    )
                if annotation_type == BoundingBox:
                    self._evaluator.groundtruth_examples[uid_index][
                        gidx
                    ] = np.array(
                        _get_annotation_representation_from_valor_dict(
                            gann[annotation_key],
                            annotation_type=annotation_type,
                        ),
                    )

                for valor_label in gann["labels"]:
                    glabel = (valor_label["key"], valor_label["value"])
                    label_idx, label_key_idx = self._add_label(glabel)
                    self.groundtruth_count[label_idx][uid_index] += 1
                    keyed_groundtruths[label_key_idx].append(
                        (
                            gidx,
                            label_idx,
                            _get_annotation_representation_from_valor_dict(
                                gann[annotation_key],
                                annotation_type=annotation_type,
                            ),
                        )
                    )
            for pidx, pann in enumerate(prediction["annotations"]):
                if (pann[annotation_key] is None) or any(
                    [pann[k] is not None for k in invalid_keys]
                ):
                    raise ValueError(
                        f"Input JSON doesn't contain {annotation_type} data, or contains data for multiple annotation types."
                    )

                if annotation_type == BoundingBox:
                    self._evaluator.prediction_examples[uid_index][
                        pidx
                    ] = np.array(
                        _get_annotation_representation_from_valor_dict(
                            pann[annotation_key],
                            annotation_type=annotation_type,
                        )
                    )
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
                            _get_annotation_representation_from_valor_dict(
                                pann[annotation_key],
                                annotation_type=annotation_type,
                            ),
                        )
                    )

            self._compute_ious_and_cache_pairs(
                uid_index=uid_index,
                keyed_groundtruths=keyed_groundtruths,
                keyed_predictions=keyed_predictions,
                annotation_type=annotation_type,
            )

    def add_bounding_boxes_from_valor_dict(
        self,
        detections: list[tuple[dict, dict]],
        show_progress: bool = False,
    ):
        """
        Adds Valor-format bounding box detections to the cache.

        Parameters
        ----------
        detections : list[tuple[dict, dict]]
            A list of groundtruth, prediction pairs in Valor-format dictionaries.
        show_progress : bool, default=False
            Toggle for tqdm progress bar.
        """
        return self._add_data_from_valor_dict(
            detections=detections,
            show_progress=show_progress,
            annotation_type=BoundingBox,
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
            label_metadata=self._evaluator._label_metadata,
        )

        return self._evaluator
