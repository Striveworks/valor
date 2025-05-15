from collections import defaultdict
from dataclasses import dataclass

import numpy as np
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
    compute_label_metadata,
    compute_polygon_iou,
    compute_precion_recall,
    compute_ranked_pairs,
)
from valor_lite.object_detection.metric import Metric, MetricType
from valor_lite.object_detection.utilities import (
    unpack_confusion_matrix_into_metric_list,
    unpack_precision_recall_into_metric_lists,
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


class Evaluator:
    """
    Object Detection Evaluator
    """

    def __init__(self):

        # external reference
        self.datum_id_to_index: dict[str, int] = dict()
        self.groundtruth_id_to_index: dict[str, int] = dict()
        self.prediction_id_to_index: dict[str, int] = dict()

        self.index_to_datum_id: list[str] = list()
        self.index_to_groundtruth_id: list[str] = list()
        self.index_to_prediction_id: list[str] = list()

        self.label_to_index: dict[str, int] = dict()
        self.index_to_label: list[str] = list()

        # cache
        self._cache: NDArray[np.float64] = np.array([])
        self._label_metadata: NDArray[np.int32] = np.array([])

        # optional filter cache
        self._filtered_cache: NDArray[np.float64] | None = None
        self._filtered_label_metadata: NDArray[np.int32] | None = None

    @property
    def is_filtered(self) -> bool:
        return self._filtered_cache is not None

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
            self._filtered_cache
            if self._filtered_cache is not None
            else self._cache
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
        unique_ids = np.unique(self.detailed_pairs[:, (1, 4)], axis=0)
        return (unique_ids[:, 0] >= 0.0).sum()

    @property
    def n_predictions(self) -> int:
        """Returns the number of prediction annotations."""
        unique_ids = np.unique(self.detailed_pairs[:, (2, 5)], axis=0)
        return (unique_ids[:, 0] >= 0.0).sum()

    @property
    def ignored_prediction_labels(self) -> list[str]:
        """
        Prediction labels that are not present in the ground truth set.
        """
        label_metadata = self.label_metadata
        glabels = set(np.where(label_metadata[:, 0] > 0)[0])
        plabels = set(np.where(label_metadata[:, 1] > 0)[0])
        return [
            self.index_to_label[label_id] for label_id in (plabels - glabels)
        ]

    @property
    def missing_prediction_labels(self) -> list[str]:
        """
        Ground truth labels that are not present in the prediction set.
        """
        label_metadata = self.label_metadata
        glabels = set(np.where(label_metadata[:, 0] > 0)[0])
        plabels = set(np.where(label_metadata[:, 1] > 0)[0])
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

    def apply_filter(
        self,
        datum_ids: list[str] | None = None,
        groundtruth_ids: list[str] | None = None,
        prediction_ids: list[str] | None = None,
        labels: list[str] | None = None,
    ):
        """
        Apply a filter on the evaluator.

        Can be reset by calling 'clear_filter'.

        Parameters
        ----------
        datum_uids : list[str], optional
            An optional list of string uids representing datums.
        groundtruth_ids : list[str], optional
            An optional list of string uids representing ground truth annotations.
        prediction_ids : list[str], optional
            An optional list of string uids representing prediction annotations.
        labels : list[str], optional
            An optional list of labels.
        """
        # mask by datums and annotations
        mask_detailed = np.ones(self._cache.shape[0], dtype=np.bool_)
        if datum_ids is not None:
            datum_ids_array = np.array(
                [self.datum_id_to_index[uid] for uid in datum_ids],
                dtype=np.int32,
            )
            mask_detailed[
                ~np.isin(self._cache[:, 0].astype(int), datum_ids_array)
            ] = False
        if groundtruth_ids is not None:
            groundtruth_ids_array = np.array(
                [self.groundtruth_id_to_index[uid] for uid in groundtruth_ids],
                dtype=np.int32,
            )
            mask_detailed[
                ~np.isin(self._cache[:, 1].astype(int), groundtruth_ids_array)
            ] = False
        if prediction_ids is not None:
            prediction_ids_array = np.array(
                [self.prediction_id_to_index[uid] for uid in prediction_ids],
                dtype=np.int32,
            )
            mask_detailed[
                ~np.isin(self._cache[:, 2].astype(int), prediction_ids_array)
            ] = False

        # mask by labels
        mask_invalid_groundtruths = np.zeros_like(mask_detailed)
        mask_invalid_predictions = np.zeros_like(mask_detailed)
        if labels is not None:
            labels_array = np.array(
                [self.label_to_index[label] for label in labels] + [-1]
            )
            mask_invalid_groundtruths[
                ~np.isin(self._cache[:, 4].astype(int), labels_array)
            ] = True
            mask_invalid_predictions[
                ~np.isin(self._cache[:, 5].astype(int), labels_array)
            ] = True

        # filter cache
        self._filtered_cache = self._cache.copy()
        if mask_invalid_groundtruths.any():
            invalid_groundtruth_indices = np.where(mask_invalid_groundtruths)[
                0
            ]
            self._filtered_cache[
                invalid_groundtruth_indices[:, None], (1, 3, 4)
            ] = np.array([[-1, 0, -1]])
        if mask_invalid_predictions.any():
            invalid_prediction_indices = np.where(mask_invalid_predictions)[0]
            self._filtered_cache[
                invalid_prediction_indices[:, None], (2, 3, 5, 6)
            ] = np.array([[-1, 0, -1, -1]])
        self._filtered_cache = self._filtered_cache[np.where(mask_detailed)[0]]

        # filter null pairs
        mask_null_pairs = np.all(
            np.isclose(
                self._filtered_cache[:, 1:],
                np.array([-1.0, -1.0, 0.0, -1.0, -1.0, -1.0]),
            ),
            axis=1,
        )
        self._filtered_cache = self._filtered_cache[~mask_null_pairs]

        # filter label metadata
        self._filtered_label_metadata = compute_label_metadata(
            detailed_pairs=self._filtered_cache,
            n_labels=self.n_labels,
        )

    def clear_filter(self):
        """Removes a filter if one exists."""
        self._filtered_cache = None
        self._filtered_label_metadata = None

    def compute_precision_recall(
        self,
        iou_thresholds: list[float],
        score_thresholds: list[float],
    ) -> dict[MetricType, list[Metric]]:
        """
        Computes all metrics except for ConfusionMatrix

        Parameters
        ----------
        iou_thresholds : list[float]
            A list of IOU thresholds to compute metrics over.
        score_thresholds : list[float]
            A list of score thresholds to compute metrics over.

        Returns
        -------
        dict[MetricType, list]
            A dictionary mapping MetricType enumerations to lists of computed metrics.
        """
        ranked_pairs = compute_ranked_pairs(
            detailed_pairs=self.detailed_pairs,
            label_metadata=self.label_metadata,
        )
        results = compute_precion_recall(
            ranked_pairs=ranked_pairs,
            label_metadata=self.label_metadata,
            iou_thresholds=np.array(iou_thresholds),
            score_thresholds=np.array(score_thresholds),
        )
        return unpack_precision_recall_into_metric_lists(
            results=results,
            label_metadata=self.label_metadata,
            iou_thresholds=iou_thresholds,
            score_thresholds=score_thresholds,
            index_to_label=self.index_to_label,
        )

    def compute_confusion_matrix(
        self,
        iou_thresholds: list[float],
        score_thresholds: list[float],
    ) -> list[Metric]:
        """
        Computes confusion matrices at various thresholds.

        Parameters
        ----------
        iou_thresholds : list[float]
            A list of IOU thresholds to compute metrics over.
        score_thresholds : list[float]
            A list of score thresholds to compute metrics over.

        Returns
        -------
        list[Metric]
            List of confusion matrices per threshold pair.
        """
        results = compute_confusion_matrix(
            detailed_pairs=self.detailed_pairs,
            label_metadata=self.label_metadata,
            iou_thresholds=np.array(iou_thresholds),
            score_thresholds=np.array(score_thresholds),
        )
        return unpack_confusion_matrix_into_metric_list(
            results=results,
            detailed_pairs=self.detailed_pairs,
            iou_thresholds=iou_thresholds,
            score_thresholds=score_thresholds,
            index_to_datum_id=self.index_to_datum_id,
            index_to_groundtruth_id=self.index_to_groundtruth_id,
            index_to_prediction_id=self.index_to_prediction_id,
            index_to_label=self.index_to_label,
        )

    def evaluate(
        self,
        iou_thresholds: list[float] = [0.1, 0.5, 0.75],
        score_thresholds: list[float] = [0.5],
    ) -> dict[MetricType, list[Metric]]:
        """
        Computes all available metrics.

        Parameters
        ----------
        iou_thresholds : list[float], default=[0.1, 0.5, 0.75]
            A list of IOU thresholds to compute metrics over.
        score_thresholds : list[float], default=[0.5]
            A list of score thresholds to compute metrics over.

        Returns
        -------
        dict[MetricType, list[Metric]]
            Lists of metrics organized by metric type.
        """
        metrics = self.compute_precision_recall(
            iou_thresholds=iou_thresholds,
            score_thresholds=score_thresholds,
        )
        metrics[MetricType.ConfusionMatrix] = self.compute_confusion_matrix(
            iou_thresholds=iou_thresholds,
            score_thresholds=score_thresholds,
        )
        return metrics


def defaultdict_int():
    return defaultdict(int)


class DataLoader:
    """
    Object Detection DataLoader
    """

    def __init__(self):
        self._evaluator = Evaluator()
        self.pairs: list[NDArray[np.float64]] = list()

    def _add_datum(self, datum_id: str) -> int:
        """
        Helper function for adding a datum to the cache.

        Parameters
        ----------
        datum_id : str
            The datum identifier.

        Returns
        -------
        int
            The datum index.
        """
        if datum_id not in self._evaluator.datum_id_to_index:
            if len(self._evaluator.datum_id_to_index) != len(
                self._evaluator.index_to_datum_id
            ):
                raise RuntimeError("datum cache size mismatch")
            idx = len(self._evaluator.datum_id_to_index)
            self._evaluator.datum_id_to_index[datum_id] = idx
            self._evaluator.index_to_datum_id.append(datum_id)
        return self._evaluator.datum_id_to_index[datum_id]

    def _add_groundtruth(self, annotation_id: str) -> int:
        """
        Helper function for adding a ground truth annotation identifier to the cache.

        Parameters
        ----------
        annotation_id : str
            The ground truth annotation identifier.

        Returns
        -------
        int
            The ground truth annotation index.
        """
        if annotation_id not in self._evaluator.groundtruth_id_to_index:
            if len(self._evaluator.groundtruth_id_to_index) != len(
                self._evaluator.index_to_groundtruth_id
            ):
                raise RuntimeError("ground truth cache size mismatch")
            idx = len(self._evaluator.groundtruth_id_to_index)
            self._evaluator.groundtruth_id_to_index[annotation_id] = idx
            self._evaluator.index_to_groundtruth_id.append(annotation_id)
        return self._evaluator.groundtruth_id_to_index[annotation_id]

    def _add_prediction(self, annotation_id: str) -> int:
        """
        Helper function for adding a prediction annotation identifier to the cache.

        Parameters
        ----------
        annotation_id : str
            The prediction annotation identifier.

        Returns
        -------
        int
            The prediction annotation index.
        """
        if annotation_id not in self._evaluator.prediction_id_to_index:
            if len(self._evaluator.prediction_id_to_index) != len(
                self._evaluator.index_to_prediction_id
            ):
                raise RuntimeError("prediction cache size mismatch")
            idx = len(self._evaluator.prediction_id_to_index)
            self._evaluator.prediction_id_to_index[annotation_id] = idx
            self._evaluator.index_to_prediction_id.append(annotation_id)
        return self._evaluator.prediction_id_to_index[annotation_id]

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
            self._evaluator.index_to_label.append(label)

            label_id += 1

        return self._evaluator.label_to_index[label]

    def _add_data(
        self,
        detections: list[Detection],
        detection_ious: list[NDArray[np.float64]],
        show_progress: bool = False,
    ):
        """
        Adds detections to the cache.

        Parameters
        ----------
        detections : list[Detection]
            A list of Detection objects.
        detection_ious : list[NDArray[np.float64]]
            A list of arrays containing IOUs per detection.
        show_progress : bool, default=False
            Toggle for tqdm progress bar.
        """
        disable_tqdm = not show_progress
        for detection, ious in tqdm(
            zip(detections, detection_ious), disable=disable_tqdm
        ):
            # cache labels and annotation pairs
            pairs = list()
            datum_idx = self._add_datum(detection.uid)
            if detection.groundtruths:
                for gidx, gann in enumerate(detection.groundtruths):
                    groundtruth_idx = self._add_groundtruth(gann.uid)
                    for glabel in gann.labels:
                        glabel_idx = self._add_label(glabel)
                        if (ious[:, gidx] < 1e-9).all():
                            pairs.append(
                                np.array(
                                    [
                                        float(datum_idx),
                                        float(groundtruth_idx),
                                        -1.0,
                                        0.0,
                                        float(glabel_idx),
                                        -1.0,
                                        -1.0,
                                    ]
                                )
                            )
                        else:
                            for pidx, pann in enumerate(detection.predictions):
                                prediction_idx = self._add_prediction(pann.uid)
                                for plabel, pscore in zip(
                                    pann.labels, pann.scores
                                ):
                                    plabel_idx = self._add_label(plabel)
                                    if ious[pidx, gidx] >= 1e-9:
                                        pairs.append(
                                            np.array(
                                                [
                                                    float(datum_idx),
                                                    float(groundtruth_idx),
                                                    float(prediction_idx),
                                                    ious[pidx, gidx],
                                                    float(glabel_idx),
                                                    float(plabel_idx),
                                                    float(pscore),
                                                ]
                                            )
                                        )
                                    else:
                                        pairs.append(
                                            np.array(
                                                [
                                                    float(datum_idx),
                                                    -1.0,
                                                    float(prediction_idx),
                                                    0.0,
                                                    -1.0,
                                                    float(plabel_idx),
                                                    float(pscore),
                                                ]
                                            )
                                        )
            else:
                for pidx, pann in enumerate(detection.predictions):
                    prediction_idx = self._add_prediction(pann.uid)
                    for plabel, pscore in zip(pann.labels, pann.scores):
                        plabel_idx = self._add_label(plabel)
                        pairs.append(
                            np.array(
                                [
                                    float(datum_idx),
                                    -1.0,
                                    float(prediction_idx),
                                    0.0,
                                    -1.0,
                                    float(plabel_idx),
                                    float(pscore),
                                ]
                            )
                        )
            self.pairs.append(np.array(pairs))

    def add_bounding_boxes(
        self,
        detections: list[Detection[BoundingBox]],
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
        ious = [
            compute_bbox_iou(
                np.array(
                    [
                        [gt.extrema, pd.extrema]
                        for pd in detection.predictions
                        for gt in detection.groundtruths
                    ],
                    dtype=np.float64,
                )
            ).reshape(len(detection.predictions), len(detection.groundtruths))
            for detection in detections
        ]
        return self._add_data(
            detections=detections,
            detection_ious=ious,
            show_progress=show_progress,
        )

    def add_polygons(
        self,
        detections: list[Detection[Polygon]],
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
        ious = [
            compute_polygon_iou(
                np.array(
                    [
                        [gt.shape, pd.shape]
                        for pd in detection.predictions
                        for gt in detection.groundtruths
                    ]
                )
            ).reshape(len(detection.predictions), len(detection.groundtruths))
            for detection in detections
        ]
        return self._add_data(
            detections=detections,
            detection_ious=ious,
            show_progress=show_progress,
        )

    def add_bitmasks(
        self,
        detections: list[Detection[Bitmask]],
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
        ious = [
            compute_bitmask_iou(
                np.array(
                    [
                        [gt.mask, pd.mask]
                        for pd in detection.predictions
                        for gt in detection.groundtruths
                    ]
                )
            ).reshape(len(detection.predictions), len(detection.groundtruths))
            for detection in detections
        ]
        return self._add_data(
            detections=detections,
            detection_ious=ious,
            show_progress=show_progress,
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
        self._evaluator._cache = np.concatenate(
            self.pairs,
            axis=0,
        )
        self._evaluator._label_metadata = compute_label_metadata(
            detailed_pairs=self._evaluator._cache,
            n_labels=self._evaluator.n_labels,
        )
        return self._evaluator
