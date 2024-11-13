from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from valor_lite.object_detection.annotation import Detection
from valor_lite.object_detection.computation import (
    compute_bbox_iou,
    compute_bitmask_iou,
    compute_confusion_matrix,
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

    def compute_precision_recall(
        self,
        iou_thresholds: list[float] = [0.5, 0.75, 0.9],
        score_thresholds: list[float] = [0.5],
        filter_: Filter | None = None,
    ) -> dict[MetricType, list[Metric]]:
        """
        Computes all metrics except for ConfusionMatrix

        Parameters
        ----------
        iou_thresholds : list[float]
            A list of IOU thresholds to compute metrics over.
        score_thresholds : list[float]
            A list of score thresholds to compute metrics over.
        filter_ : Filter, optional
            An optional filter object.

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

        results = compute_precion_recall(
            data=ranked_pairs,
            label_metadata=label_metadata,
            iou_thresholds=np.array(iou_thresholds),
            score_thresholds=np.array(score_thresholds),
        )

        return unpack_precision_recall_into_metric_lists(
            results=results,
            label_metadata=label_metadata,
            iou_thresholds=iou_thresholds,
            score_thresholds=score_thresholds,
            index_to_label=self.index_to_label,
        )

    def compute_confusion_matrix(
        self,
        iou_thresholds: list[float] = [0.5, 0.75, 0.9],
        score_thresholds: list[float] = [0.5],
        number_of_examples: int = 0,
        filter_: Filter | None = None,
    ) -> list[Metric]:
        """
        Computes confusion matrices at various thresholds.

        Parameters
        ----------
        iou_thresholds : list[float]
            A list of IOU thresholds to compute metrics over.
        score_thresholds : list[float]
            A list of score thresholds to compute metrics over.
        number_of_examples : int, default=0
            Maximum number of annotation examples to return in ConfusionMatrix.
        filter_ : Filter, optional
            An optional filter object.

        Returns
        -------
        list[Metric]
            List of confusion matrices per threshold pair.
        """

        detailed_pairs = self._detailed_pairs
        label_metadata = self._label_metadata
        if filter_ is not None:
            detailed_pairs = detailed_pairs[filter_.detailed_indices]
            label_metadata = filter_.label_metadata

        if detailed_pairs.size == 0:
            return list()

        results = compute_confusion_matrix(
            data=detailed_pairs,
            label_metadata=label_metadata,
            iou_thresholds=np.array(iou_thresholds),
            score_thresholds=np.array(score_thresholds),
            n_examples=number_of_examples,
        )

        return unpack_confusion_matrix_into_metric_list(
            results=results,
            iou_thresholds=iou_thresholds,
            score_thresholds=score_thresholds,
            number_of_examples=number_of_examples,
            index_to_uid=self.index_to_uid,
            index_to_label=self.index_to_label,
            groundtruth_examples=self.groundtruth_examples,
            prediction_examples=self.prediction_examples,
        )

    def evaluate(
        self,
        iou_thresholds: list[float] = [0.5, 0.75, 0.9],
        score_thresholds: list[float] = [0.5],
        number_of_examples: int = 0,
        filter_: Filter | None = None,
    ) -> dict[MetricType, list[Metric]]:
        """
        Computes all available metrics.

        Parameters
        ----------
        iou_thresholds : list[float]
            A list of IOU thresholds to compute metrics over.
        score_thresholds : list[float]
            A list of score thresholds to compute metrics over.
        number_of_examples : int, default=0
            Maximum number of annotation examples to return in ConfusionMatrix.
        filter_ : Filter, optional
            An optional filter object.

        Returns
        -------
        dict[MetricType, list[Metric]]
            Lists of metrics organized by metric type.
        """
        metrics = self.compute_precision_recall(
            iou_thresholds=iou_thresholds,
            score_thresholds=score_thresholds,
            filter_=filter_,
        )

        metrics[MetricType.ConfusionMatrix] = self.compute_confusion_matrix(
            iou_thresholds=iou_thresholds,
            score_thresholds=score_thresholds,
            number_of_examples=number_of_examples,
            filter_=filter_,
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
        self.groundtruth_count = defaultdict(defaultdict_int)
        self.prediction_count = defaultdict(defaultdict_int)

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

    def _cache_pairs(
        self,
        uid_index: int,
        groundtruths: list,
        predictions: list,
        ious: NDArray[np.float64],
    ) -> None:
        """
        Compute IOUs between groundtruths and preditions before storing as pairs.

        Parameters
        ----------
        uid_index : int
            The index of the detection.
        groundtruths : list
            A list of groundtruths.
        predictions : list
            A list of predictions.
        ious : NDArray[np.float64]
            An array with shape (n_preds, n_gts) containing IOUs.
        """

        predictions_with_iou_of_zero = np.where((ious < 1e-9).all(axis=1))[0]
        groundtruths_with_iou_of_zero = np.where((ious < 1e-9).all(axis=0))[0]

        pairs = [
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
            for pidx, plabel, score in predictions
            for gidx, glabel in groundtruths
            if ious[pidx, gidx] >= 1e-9
        ]
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
                        )
                    )

            for pidx, pann in enumerate(detection.predictions):
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
                        )
                    )

            self._cache_pairs(
                uid_index=uid_index,
                groundtruths=groundtruths,
                predictions=predictions,
                ious=ious,
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
        ious = [
            compute_bbox_iou(
                np.array(
                    [
                        [gt.extrema, pd.extrema]
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
        ious = [
            compute_polygon_iou(
                np.array(
                    [
                        [gt.shape, pd.shape]  # type: ignore - using the AttributeError as a validator
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
        ious = [
            compute_bitmask_iou(
                np.array(
                    [
                        [gt.mask, pd.mask]  # type: ignore - using the AttributeError as a validator
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
