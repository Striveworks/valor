import warnings

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
    rank_pairs,
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
        self.datum_id_to_index: dict[str, int] = {}
        self.groundtruth_id_to_index: dict[str, int] = {}
        self.prediction_id_to_index: dict[str, int] = {}
        self.label_to_index: dict[str, int] = {}

        self.index_to_datum_id: list[str] = []
        self.index_to_groundtruth_id: list[str] = []
        self.index_to_prediction_id: list[str] = []
        self.index_to_label: list[str] = []

        # temporary cache
        self._temp_cache: list[NDArray[np.float64]] | None = []

        # cache
        self._detailed_pairs = np.array([[]], dtype=np.float64)
        self._ranked_pairs = np.array([[]], dtype=np.float64)
        self._label_metadata: NDArray[np.int32] = np.array([[]])

        # filter cache
        self._filtered_detailed_pairs: NDArray[np.float64] | None = None
        self._filtered_ranked_pairs: NDArray[np.float64] | None = None
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
    def ranked_pairs(self) -> NDArray[np.float64]:
        return (
            self._filtered_ranked_pairs
            if self._filtered_ranked_pairs is not None
            else self._ranked_pairs
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
        if not iou_thresholds:
            raise ValueError("At least one IOU threshold must be passed.")
        elif not score_thresholds:
            raise ValueError("At least one score threshold must be passed.")
        results = compute_precion_recall(
            ranked_pairs=self.ranked_pairs,
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
        if not iou_thresholds:
            raise ValueError("At least one IOU threshold must be passed.")
        elif not score_thresholds:
            raise ValueError("At least one score threshold must be passed.")
        elif self.detailed_pairs.size == 0:
            warnings.warn("attempted to compute over an empty set")
            return []
        results = compute_confusion_matrix(
            detailed_pairs=self.detailed_pairs,
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
        if datum_id not in self.datum_id_to_index:
            if len(self.datum_id_to_index) != len(self.index_to_datum_id):
                raise RuntimeError("datum cache size mismatch")
            idx = len(self.datum_id_to_index)
            self.datum_id_to_index[datum_id] = idx
            self.index_to_datum_id.append(datum_id)
        return self.datum_id_to_index[datum_id]

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
        if annotation_id not in self.groundtruth_id_to_index:
            if len(self.groundtruth_id_to_index) != len(
                self.index_to_groundtruth_id
            ):
                raise RuntimeError("ground truth cache size mismatch")
            idx = len(self.groundtruth_id_to_index)
            self.groundtruth_id_to_index[annotation_id] = idx
            self.index_to_groundtruth_id.append(annotation_id)
        return self.groundtruth_id_to_index[annotation_id]

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
        if annotation_id not in self.prediction_id_to_index:
            if len(self.prediction_id_to_index) != len(
                self.index_to_prediction_id
            ):
                raise RuntimeError("prediction cache size mismatch")
            idx = len(self.prediction_id_to_index)
            self.prediction_id_to_index[annotation_id] = idx
            self.index_to_prediction_id.append(annotation_id)
        return self.prediction_id_to_index[annotation_id]

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
        label_id = len(self.index_to_label)
        if label not in self.label_to_index:
            if len(self.label_to_index) != len(self.index_to_label):
                raise RuntimeError("label cache size mismatch")
            self.label_to_index[label] = label_id
            self.index_to_label.append(label)
            label_id += 1
        return self.label_to_index[label]

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
            pairs = []
            datum_idx = self._add_datum(detection.uid)
            if detection.groundtruths:
                for gidx, gann in enumerate(detection.groundtruths):
                    groundtruth_idx = self._add_groundtruth(gann.uid)
                    glabel_idx = self._add_label(gann.labels[0])
                    if (ious[:, gidx] < 1e-9).all():
                        pairs.extend(
                            [
                                np.array(
                                    [
                                        float(datum_idx),
                                        float(groundtruth_idx),
                                        -1.0,
                                        float(glabel_idx),
                                        -1.0,
                                        0.0,
                                        -1.0,
                                    ]
                                )
                            ]
                        )
                    for pidx, pann in enumerate(detection.predictions):
                        prediction_idx = self._add_prediction(pann.uid)
                        if (ious[pidx, :] < 1e-9).all():
                            pairs.extend(
                                [
                                    np.array(
                                        [
                                            float(datum_idx),
                                            -1.0,
                                            float(prediction_idx),
                                            -1.0,
                                            float(self._add_label(plabel)),
                                            0.0,
                                            float(pscore),
                                        ]
                                    )
                                    for plabel, pscore in zip(
                                        pann.labels, pann.scores
                                    )
                                ]
                            )
                        if ious[pidx, gidx] >= 1e-9:
                            pairs.extend(
                                [
                                    np.array(
                                        [
                                            float(datum_idx),
                                            float(groundtruth_idx),
                                            float(prediction_idx),
                                            float(self._add_label(glabel)),
                                            float(self._add_label(plabel)),
                                            ious[pidx, gidx],
                                            float(pscore),
                                        ]
                                    )
                                    for glabel in gann.labels
                                    for plabel, pscore in zip(
                                        pann.labels, pann.scores
                                    )
                                ]
                            )
            elif detection.predictions:
                for pidx, pann in enumerate(detection.predictions):
                    prediction_idx = self._add_prediction(pann.uid)
                    pairs.extend(
                        [
                            np.array(
                                [
                                    float(datum_idx),
                                    -1.0,
                                    float(prediction_idx),
                                    -1.0,
                                    float(self._add_label(plabel)),
                                    0.0,
                                    float(pscore),
                                ]
                            )
                            for plabel, pscore in zip(pann.labels, pann.scores)
                        ]
                    )

            data = np.array(pairs)
            if data.size > 0:
                # reset filtered cache if it exists
                self.clear_filter()
                if self._temp_cache is None:
                    raise RuntimeError(
                        "cannot add data as evaluator has already been finalized"
                    )
                self._temp_cache.append(data)

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

    def finalize(self):
        """
        Performs data finalization and some preprocessing steps.

        Returns
        -------
        Evaluator
            A ready-to-use evaluator object.
        """
        if self._temp_cache is None:
            warnings.warn("evaluator is already finalized or in a bad state")
            return self
        elif not self._temp_cache:
            self._detailed_pairs = np.array([], dtype=np.float64)
            self._ranked_pairs = np.array([], dtype=np.float64)
            self._label_metadata = np.zeros((self.n_labels, 2), dtype=np.int32)
            warnings.warn("no valid pairs")
            return self
        else:
            self._detailed_pairs = np.concatenate(self._temp_cache, axis=0)
            self._temp_cache = None

        # order pairs by descending score, iou
        indices = np.lexsort(
            (
                -self._detailed_pairs[:, 5],  # iou
                -self._detailed_pairs[:, 6],  # score
            )
        )
        self._detailed_pairs = self._detailed_pairs[indices]
        self._label_metadata = compute_label_metadata(
            ids=self._detailed_pairs[:, :5].astype(np.int32),
            n_labels=self.n_labels,
        )
        self._ranked_pairs = rank_pairs(
            detailed_pairs=self.detailed_pairs,
            label_metadata=self._label_metadata,
        )
        return self

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
        self._filtered_detailed_pairs = self._detailed_pairs.copy()
        self._filtered_ranked_pairs = np.array([], dtype=np.float64)
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

        valid_groundtruth_indices = None
        if groundtruth_ids is not None:
            valid_groundtruth_indices = np.array(
                [self.groundtruth_id_to_index[uid] for uid in groundtruth_ids],
                dtype=np.int32,
            )

        valid_prediction_indices = None
        if prediction_ids is not None:
            valid_prediction_indices = np.array(
                [self.prediction_id_to_index[uid] for uid in prediction_ids],
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

        # filter ground truth annotations
        if valid_groundtruth_indices is not None:
            mask_invalid_groundtruths[
                ~np.isin(
                    self._filtered_detailed_pairs[:, 1],
                    valid_groundtruth_indices,
                )
            ] = True

        # filter prediction annotations
        if valid_prediction_indices is not None:
            mask_invalid_predictions[
                ~np.isin(
                    self._filtered_detailed_pairs[:, 2],
                    valid_prediction_indices,
                )
            ] = True

        # filter labels
        if valid_label_indices is not None:
            mask_invalid_groundtruths[
                ~np.isin(
                    self._filtered_detailed_pairs[:, 3], valid_label_indices
                )
            ] = True
            mask_invalid_predictions[
                ~np.isin(
                    self._filtered_detailed_pairs[:, 4], valid_label_indices
                )
            ] = True

        # filter cache
        if mask_invalid_groundtruths.any():
            invalid_groundtruth_indices = np.where(mask_invalid_groundtruths)[
                0
            ]
            self._filtered_detailed_pairs[
                invalid_groundtruth_indices[:, None], (1, 3, 5)
            ] = np.array([[-1, -1, 0]])

        if mask_invalid_predictions.any():
            invalid_prediction_indices = np.where(mask_invalid_predictions)[0]
            self._filtered_detailed_pairs[
                invalid_prediction_indices[:, None], (2, 4, 5, 6)
            ] = np.array([[-1, -1, 0, -1]])

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

        # sorts by score, iou with ground truth id as a tie-breaker
        indices = np.lexsort(
            (
                self._filtered_detailed_pairs[:, 1],  # ground truth id
                -self._filtered_detailed_pairs[:, 5],  # iou
                -self._filtered_detailed_pairs[:, 6],  # score
            )
        )
        self._filtered_detailed_pairs = self._filtered_detailed_pairs[indices]
        self._filtered_label_metadata = compute_label_metadata(
            ids=self._filtered_detailed_pairs[:, :5].astype(np.int32),
            n_labels=self.n_labels,
        )
        self._filtered_ranked_pairs = rank_pairs(
            detailed_pairs=self._filtered_detailed_pairs,
            label_metadata=self._filtered_label_metadata,
        )

    def clear_filter(self):
        """Removes a filter if one exists."""
        self._filtered_detailed_pairs = None
        self._filtered_ranked_pairs = None
        self._filtered_label_metadata = None


class DataLoader(Evaluator):
    """
    Used for backwards compatibility as the Evaluator now handles ingestion.
    """

    pass
