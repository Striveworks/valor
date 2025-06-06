import warnings
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from valor_lite.exceptions import (
    EmptyEvaluatorError,
    EmptyFilterError,
    InternalCacheError,
)
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
    filter_cache,
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
        # count number of ground truths
        mask_valid_gts = detailed_pairs[:, 1] >= 0
        unique_ids = np.unique(
            detailed_pairs[np.ix_(mask_valid_gts, (0, 1))], axis=0  # type: ignore - np.ix_ typing
        )
        number_of_ground_truths = int(unique_ids.shape[0])

        # count number of predictions
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
    mask_datums: NDArray[np.bool_]
    mask_groundtruths: NDArray[np.bool_]
    mask_predictions: NDArray[np.bool_]
    metadata: Metadata

    def __post_init__(self):
        # validate datums mask
        if not self.mask_datums.any():
            raise EmptyFilterError("filter removes all datums")

        # validate annotation masks
        no_gts = self.mask_groundtruths.all()
        no_pds = self.mask_predictions.all()
        if no_gts and no_pds:
            raise EmptyFilterError("filter removes all annotations")
        elif no_gts:
            warnings.warn("filter removes all ground truths")
        elif no_pds:
            warnings.warn("filter removes all predictions")


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

        # internal cache
        self._detailed_pairs = np.array([[]], dtype=np.float64)
        self._ranked_pairs = np.array([[]], dtype=np.float64)
        self._label_metadata: NDArray[np.int32] = np.array([[]])
        self._metadata = Metadata()

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
    def metadata(self) -> Metadata:
        """
        Evaluation metadata.
        """
        return self._metadata

    def create_filter(
        self,
        datums: list[str] | NDArray[np.int32] | None = None,
        groundtruths: list[str] | NDArray[np.int32] | None = None,
        predictions: list[str] | NDArray[np.int32] | None = None,
        labels: list[str] | NDArray[np.int32] | None = None,
    ) -> Filter:
        """
        Creates a filter object.

        Parameters
        ----------
        datum : list[str] | NDArray[int32], optional
            An optional list of string ids or indices representing datums to keep.
        groundtruth : list[str] | NDArray[int32], optional
            An optional list of string ids or indices representing ground truth annotations to keep.
        prediction : list[str] | NDArray[int32], optional
            An optional list of string ids or indices representing prediction annotations to keep.
        labels : list[str] | NDArray[int32], optional
            An optional list of labels or indices to keep.
        """
        mask_datums = np.ones(self._detailed_pairs.shape[0], dtype=np.bool_)

        # filter datums
        if datums is not None:
            # convert to indices
            if isinstance(datums, list):
                datums = np.array(
                    [self.datum_id_to_index[uid] for uid in datums],
                    dtype=np.int32,
                )

            # validate indices
            if datums.size == 0:
                raise EmptyFilterError(
                    "filter removes all datums"
                )  # validate indices
            elif datums.min() < 0:
                raise ValueError(
                    f"datum index cannot be negative '{datums.min()}'"
                )
            elif datums.max() >= len(self.index_to_datum_id):
                raise ValueError(
                    f"datum index cannot exceed total number of datums '{datums.max()}'"
                )

            # apply to mask
            mask_datums = np.isin(self._detailed_pairs[:, 0], datums)

        filtered_detailed_pairs = self._detailed_pairs[mask_datums]
        n_pairs = self._detailed_pairs[mask_datums].shape[0]
        mask_groundtruths = np.zeros(n_pairs, dtype=np.bool_)
        mask_predictions = np.zeros_like(mask_groundtruths)

        # filter by ground truth annotation ids
        if groundtruths is not None:
            # convert to indices
            if isinstance(groundtruths, list):
                groundtruths = np.array(
                    [
                        self.groundtruth_id_to_index[uid]
                        for uid in groundtruths
                    ],
                    dtype=np.int32,
                )

            # validate indices
            if groundtruths.size == 0:
                warnings.warn("filter removes all ground truths")
            elif groundtruths.min() < 0:
                raise ValueError(
                    f"groundtruth annotation index cannot be negative '{groundtruths.min()}'"
                )
            elif groundtruths.max() >= len(self.index_to_groundtruth_id):
                raise ValueError(
                    f"groundtruth annotation index cannot exceed total number of groundtruths '{groundtruths.max()}'"
                )

            # apply to mask
            mask_groundtruths[
                ~np.isin(
                    filtered_detailed_pairs[:, 1],
                    groundtruths,
                )
            ] = True

        # filter by prediction annotation ids
        if predictions is not None:
            # convert to indices
            if isinstance(predictions, list):
                predictions = np.array(
                    [self.prediction_id_to_index[uid] for uid in predictions],
                    dtype=np.int32,
                )

            # validate indices
            if predictions.size == 0:
                warnings.warn("filter removes all predictions")
            elif predictions.min() < 0:
                raise ValueError(
                    f"prediction annotation index cannot be negative '{predictions.min()}'"
                )
            elif predictions.max() >= len(self.index_to_prediction_id):
                raise ValueError(
                    f"prediction annotation index cannot exceed total number of predictions '{predictions.max()}'"
                )

            # apply to mask
            mask_predictions[
                ~np.isin(
                    filtered_detailed_pairs[:, 2],
                    predictions,
                )
            ] = True

        # filter by labels
        if labels is not None:
            # convert to indices
            if isinstance(labels, list):
                labels = np.array(
                    [self.label_to_index[label] for label in labels]
                )

            # validate indices
            if labels.size == 0:
                raise EmptyFilterError("filter removes all labels")
            elif labels.min() < 0:
                raise ValueError(
                    f"label index cannot be negative '{labels.min()}'"
                )
            elif labels.max() >= len(self.index_to_label):
                raise ValueError(
                    f"label index cannot exceed total number of labels '{labels.max()}'"
                )

            # apply to mask
            labels = np.concatenate([labels, np.array([-1])])  # add null label
            mask_groundtruths[
                ~np.isin(filtered_detailed_pairs[:, 3], labels)
            ] = True
            mask_predictions[
                ~np.isin(filtered_detailed_pairs[:, 4], labels)
            ] = True

        filtered_detailed_pairs, _, _ = filter_cache(
            self._detailed_pairs,
            mask_datums=mask_datums,
            mask_ground_truths=mask_groundtruths,
            mask_predictions=mask_predictions,
            n_labels=len(self.index_to_label),
        )

        number_of_datums = (
            datums.size
            if datums is not None
            else np.unique(filtered_detailed_pairs[:, 0]).size
        )

        return Filter(
            mask_datums=mask_datums,
            mask_groundtruths=mask_groundtruths,
            mask_predictions=mask_predictions,
            metadata=Metadata.create(
                detailed_pairs=filtered_detailed_pairs,
                number_of_datums=number_of_datums,
                number_of_labels=len(self.index_to_label),
            ),
        )

    def filter(
        self, filter_: Filter
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int32],]:
        """
        Performs filtering over the internal cache.

        Parameters
        ----------
        filter_ : Filter
            The filter parameterization.

        Returns
        -------
        NDArray[float64]
            Filtered detailed pairs.
        NDArray[float64]
            Filtered ranked pairs.
        NDArray[int32]
            Label metadata.
        """
        return filter_cache(
            detailed_pairs=self._detailed_pairs,
            mask_datums=filter_.mask_datums,
            mask_ground_truths=filter_.mask_groundtruths,
            mask_predictions=filter_.mask_predictions,
            n_labels=len(self.index_to_label),
        )

    def compute_precision_recall(
        self,
        iou_thresholds: list[float],
        score_thresholds: list[float],
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
            A collection of filter parameters and masks.

        Returns
        -------
        dict[MetricType, list]
            A dictionary mapping MetricType enumerations to lists of computed metrics.
        """
        if not iou_thresholds:
            raise ValueError("At least one IOU threshold must be passed.")
        elif not score_thresholds:
            raise ValueError("At least one score threshold must be passed.")

        if filter_ is not None:
            _, ranked_pairs, label_metadata = self.filter(filter_=filter_)
        else:
            ranked_pairs = self._ranked_pairs
            label_metadata = self._label_metadata

        results = compute_precion_recall(
            ranked_pairs=ranked_pairs,
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
        iou_thresholds: list[float],
        score_thresholds: list[float],
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
        filter_ : Filter, optional
            A collection of filter parameters and masks.

        Returns
        -------
        list[Metric]
            List of confusion matrices per threshold pair.
        """
        if not iou_thresholds:
            raise ValueError("At least one IOU threshold must be passed.")
        elif not score_thresholds:
            raise ValueError("At least one score threshold must be passed.")

        if filter_ is not None:
            detailed_pairs, _, _ = self.filter(filter_=filter_)
        else:
            detailed_pairs = self._detailed_pairs

        if detailed_pairs.size == 0:
            return []

        results = compute_confusion_matrix(
            detailed_pairs=detailed_pairs,
            iou_thresholds=np.array(iou_thresholds),
            score_thresholds=np.array(score_thresholds),
        )
        return unpack_confusion_matrix_into_metric_list(
            results=results,
            detailed_pairs=detailed_pairs,
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
        filter_: Filter | None = None,
    ) -> dict[MetricType, list[Metric]]:
        """
        Computes all available metrics.

        Parameters
        ----------
        iou_thresholds : list[float], default=[0.1, 0.5, 0.75]
            A list of IOU thresholds to compute metrics over.
        score_thresholds : list[float], default=[0.5]
            A list of score thresholds to compute metrics over.
        filter_ : Filter, optional
            A collection of filter parameters and masks.

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
            filter_=filter_,
        )
        return metrics


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
                raise InternalCacheError("datum cache size mismatch")
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
                raise InternalCacheError("ground truth cache size mismatch")
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
                raise InternalCacheError("prediction cache size mismatch")
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
            if len(self._evaluator.label_to_index) != len(
                self._evaluator.index_to_label
            ):
                raise InternalCacheError("label cache size mismatch")
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
                self.pairs.append(data)

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
        if not self.pairs:
            raise EmptyEvaluatorError()

        n_labels = len(self._evaluator.index_to_label)
        n_datums = len(self._evaluator.index_to_datum_id)

        self._evaluator._detailed_pairs = np.concatenate(self.pairs, axis=0)
        if self._evaluator._detailed_pairs.size == 0:
            raise EmptyEvaluatorError()

        # order pairs by descending score, iou
        indices = np.lexsort(
            (
                -self._evaluator._detailed_pairs[:, 5],  # iou
                -self._evaluator._detailed_pairs[:, 6],  # score
            )
        )
        self._evaluator._detailed_pairs = self._evaluator._detailed_pairs[
            indices
        ]
        self._evaluator._label_metadata = compute_label_metadata(
            ids=self._evaluator._detailed_pairs[:, :5].astype(np.int32),
            n_labels=n_labels,
        )
        self._evaluator._ranked_pairs = rank_pairs(
            detailed_pairs=self._evaluator._detailed_pairs,
            label_metadata=self._evaluator._label_metadata,
        )
        self._evaluator._metadata = Metadata.create(
            detailed_pairs=self._evaluator._detailed_pairs,
            number_of_datums=n_datums,
            number_of_labels=n_labels,
        )
        return self._evaluator
