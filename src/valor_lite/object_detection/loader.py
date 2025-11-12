import numpy as np
import pyarrow as pa
from numpy.typing import NDArray
from tqdm import tqdm

from valor_lite.cache import FileCacheWriter, MemoryCacheWriter
from valor_lite.object_detection.annotation import (
    Bitmask,
    BoundingBox,
    Detection,
    Polygon,
)
from valor_lite.object_detection.computation import (
    compute_bbox_iou,
    compute_bitmask_iou,
    compute_polygon_iou,
)
from valor_lite.object_detection.evaluator import Builder


class Loader(Builder):
    def __init__(
        self,
        detailed_writer: MemoryCacheWriter | FileCacheWriter,
        ranked_writer: MemoryCacheWriter | FileCacheWriter,
        metadata_fields: list[tuple[str, pa.DataType]] | None = None,
    ):
        super().__init__(
            detailed_writer=detailed_writer,
            ranked_writer=ranked_writer,
            metadata_fields=metadata_fields,
        )

        # internal state
        self._labels = {}
        self._datum_count = 0
        self._groundtruth_count = 0
        self._prediction_count = 0

    def _add_label(self, value: str) -> int:
        """Add a label to the index mapping."""
        idx = self._labels.get(value, None)
        if idx is None:
            idx = len(self._labels)
            self._labels[value] = idx
        return idx

    def _add_data(
        self,
        detections: list[Detection],
        detection_ious: list[NDArray[np.float64]],
        show_progress: bool = False,
    ):
        """Adds detections to the cache."""
        disable_tqdm = not show_progress
        for detection, ious in tqdm(
            zip(detections, detection_ious), disable=disable_tqdm
        ):
            # cache labels and annotation pairs
            datum_idx = self._datum_count
            datum_metadata = detection.metadata if detection.metadata else {}
            pairs = []
            if detection.groundtruths:
                for gidx, gann in enumerate(detection.groundtruths):
                    gt_id = self._groundtruth_count + gidx
                    glabel = gann.labels[0]
                    glabel_idx = self._add_label(gann.labels[0])
                    gann_metadata = gann.metadata if gann.metadata else {}
                    if (ious[:, gidx] < 1e-9).all():
                        pairs.append(
                            {
                                # metadata
                                **datum_metadata,
                                **gann_metadata,
                                # datum
                                "datum_uid": detection.uid,
                                "datum_id": datum_idx,
                                # groundtruth
                                "gt_uid": gann.uid,
                                "gt_id": gt_id,
                                "gt_label": glabel,
                                "gt_label_id": glabel_idx,
                                # prediction
                                "pd_uid": None,
                                "pd_id": -1,
                                "pd_label": None,
                                "pd_label_id": -1,
                                "pd_score": -1,
                                # pair
                                "iou": 0.0,
                            }
                        )
                    for pidx, pann in enumerate(detection.predictions):
                        pann_id = self._prediction_count + pidx
                        pann_metadata = pann.metadata if pann.metadata else {}
                        if (ious[pidx, :] < 1e-9).all():
                            pairs.extend(
                                [
                                    {
                                        # metadata
                                        **datum_metadata,
                                        **pann_metadata,
                                        # datum
                                        "datum_uid": detection.uid,
                                        "datum_id": datum_idx,
                                        # groundtruth
                                        "gt_uid": None,
                                        "gt_id": -1,
                                        "gt_label": None,
                                        "gt_label_id": -1,
                                        # prediction
                                        "pd_uid": pann.uid,
                                        "pd_id": pann_id,
                                        "pd_label": plabel,
                                        "pd_label_id": self._add_label(plabel),
                                        "pd_score": float(pscore),
                                        # pair
                                        "iou": 0.0,
                                    }
                                    for plabel, pscore in zip(
                                        pann.labels, pann.scores
                                    )
                                ]
                            )
                        if ious[pidx, gidx] >= 1e-9:
                            pairs.extend(
                                [
                                    {
                                        # metadata
                                        **datum_metadata,
                                        **gann_metadata,
                                        **pann_metadata,
                                        # datum
                                        "datum_uid": detection.uid,
                                        "datum_id": datum_idx,
                                        # groundtruth
                                        "gt_uid": gann.uid,
                                        "gt_id": gt_id,
                                        "gt_label": glabel,
                                        "gt_label_id": self._add_label(glabel),
                                        # prediction
                                        "pd_uid": pann.uid,
                                        "pd_id": pann_id,
                                        "pd_label": plabel,
                                        "pd_label_id": self._add_label(plabel),
                                        "pd_score": float(pscore),
                                        # pair
                                        "iou": float(ious[pidx, gidx]),
                                    }
                                    for glabel in gann.labels
                                    for plabel, pscore in zip(
                                        pann.labels, pann.scores
                                    )
                                ]
                            )
            elif detection.predictions:
                for pidx, pann in enumerate(detection.predictions):
                    pann_id = self._prediction_count + pidx
                    pann_metadata = pann.metadata if pann.metadata else {}
                    pairs.extend(
                        [
                            {
                                # metadata
                                **datum_metadata,
                                **pann_metadata,
                                # datum
                                "datum_uid": detection.uid,
                                "datum_id": datum_idx,
                                # groundtruth
                                "gt_uid": None,
                                "gt_id": -1,
                                "gt_label": None,
                                "gt_label_id": -1,
                                # prediction
                                "pd_uid": pann.uid,
                                "pd_id": pann_id,
                                "pd_label": plabel,
                                "pd_label_id": self._add_label(plabel),
                                "pd_score": float(pscore),
                                # pair
                                "iou": 0.0,
                            }
                            for plabel, pscore in zip(pann.labels, pann.scores)
                        ]
                    )

            self._datum_count += 1
            self._groundtruth_count += len(detection.groundtruths)
            self._prediction_count += len(detection.predictions)

            pairs = sorted(pairs, key=lambda x: (-x["pd_score"], -x["iou"]))
            self._detailed_writer.write_rows(pairs)

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
