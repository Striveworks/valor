from pathlib import Path

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray
from tqdm import tqdm

from valor_lite.cache import Cache, DataType
from valor_lite.object_detection.annotation import BoundingBox, Detection
from valor_lite.object_detection.computation import compute_bbox_iou
from valor_lite.object_detection.manager import Evaluator


class Loader:
    def __init__(
        self,
        output_dir: str,
        batch_size: int = 1000,
        rows_per_file: int = 10000,
        compression: str = "snappy",
        datum_metadata_types: dict[str, DataType] | None = None,
        groundtruth_metadata_types: dict[str, DataType] | None = None,
        prediction_metadata_types: dict[str, DataType] | None = None,
    ):
        datum_metadata_schema = (
            [(k, v.to_arrow()) for k, v in datum_metadata_types.items()]
            if datum_metadata_types
            else []
        )
        groundtruth_metadata_schema = (
            [(k, v.to_arrow()) for k, v in groundtruth_metadata_types.items()]
            if groundtruth_metadata_types
            else []
        )
        prediction_metadata_schema = (
            [(k, v.to_arrow()) for k, v in prediction_metadata_types.items()]
            if prediction_metadata_types
            else []
        )

        self._null_gt_metadata = {
            s[0]: None for s in groundtruth_metadata_schema
        }
        self._null_pd_metadata = {
            s[0]: None for s in prediction_metadata_schema
        }

        annotation_pair_struct = pa.struct(
            [
                # groundtruth
                ("gt_uid", pa.string()),
                ("gt_id", pa.int64()),
                ("gt_label", pa.string()),
                ("gt_label_id", pa.int64()),
                *groundtruth_metadata_schema,
                # prediction
                ("pd_uid", pa.string()),
                ("pd_id", pa.int64()),
                ("pd_label", pa.string()),
                ("pd_label_id", pa.int64()),
                ("score", pa.float64()),
                *prediction_metadata_schema,
                # pair
                ("iou", pa.float64()),
            ]
        )
        pair_list_type = pa.list_(annotation_pair_struct)
        schema = pa.schema(
            [
                ("uid", pa.string()),
                ("id", pa.int64()),
                ("pairs", pair_list_type),
                *datum_metadata_schema,
            ]
        )

        self.detailed = Cache(
            output_dir=Path(output_dir) / Path("detailed"),
            schema=schema,
            batch_size=batch_size,
            rows_per_file=rows_per_file,
            compression=compression,
        )
        self.ranked = Cache(
            output_dir=Path(output_dir) / Path("ranked"),
            schema=schema,
            batch_size=batch_size,
            rows_per_file=rows_per_file,
            compression=compression,
        )
        self._labels = {}

    def _add_label(self, value: str) -> int:
        if idx := self._labels.get(value, None):
            return idx
        self._labels[value] = len(self._labels)
        return self._labels[value]

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
            datum_idx = self.detailed.total_rows
            matched_annotations = []
            unmatched_groundtruths = []
            unmatched_predictions = []
            if detection.groundtruths:
                for gidx, gann in enumerate(detection.groundtruths):
                    glabel = gann.labels[0]
                    glabel_idx = self._add_label(gann.labels[0])
                    gann_metadata = gann.metadata if gann.metadata else {}
                    if (ious[:, gidx] < 1e-9).all():
                        unmatched_groundtruths.append(
                            {
                                "gt_uid": gann.uid,
                                "gt_id": gidx,
                                "gt_label": glabel,
                                "gt_label_id": glabel_idx,
                                **gann_metadata,
                                "pd_uid": None,
                                "pd_id": -1,
                                "pd_label": None,
                                "pd_label_id": -1,
                                "score": -1,
                                **self._null_pd_metadata,
                                "iou": 0.0,
                            }
                        )
                    for pidx, pann in enumerate(detection.predictions):
                        pann_metadata = pann.metadata if pann.metadata else {}
                        if (ious[pidx, :] < 1e-9).all():
                            unmatched_predictions.extend(
                                [
                                    {
                                        "gt_uid": None,
                                        "gt_id": -1,
                                        "gt_label": None,
                                        "gt_label_id": -1,
                                        **self._null_gt_metadata,
                                        "pd_uid": pann.uid,
                                        "pd_id": pidx,
                                        "pd_label": plabel,
                                        "pd_label_id": self._add_label(plabel),
                                        "score": float(pscore),
                                        **pann_metadata,
                                        "iou": 0.0,
                                    }
                                    for plabel, pscore in zip(
                                        pann.labels, pann.scores
                                    )
                                ]
                            )
                        if ious[pidx, gidx] >= 1e-9:
                            matched_annotations.extend(
                                [
                                    {
                                        "gt_uid": gann.uid,
                                        "gt_id": gidx,
                                        "gt_label": glabel,
                                        "gt_label_id": self._add_label(glabel),
                                        **gann_metadata,
                                        "pd_uid": pann.uid,
                                        "pd_id": pidx,
                                        "pd_label": plabel,
                                        "pd_label_id": self._add_label(plabel),
                                        "score": float(pscore),
                                        **pann_metadata,
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
                    pann_metadata = pann.metadata if pann.metadata else {}
                    unmatched_predictions.extend(
                        [
                            {
                                "gt_uid": None,
                                "gt_id": -1,
                                "gt_label": None,
                                "gt_label_id": -1,
                                **self._null_gt_metadata,
                                "pd_uid": pann.uid,
                                "pd_id": pidx,
                                "pd_label": plabel,
                                "pd_label_id": self._add_label(plabel),
                                "score": float(pscore),
                                **pann_metadata,
                                "iou": 0.0,
                            }
                            for plabel, pscore in zip(pann.labels, pann.scores)
                        ]
                    )

            annotations = sorted(
                matched_annotations
                + unmatched_groundtruths
                + unmatched_predictions,
                key=lambda x: (-x["score"], -x["iou"]),
            )
            self.detailed.write(
                {
                    "datum_uid": detection.uid,
                    "datum_id": datum_idx,
                    "pairs": annotations,
                }
            )

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

    # def finalize(self) -> Evaluator:
    #     """
    #     Performs data finalization and some preprocessing steps.

    #     Returns
    #     -------
    #     Evaluator
    #         A ready-to-use evaluator object.
    #     """

    # self._evaluator._ranked_pairs = rank_pairs(
    #     detailed_pairs=self._evaluator._detailed_pairs,
    #     label_metadata=self._evaluator._label_metadata,
    # )
    # self._evaluator._metadata = Metadata.create(
    #     detailed_pairs=self._evaluator._detailed_pairs,
    #     number_of_datums=n_datums,
    #     number_of_labels=n_labels,
    # )
    # return self._evaluator


if __name__ == "__main__":
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    tbl = pq.read_table("bench/part-000000.parquet")
    print(tbl)
    c = pc.list_value_length(tbl["pairs"]).to_numpy()
    print(c.max(), c.min(), c.mean())
