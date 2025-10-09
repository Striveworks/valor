import json
from pathlib import Path

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray
from tqdm import tqdm

from valor_lite.cache import Cache, DataType
from valor_lite.object_detection.annotation import BoundingBox, Detection
from valor_lite.object_detection.computation import compute_bbox_iou
from valor_lite.object_detection.evaluator import Evaluator


class Loader:
    def __init__(
        self,
        directory: str | Path,
        batch_size: int = 10000,
        rows_per_file: int = 1_000_000,
        compression: str = "snappy",
        datum_metadata_types: dict[str, DataType] | None = None,
        groundtruth_metadata_types: dict[str, DataType] | None = None,
        prediction_metadata_types: dict[str, DataType] | None = None,
    ):
        self._labels = {}
        self._count = 0

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

        schema = pa.schema(
            [
                ("datum_uid", pa.string()),
                ("datum_id", pa.int64()),
                *datum_metadata_schema,
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

        self._dir = Path(directory)
        self._labels_path = self._dir / Path("labels.json")
        self._detailed_path = self._dir / Path("detailed")
        self._ranked_path = self._dir / Path("ranked")

        self._detailed = Cache(
            where=self._detailed_path,
            schema=schema,
            batch_size=batch_size,
            rows_per_file=rows_per_file,
            compression=compression,
        )

    def _add_label(self, value: str) -> int:
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
            datum_idx = self._count
            self._count += 1
            datum_metadata = detection.metadata if detection.metadata else {}
            pairs = []
            if detection.groundtruths:
                for gidx, gann in enumerate(detection.groundtruths):
                    glabel = gann.labels[0]
                    glabel_idx = self._add_label(gann.labels[0])
                    gann_metadata = gann.metadata if gann.metadata else {}
                    if (ious[:, gidx] < 1e-9).all():
                        pairs.append(
                            {
                                "datum_uid": detection.uid,
                                "datum_id": datum_idx,
                                **datum_metadata,
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
                            pairs.extend(
                                [
                                    {
                                        "datum_uid": detection.uid,
                                        "datum_id": datum_idx,
                                        **datum_metadata,
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
                            pairs.extend(
                                [
                                    {
                                        "datum_uid": detection.uid,
                                        "datum_id": datum_idx,
                                        **datum_metadata,
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
                    pairs.extend(
                        [
                            {
                                "datum_uid": detection.uid,
                                "datum_id": datum_idx,
                                **datum_metadata,
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

            pairs = sorted(pairs, key=lambda x: (-x["score"], -x["iou"]))
            self._detailed.write_rows(pairs)

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

    def finalize(self):
        """
        Performs data finalization and some preprocessing steps.

        Returns
        -------
        Evaluator
            A ready-to-use evaluator object.
        """
        self._detailed.flush()
        self._detailed.sort_by(
            destination=self._ranked_path,
            sorting=[
                ("score", "descending"),
                ("iou", "descending"),
            ],
        )
        with open(self._labels_path, "w") as f:
            json.dump({v: k for k, v in self._labels.items()}, f, indent=2)
        return Evaluator(self._dir)


if __name__ == "__main__":
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    tbl = pq.read_table("bench/part-000000.parquet")
    print(tbl)
    c = pc.list_value_length(tbl["pairs"]).to_numpy()
    print(c.max(), c.min(), c.mean())
