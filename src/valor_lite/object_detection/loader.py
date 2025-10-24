import json
from pathlib import Path

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray
from tqdm import tqdm

from valor_lite.cache import (
    CacheWriter,
    DataType,
    convert_type_mapping_to_schema,
)
from valor_lite.exceptions import EmptyCacheError
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
from valor_lite.object_detection.evaluator import Evaluator, Filter
from valor_lite.object_detection.format import PathFormatter


class Loader(PathFormatter):
    def __init__(
        self,
        path: str | Path,
        writer: CacheWriter,
        datum_metadata_types: dict[str, DataType] | None = None,
        groundtruth_metadata_types: dict[str, DataType] | None = None,
        prediction_metadata_types: dict[str, DataType] | None = None,
    ):
        self._path = Path(path)
        self._cache = writer
        self._datum_metadata_types = datum_metadata_types
        self._groundtruth_metadata_types = groundtruth_metadata_types
        self._prediction_metadata_types = prediction_metadata_types

        # validate path
        if not self._path.exists():
            raise FileNotFoundError(f"Directory does not exist: {self._path}")
        elif not self._path.is_dir():
            raise NotADirectoryError(
                f"Path exists but is not a directory: {self._path}"
            )

        # internal state
        self._labels = {}
        self._datum_count = 0
        self._groundtruth_count = 0
        self._prediction_count = 0

    @classmethod
    def create(
        cls,
        path: str | Path,
        batch_size: int = 10_000,
        rows_per_file: int = 100_000,
        compression: str = "snappy",
        datum_metadata_types: dict[str, DataType] | None = None,
        groundtruth_metadata_types: dict[str, DataType] | None = None,
        prediction_metadata_types: dict[str, DataType] | None = None,
    ):
        datum_metadata_schema = convert_type_mapping_to_schema(
            datum_metadata_types
        )
        groundtruth_metadata_schema = convert_type_mapping_to_schema(
            groundtruth_metadata_types
        )
        prediction_metadata_schema = convert_type_mapping_to_schema(
            prediction_metadata_types
        )
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

        # create cache
        cache = CacheWriter.create(
            path=cls._generate_detailed_cache_path(path),
            schema=schema,
            batch_size=batch_size,
            rows_per_file=rows_per_file,
            compression=compression,
        )

        # write metadata
        metadata_path = cls._generate_metadata_path(path)
        with open(metadata_path, "w") as f:
            types = {
                "datum_metadata_types": datum_metadata_types,
                "groundtruth_metadata_types": groundtruth_metadata_types,
                "prediction_metadata_types": prediction_metadata_types,
            }
            json.dump(types, f, indent=2)

        return cls(
            path=path,
            writer=cache,
            datum_metadata_types=datum_metadata_types,
            groundtruth_metadata_types=groundtruth_metadata_types,
            prediction_metadata_types=prediction_metadata_types,
        )

    @classmethod
    def load(cls, path: str | Path):
        # load metadata specification
        metadata_path = cls._generate_metadata_path(path)
        with open(metadata_path, "r") as f:
            metadata_types = json.load(f)

        # load cache
        cache_path = cls._generate_detailed_cache_path(path)
        cache = CacheWriter.load(cache_path)

        return cls(
            path=path,
            writer=cache,
            **metadata_types,
        )

    @property
    def path(self) -> Path:
        return self._path

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
                                "datum_uid": detection.uid,
                                "datum_id": datum_idx,
                                **datum_metadata,
                                "gt_uid": gann.uid,
                                "gt_id": gt_id,
                                "gt_label": glabel,
                                "gt_label_id": glabel_idx,
                                **gann_metadata,
                                "pd_uid": None,
                                "pd_id": -1,
                                "pd_label": None,
                                "pd_label_id": -1,
                                "score": -1,
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
                                        "datum_uid": detection.uid,
                                        "datum_id": datum_idx,
                                        **datum_metadata,
                                        "gt_uid": None,
                                        "gt_id": -1,
                                        "gt_label": None,
                                        "gt_label_id": -1,
                                        "pd_uid": pann.uid,
                                        "pd_id": pann_id,
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
                                        "gt_id": gt_id,
                                        "gt_label": glabel,
                                        "gt_label_id": self._add_label(glabel),
                                        **gann_metadata,
                                        "pd_uid": pann.uid,
                                        "pd_id": pann_id,
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
                    pann_id = self._prediction_count + pidx
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
                                "pd_uid": pann.uid,
                                "pd_id": pann_id,
                                "pd_label": plabel,
                                "pd_label_id": self._add_label(plabel),
                                "score": float(pscore),
                                **pann_metadata,
                                "iou": 0.0,
                            }
                            for plabel, pscore in zip(pann.labels, pann.scores)
                        ]
                    )

            self._datum_count += 1
            self._groundtruth_count += len(detection.groundtruths)
            self._prediction_count += len(detection.predictions)

            pairs = sorted(pairs, key=lambda x: (-x["score"], -x["iou"]))
            self._cache.write_rows(pairs)

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

    @classmethod
    def filter(
        cls,
        path: str | Path,
        evaluator: Evaluator,
        filter_expr: Filter,
    ) -> Evaluator:
        loader = cls.create(
            path=path,
            batch_size=evaluator.detailed.batch_size,
            rows_per_file=evaluator.detailed.rows_per_file,
            compression=evaluator.detailed.compression,
            datum_metadata_types=evaluator.info.datum_metadata_types,
            groundtruth_metadata_types=evaluator.info.groundtruth_metadata_types,
            prediction_metadata_types=evaluator.info.prediction_metadata_types,
        )
        for fragment in evaluator.detailed.dataset.get_fragments():
            tbl = fragment.to_table(filter=filter_expr.datums)

            columns = (
                "datum_id",
                "gt_id",
                "pd_id",
                "gt_label_id",
                "pd_label_id",
                "iou",
                "score",
            )
            pairs = np.column_stack([tbl[col].to_numpy() for col in columns])

            n_pairs = pairs.shape[0]
            gt_ids = pairs[:, (0, 1)].astype(np.int64)
            pd_ids = pairs[:, (0, 2)].astype(np.int64)

            if filter_expr.groundtruths is not None:
                mask_valid_gt = np.zeros(n_pairs, dtype=np.bool_)
                gt_tbl = tbl.filter(filter_expr.groundtruths)
                gt_pairs = np.column_stack(
                    [gt_tbl[col].to_numpy() for col in ("datum_id", "gt_id")]
                ).astype(np.int64)
                for gt in np.unique(gt_pairs, axis=0):
                    mask_valid_gt |= (gt_ids == gt).all(axis=1)
            else:
                mask_valid_gt = np.ones(n_pairs, dtype=np.bool_)

            if filter_expr.predictions is not None:
                mask_valid_pd = np.zeros(n_pairs, dtype=np.bool_)
                pd_tbl = tbl.filter(filter_expr.predictions)
                pd_pairs = np.column_stack(
                    [pd_tbl[col].to_numpy() for col in ("datum_id", "pd_id")]
                ).astype(np.int64)
                for pd in np.unique(pd_pairs, axis=0):
                    mask_valid_pd |= (pd_ids == pd).all(axis=1)
            else:
                mask_valid_pd = np.ones(n_pairs, dtype=np.bool_)

            mask_valid = mask_valid_gt | mask_valid_pd
            mask_valid_gt &= mask_valid
            mask_valid_pd &= mask_valid

            pairs[np.ix_(~mask_valid_gt, (1, 3))] = -1.0  # type: ignore - numpy ix_
            pairs[np.ix_(~mask_valid_pd, (2, 4, 6))] = -1.0  # type: ignore - numpy ix_
            pairs[~mask_valid_pd | ~mask_valid_gt, 5] = 0.0

            for idx, col in enumerate(columns):
                tbl = tbl.set_column(
                    tbl.schema.names.index(col), col, pa.array(pairs[:, idx])
                )

            mask_invalid = ~mask_valid | (pairs[:, (1, 2)] < 0).all(axis=1)
            filtered_tbl = tbl.filter(pa.array(~mask_invalid))
            loader._cache.write_table(filtered_tbl)

        loader._cache.flush()
        if loader._cache.dataset.count_rows() == 0:
            raise EmptyCacheError()

        return Evaluator.create(
            path=loader.path,
            index_to_label_override=evaluator._index_to_label,
        )

    def finalize(
        self,
        batch_size: int = 1000,
    ):
        """
        Performs data finalization and some preprocessing steps.

        Returns
        -------
        Evaluator
            A ready-to-use evaluator object.
        """
        self._cache.flush()
        if self._cache.dataset.count_rows() == 0:
            raise EmptyCacheError()

        return Evaluator.create(
            path=self.path,
            batch_size=batch_size,
        )
