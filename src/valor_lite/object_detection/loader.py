import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pyarrow import DataType
from tqdm import tqdm

from valor_lite import cache
from valor_lite.cache import FileCacheWriter, MemoryCacheWriter
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
    rank_table,
)
from valor_lite.object_detection.evaluator import Evaluator
from valor_lite.object_detection.shared import Base


class Loader(Base):
    def __init__(
        self,
        detailed_writer: MemoryCacheWriter | FileCacheWriter,
        ranked_writer: MemoryCacheWriter | FileCacheWriter,
        datum_metadata_fields: list[tuple[str, DataType]] | None = None,
        groundtruth_metadata_fields: list[tuple[str, DataType]] | None = None,
        prediction_metadata_fields: list[tuple[str, DataType]] | None = None,
        path: str | Path | None = None,
    ):
        self._path = Path(path) if path else None
        self._detailed_writer = detailed_writer
        self._ranked_writer = ranked_writer
        self._datum_metadata_fields = datum_metadata_fields
        self._groundtruth_metadata_fields = groundtruth_metadata_fields
        self._prediction_metadata_fields = prediction_metadata_fields

        # internal state
        self._labels = {}
        self._datum_count = 0
        self._groundtruth_count = 0
        self._prediction_count = 0

    @classmethod
    def in_memory(
        cls,
        batch_size: int = 10_000,
        datum_metadata_fields: list[tuple[str, DataType]] | None = None,
        groundtruth_metadata_fields: list[tuple[str, DataType]] | None = None,
        prediction_metadata_fields: list[tuple[str, DataType]] | None = None,
    ):
        """
        Create an in-memory evaluator cache.

        Parameters
        ----------
        batch_size : int, default=10_000
            The target number of rows to buffer before writing to the cache. Defaults to 10_000.
        datum_metadata_fields : list[tuple[str, DataType]], optional
            Optional datum metadata field definition.
        groundtruth_metadata_fields : list[tuple[str, DataType]], optional
            Optional ground truth annotation metadata field definition.
        prediction_metadata_fields : list[tuple[str, DataType]], optional
            Optional prediction metadata field definition.
        """
        # create cache
        detailed_writer = MemoryCacheWriter.create(
            schema=cls._generate_detailed_schema(
                datum_metadata_fields=datum_metadata_fields,
                groundtruth_metadata_fields=groundtruth_metadata_fields,
                prediction_metadata_fields=prediction_metadata_fields,
            ),
            batch_size=batch_size,
        )
        ranked_writer = MemoryCacheWriter.create(
            schema=cls._generate_ranked_schema(
                datum_metadata_fields=datum_metadata_fields,
            ),
            batch_size=batch_size,
        )

        return cls(
            detailed_writer=detailed_writer,
            ranked_writer=ranked_writer,
            datum_metadata_fields=datum_metadata_fields,
            groundtruth_metadata_fields=groundtruth_metadata_fields,
            prediction_metadata_fields=prediction_metadata_fields,
        )

    @classmethod
    def persistent(
        cls,
        path: str | Path,
        batch_size: int = 10_000,
        rows_per_file: int = 100_000,
        compression: str = "snappy",
        datum_metadata_fields: list[tuple[str, DataType]] | None = None,
        groundtruth_metadata_fields: list[tuple[str, DataType]] | None = None,
        prediction_metadata_fields: list[tuple[str, DataType]] | None = None,
        delete_if_exists: bool = False,
    ):
        """
        Create a persistent file-based evaluator cache.

        Parameters
        ----------
        path : str | Path
            Where to store the file-based cache.
        batch_size : int, default=10_000
            The target number of rows to buffer before writing to the cache. Defaults to 10_000.
        rows_per_file : int, default=100_000
            The target number of rows to store per cache file. Defaults to 100_000.
        compression : str, default="snappy"
            The compression methods used when writing cache files.
        datum_metadata_fields : list[tuple[str, DataType]], optional
            Optional datum metadata field definition.
        groundtruth_metadata_fields : list[tuple[str, DataType]], optional
            Optional ground truth annotation metadata field definition.
        prediction_metadata_fields : list[tuple[str, DataType]], optional
            Optional prediction metadata field definition.
        delete_if_exists : bool, default=False
            Option to delete any pre-exisiting cache at the given path.
        """
        path = Path(path)
        if delete_if_exists and path.exists():
            cls.delete_at_path(path)

        # create caches
        detailed_writer = FileCacheWriter.create(
            path=cls._generate_detailed_cache_path(path),
            schema=cls._generate_detailed_schema(
                datum_metadata_fields=datum_metadata_fields,
                groundtruth_metadata_fields=groundtruth_metadata_fields,
                prediction_metadata_fields=prediction_metadata_fields,
            ),
            batch_size=batch_size,
            rows_per_file=rows_per_file,
            compression=compression,
        )
        ranked_writer = FileCacheWriter.create(
            path=cls._generate_ranked_cache_path(path),
            schema=cls._generate_ranked_schema(
                datum_metadata_fields=datum_metadata_fields,
            ),
            batch_size=batch_size,
            rows_per_file=rows_per_file,
            compression=compression,
        )

        # write metadata
        metadata_path = cls._generate_metadata_path(path)
        with open(metadata_path, "w") as f:
            encoded_types = cls._encode_metadata_fields(
                datum_metadata_fields=datum_metadata_fields,
                groundtruth_metadata_fields=groundtruth_metadata_fields,
                prediction_metadata_fields=prediction_metadata_fields,
            )
            json.dump(encoded_types, f, indent=2)

        return cls(
            detailed_writer=detailed_writer,
            ranked_writer=ranked_writer,
            datum_metadata_fields=datum_metadata_fields,
            groundtruth_metadata_fields=groundtruth_metadata_fields,
            prediction_metadata_fields=prediction_metadata_fields,
            path=path,
        )

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

    def _rank(
        self,
        n_labels: int,
        batch_size: int = 1_000,
    ):
        """Perform pair ranking over the detailed cache."""

        detailed_reader = self._detailed_writer.to_reader()
        cache.sort(
            source=detailed_reader,
            sink=self._ranked_writer,
            batch_size=batch_size,
            sorting=[
                ("score", "descending"),
                ("iou", "descending"),
            ],
            columns=[
                field.name
                for field in self._ranked_writer.schema
                if field.name not in {"high_score", "iou_prev"}
            ],
            table_sort_override=lambda tbl: rank_table(
                tbl, number_of_labels=n_labels
            ),
        )
        self._ranked_writer.flush()

    def finalize(
        self,
        batch_size: int = 1_000,
        index_to_label_override: dict[int, str] | None = None,
    ):
        """
        Performs data finalization and preprocessing.

        Parameters
        ----------
        batch_size : int, default=1_000
            Sets the batch size for reading. Defaults to 1_000.
        index_to_label_override : dict[int, str], optional
            Pre-configures label mapping. Used when operating over filtered subsets.
        """
        self._detailed_writer.flush()
        if self._detailed_writer.count_rows() == 0:
            raise EmptyCacheError()

        detailed_reader = self._detailed_writer.to_reader()

        # build evaluator meta
        (
            index_to_label,
            number_of_groundtruths_per_label,
            info,
        ) = self._generate_meta(detailed_reader, index_to_label_override)
        info.datum_metadata_fields = self._datum_metadata_fields
        info.groundtruth_metadata_fields = self._groundtruth_metadata_fields
        info.prediction_metadata_fields = self._prediction_metadata_fields

        # populate ranked cache
        self._rank(
            n_labels=len(index_to_label),
            batch_size=batch_size,
        )

        ranked_reader = self._ranked_writer.to_reader()
        return Evaluator(
            detailed_reader=detailed_reader,
            ranked_reader=ranked_reader,
            info=info,
            index_to_label=index_to_label,
            number_of_groundtruths_per_label=number_of_groundtruths_per_label,
            path=self._path,
        )

    def delete(self):
        """Delete any cached files."""
        if self._path and self._path.exists():
            self.delete_at_path(self._path)
