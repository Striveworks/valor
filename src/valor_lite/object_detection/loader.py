import heapq
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray
from tqdm import tqdm

from valor_lite.common.datatype import DataType, convert_type_mapping_to_schema
from valor_lite.common.ephemeral import MemoryCacheReader, MemoryCacheWriter
from valor_lite.common.persistent import FileCacheWriter
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
from valor_lite.object_detection.format import PathFormatter


class Loader(PathFormatter):
    def __init__(
        self,
        detailed_writer: MemoryCacheWriter | FileCacheWriter,
        ranked_writer: MemoryCacheWriter | FileCacheWriter,
        datum_metadata_types: dict[str, DataType] | None = None,
        groundtruth_metadata_types: dict[str, DataType] | None = None,
        prediction_metadata_types: dict[str, DataType] | None = None,
        path: str | Path | None = None,
    ):
        self._path = Path(path) if path else None
        self._detailed_writer = detailed_writer
        self._ranked_writer = ranked_writer
        self._datum_metadata_types = datum_metadata_types
        self._groundtruth_metadata_types = groundtruth_metadata_types
        self._prediction_metadata_types = prediction_metadata_types

        # internal state
        self._labels = {}
        self._datum_count = 0
        self._groundtruth_count = 0
        self._prediction_count = 0

    @classmethod
    def in_memory(
        cls,
        batch_size: int = 10_000,
        datum_metadata_types: dict[str, DataType] | None = None,
        groundtruth_metadata_types: dict[str, DataType] | None = None,
        prediction_metadata_types: dict[str, DataType] | None = None,
    ):
        datum_metadata_fields = convert_type_mapping_to_schema(
            datum_metadata_types
        )
        groundtruth_metadata_fields = convert_type_mapping_to_schema(
            groundtruth_metadata_types
        )
        prediction_metadata_fields = convert_type_mapping_to_schema(
            prediction_metadata_types
        )

        # create cache
        detailed_writer = MemoryCacheWriter.create(
            schema=cls.detailed_schema(
                datum_metadata_fields=datum_metadata_fields,
                groundtruth_metadata_fields=groundtruth_metadata_fields,
                prediction_metadata_fields=prediction_metadata_fields,
            ),
            batch_size=batch_size,
        )
        ranked_writer = MemoryCacheWriter.create(
            schema=cls.ranked_schema(
                datum_metadata_fields=datum_metadata_fields,
            ),
            batch_size=batch_size,
        )

        return cls(
            detailed_writer=detailed_writer,
            ranked_writer=ranked_writer,
            datum_metadata_types=datum_metadata_types,
            groundtruth_metadata_types=groundtruth_metadata_types,
            prediction_metadata_types=prediction_metadata_types,
        )

    @classmethod
    def persistent(
        cls,
        path: str | Path,
        batch_size: int = 10_000,
        rows_per_file: int = 100_000,
        compression: str = "snappy",
        datum_metadata_types: dict[str, DataType] | None = None,
        groundtruth_metadata_types: dict[str, DataType] | None = None,
        prediction_metadata_types: dict[str, DataType] | None = None,
        delete_if_exists: bool = False,
    ):
        path = Path(path)
        if delete_if_exists and path.exists():
            cls.delete(path)

        datum_metadata_fields = convert_type_mapping_to_schema(
            datum_metadata_types
        )
        groundtruth_metadata_fields = convert_type_mapping_to_schema(
            groundtruth_metadata_types
        )
        prediction_metadata_fields = convert_type_mapping_to_schema(
            prediction_metadata_types
        )

        # create caches
        detailed_writer = FileCacheWriter.create(
            path=cls._generate_detailed_cache_path(path),
            schema=cls.detailed_schema(
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
            schema=cls.ranked_schema(
                datum_metadata_fields=datum_metadata_fields,
            ),
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
            detailed_writer=detailed_writer,
            ranked_writer=ranked_writer,
            datum_metadata_types=datum_metadata_types,
            groundtruth_metadata_types=groundtruth_metadata_types,
            prediction_metadata_types=prediction_metadata_types,
            path=path,
        )

    @classmethod
    def delete(cls, path: str | Path):
        """
        Delete file-based cache.

        Parameters
        ----------
        path : str | Path
            Where the file-based cache is located.
        """
        path = Path(path)
        if not path.exists():
            return
        detailed_path = cls._generate_detailed_cache_path(path)
        ranked_path = cls._generate_ranked_cache_path(path)
        metadata_path = cls._generate_metadata_path(path)
        FileCacheWriter.delete(detailed_path)
        FileCacheWriter.delete(ranked_path)
        if metadata_path.exists() and metadata_path.is_file():
            metadata_path.unlink()
        path.rmdir()

    @staticmethod
    def detailed_schema(
        datum_metadata_fields: list[tuple[str, pa.DataType]],
        groundtruth_metadata_fields: list[tuple[str, pa.DataType]],
        prediction_metadata_fields: list[tuple[str, pa.DataType]],
    ) -> pa.Schema:
        return pa.schema(
            [
                ("datum_uid", pa.string()),
                ("datum_id", pa.int64()),
                *datum_metadata_fields,
                # groundtruth
                ("gt_uid", pa.string()),
                ("gt_id", pa.int64()),
                ("gt_label", pa.string()),
                ("gt_label_id", pa.int64()),
                *groundtruth_metadata_fields,
                # prediction
                ("pd_uid", pa.string()),
                ("pd_id", pa.int64()),
                ("pd_label", pa.string()),
                ("pd_label_id", pa.int64()),
                ("score", pa.float64()),
                *prediction_metadata_fields,
                # pair
                ("iou", pa.float64()),
            ]
        )

    @staticmethod
    def ranked_schema(
        datum_metadata_fields: list[tuple[str, pa.DataType]],
    ) -> pa.Schema:
        return pa.schema(
            [
                ("datum_uid", pa.string()),
                ("datum_id", pa.int64()),
                *datum_metadata_fields,
                # groundtruth
                ("gt_id", pa.int64()),
                ("gt_label_id", pa.int64()),
                # prediction
                ("pd_id", pa.int64()),
                ("pd_label_id", pa.int64()),
                ("score", pa.float64()),
                # pair
                ("iou", pa.float64()),
                ("high_score", pa.bool_()),
                ("iou_prev", pa.float64()),
            ]
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

    def rank(
        self,
        n_labels: int,
        batch_size: int = 1_000,
    ):
        detailed_reader = self._detailed_writer.to_reader()
        subset_columns = [
            field.name
            for field in self._ranked_writer._schema
            if field.name not in {"high_score", "iou_prev"}
        ]
        if (
            isinstance(detailed_reader, MemoryCacheReader)
            or detailed_reader.count_tables() == 1
        ):
            for tbl in detailed_reader.iterate_tables(columns=subset_columns):
                ranked_tbl = rank_table(tbl, n_labels)
                self._ranked_writer.write_table(ranked_tbl)
        elif isinstance(self._ranked_writer, FileCacheWriter):
            if not self._path:
                raise ValueError(
                    "missing path definition in file-based loader"
                )
            path = self._generate_temporary_cache_path(self._path)
            with FileCacheWriter.create(
                path=path,
                schema=self._ranked_writer._schema,
                batch_size=self._ranked_writer._batch_size,
                rows_per_file=self._ranked_writer._rows_per_file,
                compression=self._ranked_writer._compression,
                delete_if_exists=True,
            ) as tmp_writer:

                # rank individual files
                for tbl in detailed_reader.iterate_tables(
                    columns=subset_columns
                ):
                    ranked_tbl = rank_table(tbl, n_labels)
                    tmp_writer.write_table(ranked_tbl)

            tmp_reader = tmp_writer.to_reader()

            def generate_heap_item(batches, batch_idx, row_idx) -> tuple:
                score = batches[batch_idx]["score"][row_idx].as_py()
                iou = batches[batch_idx]["iou"][row_idx].as_py()
                return (
                    -score,
                    -iou,
                    batch_idx,
                    row_idx,
                )

            # merge sorted rows
            heap = []
            batch_iterators = []
            batches = []
            for batch_idx, batch_fragment in enumerate(
                tmp_reader.iterate_fragments()
            ):
                batch_iter = batch_fragment.to_batches(batch_size=batch_size)
                batch_iterators.append(batch_iter)
                batches.append(next(batch_iterators[batch_idx], None))
                if (
                    batches[batch_idx] is not None
                    and len(batches[batch_idx]) > 0
                ):
                    heapq.heappush(
                        heap, generate_heap_item(batches, batch_idx, 0)
                    )

            while heap:
                _, _, batch_idx, row_idx = heapq.heappop(heap)
                row_table = batches[batch_idx].slice(row_idx, 1)
                self._ranked_writer.write_batch(row_table)
                row_idx += 1
                if row_idx < len(batches[batch_idx]):
                    heapq.heappush(
                        heap,
                        generate_heap_item(batches, batch_idx, row_idx),
                    )
                else:
                    batches[batch_idx] = next(batch_iterators[batch_idx], None)
                    if (
                        batches[batch_idx] is not None
                        and len(batches[batch_idx]) > 0
                    ):
                        heapq.heappush(
                            heap,
                            generate_heap_item(batches, batch_idx, 0),
                        )

            FileCacheWriter.delete(path)

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
        ) = Evaluator.generate_meta(detailed_reader, index_to_label_override)
        info.datum_metadata_types = self._datum_metadata_types
        info.groundtruth_metadata_types = self._groundtruth_metadata_types
        info.prediction_metadata_types = self._prediction_metadata_types

        n_labels = len(index_to_label)

        self.rank(n_labels=n_labels, batch_size=batch_size)

        ranked_reader = self._ranked_writer.to_reader()
        return Evaluator(
            detailed_reader=detailed_reader,
            ranked_reader=ranked_reader,
            info=info,
            index_to_label=index_to_label,
            number_of_groundtruths_per_label=number_of_groundtruths_per_label,
            path=self._path,
        )
