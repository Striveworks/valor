import heapq
import json
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray
from tqdm import tqdm

from valor_lite.cache import (
    CacheReader,
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
    rank_table,
)
from valor_lite.object_detection.evaluator import Evaluator
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

        # internal state
        self._labels = {}
        self._datum_count = 0
        self._groundtruth_count = 0
        self._prediction_count = 0

    @property
    def path(self) -> Path:
        return self._path

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
        delete_if_exists: bool = False,
    ):
        path = Path(path)
        if delete_if_exists:
            cls.delete(path)

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
    def delete(cls, path: str | Path):
        path = Path(path)
        if not path.exists():
            return
        CacheWriter.delete(cls._generate_detailed_cache_path(path))
        CacheWriter.delete(cls._generate_ranked_cache_path(path))
        metadata_path = cls._generate_metadata_path(path)
        if metadata_path.exists() and metadata_path.is_file():
            metadata_path.unlink()
        path.rmdir()

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
        self._cache.flush()
        if self._cache.dataset.count_rows() == 0:
            raise EmptyCacheError()

        detailed_cache = CacheReader.load(
            self._generate_detailed_cache_path(self.path)
        )

        # build evaluator meta
        (
            index_to_label,
            number_of_groundtruths_per_label,
            info,
        ) = Evaluator.generate_meta(
            detailed_cache.dataset, index_to_label_override
        )

        # read config
        metadata_path = self._generate_metadata_path(self.path)
        with open(metadata_path, "r") as f:
            types = json.load(f)
            info.datum_metadata_types = types["datum_metadata_types"]
            info.groundtruth_metadata_types = types[
                "groundtruth_metadata_types"
            ]
            info.prediction_metadata_types = types["prediction_metadata_types"]

        # create ranked cache schema
        annotation_metadata_keys = {
            *(
                set(info.groundtruth_metadata_types.keys())
                if info.groundtruth_metadata_types
                else {}
            ),
            *(
                set(info.prediction_metadata_types.keys())
                if info.prediction_metadata_types
                else {}
            ),
        }
        pruned_schema = pa.schema(
            [
                field
                for field in detailed_cache.schema
                if field.name not in annotation_metadata_keys
            ]
        )
        ranked_schema = pruned_schema.append(
            pa.field("iou_prev", pa.float64())
        )
        ranked_schema = ranked_schema.append(
            pa.field("high_score", pa.bool_())
        )

        n_labels = len(index_to_label)

        with CacheWriter.create(
            path=self._generate_ranked_cache_path(self.path),
            schema=ranked_schema,
            batch_size=detailed_cache.batch_size,
            rows_per_file=detailed_cache.rows_per_file,
            compression=detailed_cache.compression,
        ) as ranked_cache:
            if detailed_cache.num_dataset_files == 1:
                pf = pq.ParquetFile(detailed_cache.dataset_files[0])
                tbl = pf.read()
                ranked_tbl = rank_table(tbl, n_labels)
                ranked_cache.write_table(ranked_tbl)
            else:
                pruned_detailed_columns = [
                    field.name for field in pruned_schema
                ]
                with tempfile.TemporaryDirectory() as tmpdir:

                    # rank individual files
                    tmpfiles = []
                    for idx, fragment in enumerate(
                        detailed_cache.dataset.get_fragments()
                    ):
                        fragment_path = Path(tmpdir) / f"{idx:06d}.parquet"
                        tbl = fragment.to_table(
                            columns=pruned_detailed_columns
                        )
                        ranked_tbl = rank_table(tbl, n_labels)
                        pq.write_table(ranked_tbl, fragment_path)
                        tmpfiles.append(fragment_path)

                    def generate_heap_item(batches, batch_idx, row_idx):
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
                    for batch_idx, batch_path in enumerate(tmpfiles):
                        pf = pq.ParquetFile(batch_path)
                        batch_iter = pf.iter_batches(batch_size=batch_size)
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
                        ranked_cache.write_batch(row_table)
                        row_idx += 1
                        if row_idx < len(batches[batch_idx]):
                            heapq.heappush(
                                heap,
                                generate_heap_item(
                                    batches, batch_idx, row_idx
                                ),
                            )
                        else:
                            batches[batch_idx] = next(
                                batch_iterators[batch_idx], None
                            )
                            if (
                                batches[batch_idx] is not None
                                and len(batches[batch_idx]) > 0
                            ):
                                heapq.heappush(
                                    heap,
                                    generate_heap_item(batches, batch_idx, 0),
                                )

        ranked_cache = CacheReader.load(
            self._generate_ranked_cache_path(self.path)
        )
        return Evaluator(
            path=self.path,
            detailed_cache=detailed_cache,
            ranked_cache=ranked_cache,
            info=info,
            index_to_label=index_to_label,
            number_of_groundtruths_per_label=number_of_groundtruths_per_label,
        )
