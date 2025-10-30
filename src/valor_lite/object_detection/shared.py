from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray

from valor_lite.cache import (
    DataType,
    FileCacheReader,
    FileCacheWriter,
    MemoryCacheReader,
)


@dataclass
class EvaluatorInfo:
    number_of_datums: int = 0
    number_of_groundtruth_annotations: int = 0
    number_of_prediction_annotations: int = 0
    number_of_labels: int = 0
    number_of_rows: int = 0
    datum_metadata_types: dict[str, DataType] | None = None
    groundtruth_metadata_types: dict[str, DataType] | None = None
    prediction_metadata_types: dict[str, DataType] | None = None


class Base:
    @staticmethod
    def _generate_detailed_cache_path(path: str | Path) -> Path:
        return Path(path) / "detailed"

    @staticmethod
    def _generate_ranked_cache_path(path: str | Path) -> Path:
        return Path(path) / "ranked"

    @staticmethod
    def _generate_temporary_cache_path(path: str | Path) -> Path:
        return Path(path) / "tmp"

    @staticmethod
    def _generate_metadata_path(path: str | Path) -> Path:
        return Path(path) / "metadata.json"

    @staticmethod
    def _generate_detailed_schema(
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
    def _generate_ranked_schema(
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

    @staticmethod
    def _generate_meta(
        reader: MemoryCacheReader | FileCacheReader,
        labels_override: dict[int, str] | None = None,
    ) -> tuple[dict[int, str], NDArray[np.uint64], EvaluatorInfo]:
        """
        Generate cache statistics.

        Parameters
        ----------
        dataset : Dataset
            Valor cache.
        labels_override : dict[int, str], optional
            Optional labels override. Use when operating over filtered data.

        Returns
        -------
        labels : dict[int, str]
            Mapping of label ID's to label values.
        number_of_groundtruths_per_label : NDArray[np.uint64]
            Array of size (n_labels,) containing ground truth counts.
        info : EvaluatorInfo
            Evaluator cache details.
        """
        gt_counts_per_lbl = defaultdict(int)
        labels = labels_override if labels_override else {}
        info = EvaluatorInfo()

        for tbl in reader.iterate_tables():
            columns = (
                "datum_id",
                "gt_id",
                "pd_id",
                "gt_label_id",
                "pd_label_id",
            )
            ids = np.column_stack(
                [tbl[col].to_numpy() for col in columns]
            ).astype(np.int64)

            # count number of rows
            info.number_of_rows += int(tbl.shape[0])

            # count unique datums
            datum_ids = np.unique(ids[:, 0])
            info.number_of_datums += int(datum_ids.size)

            # count unique groundtruths
            gt_ids = ids[:, 1]
            gt_ids = np.unique(gt_ids[gt_ids >= 0])
            info.number_of_groundtruth_annotations += int(gt_ids.shape[0])

            # count unique predictions
            pd_ids = ids[:, 2]
            pd_ids = np.unique(pd_ids[pd_ids >= 0])
            info.number_of_prediction_annotations += int(pd_ids.shape[0])

            # get gt labels
            gt_label_ids = ids[:, 3]
            gt_label_ids, gt_indices = np.unique(
                gt_label_ids, return_index=True
            )
            gt_labels = tbl["gt_label"].take(gt_indices).to_pylist()
            gt_labels = dict(zip(gt_label_ids.astype(int).tolist(), gt_labels))
            gt_labels.pop(-1, None)
            labels.update(gt_labels)

            # get pd labels
            pd_label_ids = ids[:, 4]
            pd_label_ids, pd_indices = np.unique(
                pd_label_ids, return_index=True
            )
            pd_labels = tbl["pd_label"].take(pd_indices).to_pylist()
            pd_labels = dict(zip(pd_label_ids.astype(int).tolist(), pd_labels))
            pd_labels.pop(-1, None)
            labels.update(pd_labels)

            # count gts per label
            gts = ids[:, (1, 3)].astype(np.int64)
            unique_ann = np.unique(gts[gts[:, 0] >= 0], axis=0)
            unique_labels, label_counts = np.unique(
                unique_ann[:, 1], return_counts=True
            )
            for label_id, count in zip(unique_labels, label_counts):
                gt_counts_per_lbl[int(label_id)] += int(count)

        # post-process
        labels.pop(-1, None)

        # complete info object
        info.number_of_labels = len(labels)

        # convert gt counts to numpy
        number_of_groundtruths_per_label = np.zeros(
            len(labels), dtype=np.uint64
        )
        for k, v in gt_counts_per_lbl.items():
            number_of_groundtruths_per_label[int(k)] = v

        return labels, number_of_groundtruths_per_label, info

    @classmethod
    def delete_at_path(cls, path: str | Path):
        """
        Delete file-based cache at the given path.

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
