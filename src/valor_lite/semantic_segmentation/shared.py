from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray
from pyarrow import DataType

from valor_lite.cache import (
    FileCacheReader,
    FileCacheWriter,
    MemoryCacheReader,
)


@dataclass
class EvaluatorInfo:
    number_of_rows: int = 0
    number_of_datums: int = 0
    number_of_labels: int = 0
    number_of_pixels: int = 0
    number_of_groundtruth_pixels: int = 0
    number_of_prediction_pixels: int = 0
    datum_metadata_fields: list[tuple[str, DataType]] | None = None
    groundtruth_metadata_fields: list[tuple[str, DataType]] | None = None
    prediction_metadata_fields: list[tuple[str, DataType]] | None = None


class Base:
    @staticmethod
    def _generate_cache_path(path: str | Path) -> Path:
        return Path(path) / "counts"

    @staticmethod
    def _generate_metadata_path(path: str | Path) -> Path:
        return Path(path) / "metadata.json"

    @staticmethod
    def _generate_schema(
        datum_metadata_fields: list[tuple[str, DataType]] | None,
        groundtruth_metadata_fields: list[tuple[str, DataType]] | None,
        prediction_metadata_fields: list[tuple[str, DataType]] | None,
    ) -> pa.Schema:
        datum_metadata_fields = (
            datum_metadata_fields if datum_metadata_fields else []
        )
        groundtruth_metadata_fields = (
            groundtruth_metadata_fields if groundtruth_metadata_fields else []
        )
        prediction_metadata_fields = (
            prediction_metadata_fields if prediction_metadata_fields else []
        )
        return pa.schema(
            [
                ("datum_uid", pa.string()),
                ("datum_id", pa.int64()),
                *datum_metadata_fields,
                # groundtruth
                ("gt_label", pa.string()),
                ("gt_label_id", pa.int64()),
                *groundtruth_metadata_fields,
                # prediction
                ("pd_label", pa.string()),
                ("pd_label_id", pa.int64()),
                *prediction_metadata_fields,
                # pair
                ("count", pa.uint64()),
            ]
        )

    @staticmethod
    def _encode_metadata_fields(
        datum_metadata_fields: list[tuple[str, DataType]] | None,
        groundtruth_metadata_fields: list[tuple[str, DataType]] | None,
        prediction_metadata_fields: list[tuple[str, DataType]] | None,
    ) -> dict[str, dict[str, str]]:
        datum_metadata_fields = (
            datum_metadata_fields if datum_metadata_fields else []
        )
        groundtruth_metadata_fields = (
            groundtruth_metadata_fields if groundtruth_metadata_fields else []
        )
        prediction_metadata_fields = (
            prediction_metadata_fields if prediction_metadata_fields else []
        )
        return {
            "datum": {k: str(v) for k, v in datum_metadata_fields},
            "groundtruth": {k: str(v) for k, v in groundtruth_metadata_fields},
            "prediction": {k: str(v) for k, v in prediction_metadata_fields},
        }

    @staticmethod
    def _decode_metadata_fields(
        encoded_metadata_fields: dict[str, dict[str, str]]
    ) -> tuple[
        list[tuple[str, DataType]],
        list[tuple[str, DataType]],
        list[tuple[str, DataType]],
    ]:
        datum_metadata_fields = [
            (k, v) for k, v in encoded_metadata_fields["datum"].items()
        ]
        groundtruth_metadata_fields = [
            (k, v) for k, v in encoded_metadata_fields["groundtruth"].items()
        ]
        prediction_metadata_fields = [
            (k, v) for k, v in encoded_metadata_fields["prediction"].items()
        ]
        return (
            datum_metadata_fields,
            groundtruth_metadata_fields,
            prediction_metadata_fields,
        )

    @staticmethod
    def _generate_meta(
        reader: MemoryCacheReader | FileCacheReader,
        labels_override: dict[int, str] | None,
    ) -> tuple[dict[int, str], NDArray[np.uint64], EvaluatorInfo]:
        """
        Generate cache statistics.

        Parameters
        ----------
        reader : MemoryCacheReader | FileCacheReader
            Valor cache reader.
        labels_override : dict[int, str], optional
            Optional labels override. Use when operating over filtered data.

        Returns
        -------
        labels : dict[int, str]
            Mapping of label ID's to label values.
        confusion_matrix : NDArray[np.uint64]
            Array of size (n_labels + 1, n_labels + 1) containing pair counts.
        info : EvaluatorInfo
            Evaluator cache details.
        """
        labels = labels_override if labels_override else {}
        info = EvaluatorInfo()

        for tbl in reader.iterate_tables():
            columns = (
                "datum_id",
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

            # get gt labels
            gt_label_ids = ids[:, 1]
            gt_label_ids, gt_indices = np.unique(
                gt_label_ids, return_index=True
            )
            gt_labels = tbl["gt_label"].take(gt_indices).to_pylist()
            gt_labels = dict(zip(gt_label_ids.astype(int).tolist(), gt_labels))
            gt_labels.pop(-1, None)
            labels.update(gt_labels)

            # get pd labels
            pd_label_ids = ids[:, 2]
            pd_label_ids, pd_indices = np.unique(
                pd_label_ids, return_index=True
            )
            pd_labels = tbl["pd_label"].take(pd_indices).to_pylist()
            pd_labels = dict(zip(pd_label_ids.astype(int).tolist(), pd_labels))
            pd_labels.pop(-1, None)
            labels.update(pd_labels)

        # post-process
        labels.pop(-1, None)

        # create confusion matrix
        n_labels = len(labels)
        matrix = np.zeros((n_labels + 1, n_labels + 1), dtype=np.uint64)
        for tbl in reader.iterate_tables():
            columns = (
                "datum_id",
                "gt_label_id",
                "pd_label_id",
            )
            ids = np.column_stack(
                [tbl[col].to_numpy() for col in columns]
            ).astype(np.int64)
            counts = tbl["count"].to_numpy()

            mask_null_gts = ids[:, 1] == -1
            mask_null_pds = ids[:, 2] == -1
            matrix[0, 0] += counts[mask_null_gts & mask_null_pds].sum()
            for idx in range(n_labels):
                mask_gts = ids[:, 1] == idx
                for pidx in range(n_labels):
                    mask_pds = ids[:, 2] == pidx
                    matrix[idx + 1, pidx + 1] += counts[
                        mask_gts & mask_pds
                    ].sum()

                mask_unmatched_gts = mask_gts & mask_null_pds
                matrix[idx + 1, 0] += counts[mask_unmatched_gts].sum()
                mask_unmatched_pds = mask_null_gts & (ids[:, 2] == idx)
                matrix[0, idx + 1] += counts[mask_unmatched_pds].sum()

        # complete info object
        info.number_of_labels = len(labels)
        info.number_of_pixels = matrix.sum()
        info.number_of_groundtruth_pixels = matrix[1:, :].sum()
        info.number_of_prediction_pixels = matrix[:, 1:].sum()

        return labels, matrix, info

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
        cache_path = cls._generate_cache_path(path)
        metadata_path = cls._generate_metadata_path(path)
        FileCacheWriter.delete(cache_path)
        if metadata_path.exists() and metadata_path.is_file():
            metadata_path.unlink()
        path.rmdir()
