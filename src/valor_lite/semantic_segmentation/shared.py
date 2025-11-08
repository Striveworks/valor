from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray
from pyarrow import DataType

from valor_lite.cache import FileCacheReader, MemoryCacheReader


@dataclass
class EvaluatorInfo:
    number_of_rows: int = 0
    number_of_datums: int = 0
    number_of_labels: int = 0
    number_of_pixels: int = 0
    number_of_groundtruth_pixels: int = 0
    number_of_prediction_pixels: int = 0
    metadata_fields: list[tuple[str, DataType]] | None = None


def generate_cache_path(path: str | Path) -> Path:
    """Generate cache path from parent directory."""
    return Path(path) / "counts"


def generate_metadata_path(path: str | Path) -> Path:
    """Generate metadata path from parent directory."""
    return Path(path) / "metadata.json"


def generate_schema(
    metadata_fields: list[tuple[str, DataType]] | None
) -> pa.Schema:
    """Generate PyArrow schema from metadata fields."""

    metadata_fields = metadata_fields if metadata_fields else []
    reserved_fields = [
        ("datum_uid", pa.string()),
        ("datum_id", pa.int64()),
        # groundtruth
        ("gt_label", pa.string()),
        ("gt_label_id", pa.int64()),
        # prediction
        ("pd_label", pa.string()),
        ("pd_label_id", pa.int64()),
        # pair
        ("count", pa.uint64()),
    ]

    # validate
    reserved_field_names = {f[0] for f in reserved_fields}
    metadata_field_names = {f[0] for f in metadata_fields}
    if conflicting := reserved_field_names & metadata_field_names:
        raise ValueError(
            f"metadata fields {conflicting} conflict with reserved fields"
        )

    return pa.schema(
        [
            *reserved_fields,
            *metadata_fields,
        ]
    )


def encode_metadata_fields(
    metadata_fields: list[tuple[str, str | DataType]] | None
) -> dict[str, str]:
    """Encode metadata fields into JSON format."""
    metadata_fields = metadata_fields if metadata_fields else []
    return {k: str(v) for k, v in metadata_fields}


def decode_metadata_fields(
    encoded_metadata_fields: dict[str, str]
) -> list[tuple[str, str | DataType]]:
    """Decode metadata fields from JSON format."""
    return [(k, v) for k, v in encoded_metadata_fields.items()]


def generate_meta(
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
        ids = np.column_stack([tbl[col].to_numpy() for col in columns]).astype(
            np.int64
        )

        # count number of rows
        info.number_of_rows += int(tbl.shape[0])

        # count unique datums
        datum_ids = np.unique(ids[:, 0])
        info.number_of_datums += int(datum_ids.size)

        # get gt labels
        gt_label_ids = ids[:, 1]
        gt_label_ids, gt_indices = np.unique(gt_label_ids, return_index=True)
        gt_labels = tbl["gt_label"].take(gt_indices).to_pylist()
        gt_labels = dict(zip(gt_label_ids.astype(int).tolist(), gt_labels))
        gt_labels.pop(-1, None)
        labels.update(gt_labels)

        # get pd labels
        pd_label_ids = ids[:, 2]
        pd_label_ids, pd_indices = np.unique(pd_label_ids, return_index=True)
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
        ids = np.column_stack([tbl[col].to_numpy() for col in columns]).astype(
            np.int64
        )
        counts = tbl["count"].to_numpy()

        mask_null_gts = ids[:, 1] == -1
        mask_null_pds = ids[:, 2] == -1
        matrix[0, 0] += counts[mask_null_gts & mask_null_pds].sum()
        for idx in range(n_labels):
            mask_gts = ids[:, 1] == idx
            for pidx in range(n_labels):
                mask_pds = ids[:, 2] == pidx
                matrix[idx + 1, pidx + 1] += counts[mask_gts & mask_pds].sum()

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
