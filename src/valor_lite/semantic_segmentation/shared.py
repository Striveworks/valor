from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa

from valor_lite.cache import FileCacheReader, MemoryCacheReader


@dataclass
class EvaluatorInfo:
    number_of_rows: int = 0
    number_of_datums: int = 0
    number_of_labels: int = 0
    number_of_pixels: int = 0
    number_of_groundtruth_pixels: int = 0
    number_of_prediction_pixels: int = 0
    metadata_fields: list[tuple[str, str | pa.DataType]] | None = None


def generate_cache_path(path: str | Path) -> Path:
    """Generate cache path from parent directory."""
    return Path(path) / "counts"


def generate_metadata_path(path: str | Path) -> Path:
    """Generate metadata path from parent directory."""
    return Path(path) / "metadata.json"


def generate_schema(
    metadata_fields: list[tuple[str, str | pa.DataType]] | None
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
    metadata_fields: list[tuple[str, str | pa.DataType]] | None
) -> dict[str, str]:
    """Encode metadata fields into JSON format."""
    metadata_fields = metadata_fields if metadata_fields else []
    return {k: str(v) for k, v in metadata_fields}


def decode_metadata_fields(
    encoded_metadata_fields: dict[str, str]
) -> list[tuple[str, str | pa.DataType]]:
    """Decode metadata fields from JSON format."""
    return [(k, v) for k, v in encoded_metadata_fields.items()]


def generate_meta(
    reader: MemoryCacheReader | FileCacheReader,
    labels_override: dict[int, str] | None,
) -> tuple[dict[int, str], EvaluatorInfo]:
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

    # complete info object
    info.number_of_labels = len(labels)
    info.number_of_pixels = 0
    info.number_of_groundtruth_pixels = 0
    info.number_of_prediction_pixels = 0

    for tbl in reader.iterate_tables():
        columns = (
            "datum_id",
            "gt_label_id",
            "pd_label_id",
        )
        ids = np.column_stack([tbl[col].to_numpy() for col in columns]).astype(
            np.int64
        )
        counts = tbl["count"].to_numpy().astype(np.uint64)

        # total count
        info.number_of_pixels += int(counts.sum())

        # gt pixel count
        indices = np.where(ids[:, 1] > -0.5)[0]
        info.number_of_groundtruth_pixels += int(counts[indices].sum())

        # pd pixel count
        indices = np.where(ids[:, 2] > -0.5)[0]
        info.number_of_prediction_pixels += int(counts[indices].sum())

    return labels, info
