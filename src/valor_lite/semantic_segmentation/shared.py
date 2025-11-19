from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

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


def extract_labels(
    reader: MemoryCacheReader | FileCacheReader,
    index_to_label_override: dict[int, str] | None = None,
) -> dict[int, str]:
    if index_to_label_override is not None:
        return index_to_label_override

    index_to_label = {}
    for tbl in reader.iterate_tables(
        columns=[
            "gt_label_id",
            "gt_label",
            "pd_label_id",
            "pd_label",
        ]
    ):

        # get gt labels
        gt_label_ids = tbl["gt_label_id"].to_numpy()
        gt_label_ids, gt_indices = np.unique(gt_label_ids, return_index=True)
        gt_labels = tbl["gt_label"].take(gt_indices).to_pylist()
        gt_labels = dict(zip(gt_label_ids.astype(int).tolist(), gt_labels))
        gt_labels.pop(-1, None)
        index_to_label.update(gt_labels)

        # get pd labels
        pd_label_ids = tbl["pd_label_id"].to_numpy()
        pd_label_ids, pd_indices = np.unique(pd_label_ids, return_index=True)
        pd_labels = tbl["pd_label"].take(pd_indices).to_pylist()
        pd_labels = dict(zip(pd_label_ids.astype(int).tolist(), pd_labels))
        pd_labels.pop(-1, None)
        index_to_label.update(pd_labels)

    return index_to_label


def extract_counts(
    reader: MemoryCacheReader | FileCacheReader,
    datums: pc.Expression | None = None,
    groundtruths: pc.Expression | None = None,
    predictions: pc.Expression | None = None,
):
    n_dts, n_total, n_gts, n_pds = 0, 0, 0, 0
    for tbl in reader.iterate_tables(filter=datums):

        # count datums
        n_dts += int(np.unique(tbl["datum_id"].to_numpy()).shape[0])

        # count pixels
        n_total += int(tbl["count"].to_numpy().sum())

        # count groundtruth pixels
        gt_tbl = tbl
        gt_expr = pc.field("gt_label_id") >= 0
        if groundtruths is not None:
            gt_expr &= groundtruths
        gt_tbl = tbl.filter(gt_expr)
        n_gts += int(gt_tbl["count"].to_numpy().sum())

        # count prediction pixels
        pd_tbl = tbl
        pd_expr = pc.field("pd_label_id") >= 0
        if predictions is not None:
            pd_expr &= predictions
        pd_tbl = tbl.filter(pd_expr)
        n_pds += int(pd_tbl["count"].to_numpy().sum())

    return n_dts, n_total, n_gts, n_pds
