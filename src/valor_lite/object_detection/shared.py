from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from numpy.typing import NDArray

from valor_lite.cache import FileCacheReader, MemoryCacheReader


@dataclass
class EvaluatorInfo:
    number_of_datums: int = 0
    number_of_groundtruth_annotations: int = 0
    number_of_prediction_annotations: int = 0
    number_of_labels: int = 0
    number_of_rows: int = 0
    metadata_fields: list[tuple[str, str | pa.DataType]] | None = None


def generate_detailed_cache_path(path: str | Path) -> Path:
    return Path(path) / "detailed"


def generate_ranked_cache_path(path: str | Path) -> Path:
    return Path(path) / "ranked"


def generate_temporary_cache_path(path: str | Path) -> Path:
    return Path(path) / "tmp"


def generate_metadata_path(path: str | Path) -> Path:
    return Path(path) / "metadata.json"


def generate_detailed_schema(
    metadata_fields: list[tuple[str, str | pa.DataType]] | None
) -> pa.Schema:
    metadata_fields = metadata_fields if metadata_fields else []
    reserved_fields = [
        ("datum_uid", pa.string()),
        ("datum_id", pa.int64()),
        # groundtruth
        ("gt_uid", pa.string()),
        ("gt_id", pa.int64()),
        ("gt_label", pa.string()),
        ("gt_label_id", pa.int64()),
        # prediction
        ("pd_uid", pa.string()),
        ("pd_id", pa.int64()),
        ("pd_label", pa.string()),
        ("pd_label_id", pa.int64()),
        ("pd_score", pa.float64()),
        # pair
        ("iou", pa.float64()),
    ]

    # validate
    reserved_field_names = {f[0] for f in reserved_fields}
    metadata_field_names = {f[0] for f in metadata_fields}
    if conflicting := reserved_field_names & metadata_field_names:
        raise ValueError(
            f"metadata fields {conflicting} conflict with reserved fields"
        )

    return pa.schema(reserved_fields + metadata_fields)


def generate_ranked_schema() -> pa.Schema:
    reserved_fields = [
        ("datum_uid", pa.string()),
        ("datum_id", pa.int64()),
        # groundtruth
        ("gt_id", pa.int64()),
        ("gt_label_id", pa.int64()),
        # prediction
        ("pd_id", pa.int64()),
        ("pd_label_id", pa.int64()),
        ("pd_score", pa.float64()),
        # pair
        ("iou", pa.float64()),
        ("high_score", pa.bool_()),
        ("iou_prev", pa.float64()),
    ]
    return pa.schema(reserved_fields)


def encode_metadata_fields(
    metadata_fields: list[tuple[str, str | pa.DataType]] | None
) -> dict[str, str]:
    metadata_fields = metadata_fields if metadata_fields else []
    return {k: str(v) for k, v in metadata_fields}


def decode_metadata_fields(
    encoded_metadata_fields: dict[str, str]
) -> list[tuple[str, str]]:
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
    n_dts, n_gts, n_pds = 0, 0, 0
    for tbl in reader.iterate_tables(filter=datums):
        # count datums
        n_dts += int(np.unique(tbl["datum_id"].to_numpy()).shape[0])

        # count groundtruths
        if groundtruths is not None:
            gts = tbl.filter(groundtruths)["gt_id"].to_numpy()
        else:
            gts = tbl["gt_id"].to_numpy()
        n_gts += int(np.unique(gts[gts >= 0]).shape[0])

        # count predictions
        if predictions is not None:
            pds = tbl.filter(predictions)["pd_id"].to_numpy()
        else:
            pds = tbl["pd_id"].to_numpy()
        n_pds += int(np.unique(pds[pds >= 0]).shape[0])

    return n_dts, n_gts, n_pds


def extract_groundtruth_count_per_label(
    reader: MemoryCacheReader | FileCacheReader,
    number_of_labels: int,
    datums: pc.Expression | None = None,
    groundtruths: pc.Expression | None = None,
) -> NDArray[np.uint64]:
    expr = None
    if datums is not None and groundtruths is not None:
        expr = datums & groundtruths
    elif datums is not None:
        expr = datums
    elif groundtruths is not None:
        expr = groundtruths

    gt_counts_per_lbl = np.zeros(number_of_labels, dtype=np.uint64)
    for gts in reader.iterate_arrays(
        numeric_columns=["gt_id", "gt_label_id"],
        filter=expr,
    ):
        # count gts per label
        unique_ann = np.unique(gts[gts[:, 0] >= 0], axis=0)
        unique_labels, label_counts = np.unique(
            unique_ann[:, 1], return_counts=True
        )
        for label_id, count in zip(unique_labels, label_counts):
            gt_counts_per_lbl[int(label_id)] += int(count)

    return gt_counts_per_lbl
