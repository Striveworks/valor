from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from numpy.typing import NDArray

from valor_lite.cache import FileCacheReader, MemoryCacheReader


@dataclass
class EvaluatorInfo:
    number_of_rows: int = 0
    number_of_datums: int = 0
    number_of_labels: int = 0
    metadata_fields: list[tuple[str, str]] | None = None


def generate_cache_path(path: str | Path) -> Path:
    return Path(path) / "cache"


def generate_intermediate_cache_path(path: str | Path) -> Path:
    return Path(path) / "intermediate"


def generate_roc_curve_cache_path(path: str | Path) -> Path:
    return Path(path) / "roc_curve"


def generate_metadata_path(path: str | Path) -> Path:
    return Path(path) / "metadata.json"


def generate_schema(
    metadata_fields: list[tuple[str, str | pa.DataType]] | None
) -> pa.Schema:
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
        ("pd_score", pa.float64()),
        ("pd_winner", pa.bool_()),
        # pair
        ("match", pa.bool_()),
    ]

    # validate
    reserved_field_names = {f[0] for f in reserved_fields}
    metadata_field_names = {f[0] for f in metadata_fields}
    if conflicting := reserved_field_names & metadata_field_names:
        raise ValueError(
            f"metadata fields {conflicting} conflict with reserved fields"
        )
    return pa.schema(reserved_fields + metadata_fields)


def generate_intermediate_schema() -> pa.Schema:
    return pa.schema(
        [
            ("pd_label_id", pa.int64()),
            ("pd_score", pa.float64()),
            ("match", pa.bool_()),
        ]
    )


def generate_roc_curve_schema() -> pa.Schema:
    return pa.schema(
        [
            ("pd_label_id", pa.int64()),
            ("cumulative_fp", pa.uint64()),
            ("cumulative_tp", pa.uint64()),
        ]
    )


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
):
    n_dts = 0
    for tbl in reader.iterate_tables(filter=datums):
        n_dts += int(np.unique(tbl["datum_id"].to_numpy()).shape[0])
    return n_dts


def extract_groundtruth_count_per_label(
    reader: MemoryCacheReader | FileCacheReader,
    number_of_labels: int,
    datums: pc.Expression | None = None,
    groundtruths: pc.Expression | None = None,
    predictions: pc.Expression | None = None,
) -> NDArray[np.uint64]:

    # count ground truth and prediction label occurences
    label_counts = np.zeros((number_of_labels, 2), dtype=np.uint64)
    for tbl in reader.iterate_tables(filter=datums):

        # count unique gt labels
        gt_expr = pc.field("gt_label_id") >= 0
        if groundtruths is not None:
            gt_expr &= groundtruths
        gt_tbl = tbl.filter(gt_expr)
        gt_ids = np.column_stack(
            [gt_tbl[col].to_numpy() for col in ["datum_id", "gt_label_id"]]
        ).astype(np.int64)
        unique_gts = np.unique(gt_ids, axis=0)
        unique_gt_labels, gt_label_counts = np.unique(
            unique_gts[:, 1], return_counts=True
        )
        label_counts[unique_gt_labels, 0] += gt_label_counts.astype(np.uint64)

        # count unique pd labels
        pd_expr = pc.field("pd_label_id") >= 0
        if predictions is not None:
            pd_expr &= predictions
        pd_tbl = tbl.filter(pd_expr)
        pd_ids = np.column_stack(
            [pd_tbl[col].to_numpy() for col in ["datum_id", "pd_label_id"]]
        ).astype(np.int64)
        unique_pds = np.unique(pd_ids, axis=0)
        unique_pd_labels, pd_label_counts = np.unique(
            unique_pds[:, 1], return_counts=True
        )
        label_counts[unique_pd_labels, 1] += pd_label_counts.astype(np.uint64)

    return label_counts


# @staticmethod
# def generate_meta(
#     reader: MemoryCacheReader | FileCacheReader,
#     index_to_label_override: dict[int, str] | None,
# ) -> tuple[dict[int, str], NDArray[np.uint64], EvaluatorInfo]:
#     """
#     Generate cache statistics.

#     Parameters
#     ----------
#     dataset : Dataset
#         Valor cache.
#     index_to_label_override : dict[int, str], optional
#         Optional labels override. Use when operating over filtered data.

#     Returns
#     -------
#     labels : dict[int, str]
#         Mapping of label ID's to label values.
#     label_counts : NDArray[np.uint64]
#         Array of size (n_labels, 2) containing counts of ground truths and predictions per label.
#     info : EvaluatorInfo
#         Evaluator cache details.
#     """
#     labels = index_to_label_override if index_to_label_override else {}
#     info = EvaluatorInfo()

#     for tbl in reader.iterate_tables():
#         columns = (
#             "datum_id",
#             "gt_label_id",
#             "pd_label_id",
#         )
#         ids = np.column_stack([tbl[col].to_numpy() for col in columns]).astype(
#             np.int64
#         )

#         # count number of rows
#         info.number_of_rows += int(tbl.shape[0])

#         # count unique datums
#         datum_ids = np.unique(ids[:, 0])
#         info.number_of_datums += int(datum_ids.size)

#         # get gt labels
#         gt_label_ids = ids[:, 1]
#         gt_label_ids, gt_indices = np.unique(gt_label_ids, return_index=True)
#         gt_labels = tbl["gt_label"].take(gt_indices).to_pylist()
#         gt_labels = dict(zip(gt_label_ids.astype(int).tolist(), gt_labels))
#         gt_labels.pop(-1, None)
#         labels.update(gt_labels)

#         # get pd labels
#         pd_label_ids = ids[:, 2]
#         pd_label_ids, pd_indices = np.unique(pd_label_ids, return_index=True)
#         pd_labels = tbl["pd_label"].take(pd_indices).to_pylist()
#         pd_labels = dict(zip(pd_label_ids.astype(int).tolist(), pd_labels))
#         pd_labels.pop(-1, None)
#         labels.update(pd_labels)

#     # post-process
#     labels.pop(-1, None)

#     # count ground truth and prediction label occurences
#     n_labels = len(labels)
#     label_counts = np.zeros((n_labels, 2), dtype=np.uint64)
#     for tbl in reader.iterate_tables():
#         columns = (
#             "datum_id",
#             "gt_label_id",
#             "pd_label_id",
#         )
#         ids = np.column_stack([tbl[col].to_numpy() for col in columns]).astype(
#             np.int64
#         )

#         # count unique gt labels
#         unique_gts = np.unique(ids[:, (0, 1)], axis=0)
#         unique_gt_labels, gt_label_counts = np.unique(
#             unique_gts[:, 1], return_counts=True
#         )
#         label_counts[unique_gt_labels, 0] += gt_label_counts.astype(np.uint64)

#         # count unique pd labels
#         unique_pds = np.unique(ids[:, (0, 2)], axis=0)
#         unique_pd_labels, pd_label_counts = np.unique(
#             unique_pds[:, 1], return_counts=True
#         )
#         label_counts[unique_pd_labels, 1] += pd_label_counts.astype(np.uint64)

#     # complete info object
#     info.number_of_labels = len(labels)

#     return labels, label_counts, info
