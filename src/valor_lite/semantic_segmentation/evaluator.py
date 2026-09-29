from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from numpy.typing import NDArray

from valor_lite.cache import (
    FileCacheReader,
    FileCacheWriter,
    MemoryCacheReader,
    MemoryCacheWriter,
)
from valor_lite.exceptions import EmptyCacheError
from valor_lite.semantic_segmentation.computation import compute_metrics
from valor_lite.semantic_segmentation.metric import MetricType
from valor_lite.semantic_segmentation.shared import (
    EvaluatorInfo,
    decode_metadata_fields,
    encode_metadata_fields,
    ensure_annotation_ids,
    extract_counts,
    extract_labels,
    generate_cache_path,
    generate_metadata_path,
    generate_schema,
)
from valor_lite.semantic_segmentation.utilities import (
    unpack_precision_recall_iou_into_metric_lists,
)


def _annotation_mask(
    table: pa.Table, expression: pc.Expression | None, side: str
) -> NDArray[np.bool_]:
    """Select whole original annotations, even after their labels collide."""
    if expression is None:
        return np.ones(table.num_rows, dtype=np.bool_)

    selected = table.filter(expression & (pc.field(f"{side}_label_id") >= 0))
    annotation_column = f"__valor_{side}_annotation_id"

    def keys(tbl: pa.Table):
        return np.rec.fromarrays(
            [
                tbl["datum_id"].to_numpy(),
                tbl[annotation_column].to_numpy(),
            ],
            dtype=[("datum", np.int64), ("annotation", np.int64)],
        )

    return np.isin(keys(table), keys(selected))


class Builder:
    def __init__(
        self,
        writer: MemoryCacheWriter | FileCacheWriter,
        metadata_fields: list[tuple[str, str | pa.DataType]] | None = None,
    ):
        self._writer = writer
        self._metadata_fields = metadata_fields

    @classmethod
    def in_memory(
        cls,
        batch_size: int = 10_000,
        metadata_fields: list[tuple[str, str | pa.DataType]] | None = None,
    ):
        """
        Create an in-memory evaluator cache.

        Parameters
        ----------
        batch_size : int, default=10_000
            The target number of rows to buffer before writing to the cache. Defaults to 10_000.
        metadata_fields : list[tuple[str, str | pa.DataType]], optional
            Optional metadata field definitions.
        """
        # create cache
        writer = MemoryCacheWriter.create(
            schema=generate_schema(metadata_fields),
            batch_size=batch_size,
        )
        return cls(
            writer=writer,
            metadata_fields=metadata_fields,
        )

    @classmethod
    def persistent(
        cls,
        path: str | Path,
        batch_size: int = 10_000,
        rows_per_file: int = 100_000,
        compression: str = "snappy",
        metadata_fields: list[tuple[str, str | pa.DataType]] | None = None,
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
        metadata_fields : list[tuple[str, str | pa.DataType]], optional
            Optional metadata field definitions.
        """
        path = Path(path)

        # create cache
        writer = FileCacheWriter.create(
            path=generate_cache_path(path),
            schema=generate_schema(metadata_fields),
            batch_size=batch_size,
            rows_per_file=rows_per_file,
            compression=compression,
        )

        # write metadata
        metadata_path = generate_metadata_path(path)
        with open(metadata_path, "w") as f:
            encoded_types = encode_metadata_fields(metadata_fields)
            json.dump(encoded_types, f, indent=2)

        return cls(
            writer=writer,
            metadata_fields=metadata_fields,
        )

    def finalize(
        self,
        index_to_label_override: dict[int, str] | None = None,
    ):
        """
        Performs data finalization and some preprocessing steps.

        Parameters
        ----------
        index_to_label_override : dict[int, str], optional
            Pre-configures label mapping. Used when operating over filtered subsets.

        Returns
        -------
        Evaluator
            A ready-to-use evaluator object.
        """
        self._writer.flush()
        if self._writer.count_rows() == 0:
            raise EmptyCacheError()

        reader = self._writer.to_reader()

        # extract labels
        index_to_label = extract_labels(
            reader=reader,
            index_to_label_override=index_to_label_override,
        )

        return Evaluator(
            reader=reader,
            index_to_label=index_to_label,
            metadata_fields=self._metadata_fields,
        )


class Evaluator:
    def __init__(
        self,
        reader: MemoryCacheReader | FileCacheReader,
        index_to_label: dict[int, str],
        metadata_fields: list[tuple[str, str | pa.DataType]] | None = None,
    ):
        self._reader = reader
        self._index_to_label = index_to_label
        self._metadata_fields = metadata_fields

    @property
    def info(self) -> EvaluatorInfo:
        return self.get_info()

    def get_info(
        self,
        datums: pc.Expression | None = None,
        groundtruths: pc.Expression | None = None,
        predictions: pc.Expression | None = None,
    ) -> EvaluatorInfo:
        info = EvaluatorInfo()
        info.number_of_rows = self._reader.count_rows()
        info.number_of_labels = len(self._index_to_label)
        info.metadata_fields = self._metadata_fields
        (
            info.number_of_datums,
            info.number_of_pixels,
            info.number_of_groundtruth_pixels,
            info.number_of_prediction_pixels,
        ) = extract_counts(
            reader=self._reader,
            datums=datums,
            groundtruths=groundtruths,
            predictions=predictions,
        )
        return info

    @classmethod
    def load(
        cls,
        path: str | Path,
        index_to_label_override: dict[int, str] | None = None,
    ):
        """
        Load from an existing semantic segmentation cache.

        Parameters
        ----------
        path : str | Path
            Path to the existing cache.
        index_to_label_override : dict[int, str], optional
            Option to preset index to label dictionary. Used when loading from filtered caches.
        """
        # validate path
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Directory does not exist: {path}")
        elif not path.is_dir():
            raise NotADirectoryError(
                f"Path exists but is not a directory: {path}"
            )

        # load cache
        reader = FileCacheReader.load(generate_cache_path(path))

        # extract labels
        index_to_label = extract_labels(
            reader=reader,
            index_to_label_override=index_to_label_override,
        )

        # read config
        metadata_path = generate_metadata_path(path)
        metadata_fields = None
        with open(metadata_path, "r") as f:
            metadata_types = json.load(f)
            metadata_fields = decode_metadata_fields(metadata_types)

        return cls(
            reader=reader,
            index_to_label=index_to_label,
            metadata_fields=metadata_fields,
        )

    def filter(
        self,
        datums: pc.Expression | None = None,
        groundtruths: pc.Expression | None = None,
        predictions: pc.Expression | None = None,
        path: str | Path | None = None,
    ) -> Evaluator:
        """
        Filter evaluator cache.

        Parameters
        ----------
        datums : pc.Expression | None = None
            A filter expression used to filter datums.
        groundtruths : pc.Expression | None = None
            Select whole original ground truth annotations when any of their
            cache rows match, including after label remapping.
        predictions : pc.Expression | None = None
            Select whole original prediction annotations when any of their
            cache rows match, including after label remapping.
        path : str | Path, optional
            Where to store the filtered cache if storing on disk.

        Returns
        -------
        Evaluator
            A new evaluator object containing the filtered cache.
        """
        if isinstance(self._reader, FileCacheReader):
            if not path:
                raise ValueError(
                    "expected path to be defined for file-based cache"
                )
            builder = Builder.persistent(
                path=path,
                batch_size=self._reader.batch_size,
                rows_per_file=self._reader.rows_per_file,
                compression=self._reader.compression,
                metadata_fields=self._metadata_fields,
            )
        else:
            builder = Builder.in_memory(
                batch_size=self._reader.batch_size,
                metadata_fields=self._metadata_fields,
            )

        for tbl in self._reader.iterate_tables(filter=datums):
            tbl = ensure_annotation_ids(tbl)
            masks = (
                _annotation_mask(tbl, groundtruths, "gt"),
                _annotation_mask(tbl, predictions, "pd"),
            )
            for side, mask in zip(("gt", "pd"), masks):
                column = f"{side}_label_id"
                index = tbl.schema.get_field_index(column)
                tbl = tbl.set_column(
                    index,
                    tbl.schema.field(index),
                    pc.if_else(pa.array(mask), tbl[column], -1),
                )
            builder._writer.write_table(tbl)

        return builder.finalize(index_to_label_override=self._index_to_label)

    def remap_labels(
        self,
        mapping: dict[str, str],
        path: str | Path | None = None,
    ) -> Evaluator:
        """
        Remap labels into a new evaluator cache.

        Parameters
        ----------
        mapping : dict[str, str]
            Label replacements applied to both ground truths and predictions.
            Labels absent from the mapping retain their names. Unknown source
            labels are ignored. Replacements apply once, without chaining.
            Multiple labels may map to the same label; their pixel counts are
            combined during metric computation. Cache rows and their metadata
            are preserved, including background and filtered pixels.
        path : str | Path, optional
            Destination for a file-based cache. Required for file-based
            evaluators and ignored for in-memory evaluators, as with filter.

        Returns
        -------
        Evaluator
            A new evaluator containing remapped pixel counts. The source
            evaluator is unchanged.

        Notes
        -----
        Original annotation identities survive remapping and persistent
        reloads, so metadata filters still select whole original annotations.
        Legacy caches derive identities from their existing label IDs. If a
        legacy cache already merged labels, recreate it from the unmerged
        source to recover those original identities.
        """
        if not isinstance(mapping, dict) or any(
            not isinstance(source, str) or not isinstance(target, str)
            for source, target in mapping.items()
        ):
            raise TypeError("mapping must be a dict[str, str]")

        label_to_index: dict[str, int] = {}
        old_indices = [-1]
        new_indices = [-1]
        new_labels: list[str | None] = [None]
        for old_index, label in self._index_to_label.items():
            new_label = mapping.get(label, label)
            new_index = label_to_index.setdefault(
                new_label, len(label_to_index)
            )
            old_indices.append(old_index)
            new_indices.append(new_index)
            new_labels.append(new_label)

        if isinstance(self._reader, FileCacheReader):
            if not path:
                raise ValueError(
                    "expected path to be defined for file-based cache"
                )
            builder = Builder.persistent(
                path=path,
                batch_size=self._reader.batch_size,
                rows_per_file=self._reader.rows_per_file,
                compression=self._reader.compression,
                metadata_fields=self._metadata_fields,
            )
        else:
            builder = Builder.in_memory(
                batch_size=self._reader.batch_size,
                metadata_fields=self._metadata_fields,
            )

        old_ids = pa.array(old_indices, type=pa.int64())
        new_ids = pa.array(new_indices, type=pa.int64())
        labels = pa.array(new_labels, type=pa.string())
        for tbl in self._reader.iterate_tables():
            tbl = ensure_annotation_ids(tbl)
            for side in ("gt", "pd"):
                id_column = f"{side}_label_id"
                label_column = f"{side}_label"
                indices = pc.index_in(tbl[id_column], value_set=old_ids)
                # Filtered pixels can retain label strings with a -1 ID.
                # Preserve those strings without reviving the removed labels.
                remapped_labels = pc.if_else(
                    pc.greater_equal(tbl[id_column], 0),
                    pc.take(labels, indices),
                    tbl[label_column],
                )
                for column, values in (
                    (id_column, pc.take(new_ids, indices)),
                    (label_column, remapped_labels),
                ):
                    index = tbl.schema.get_field_index(column)
                    tbl = tbl.set_column(
                        index, tbl.schema.field(index), values
                    )

            # Keep datum-aligned fragments and annotation metadata intact.
            # Metric computation sums counts for the resulting label pairs.
            builder._writer.write_table(tbl)

        return builder.finalize(
            index_to_label_override={
                index: label for label, index in label_to_index.items()
            },
        )

    def _compute_confusion_matrix_intermediate(
        self, datums: pc.Expression | None = None
    ) -> NDArray[np.uint64]:
        """
        Performs an evaluation and returns metrics.

        Parameters
        ----------
        datums : pyarrow.compute.Expression, optional
            Option to filter datums by an expression.

        Returns
        -------
        dict[MetricType, list]
            A dictionary mapping MetricType enumerations to lists of computed metrics.
        """
        n_labels = len(self._index_to_label)
        confusion_matrix = np.zeros(
            (n_labels + 1, n_labels + 1), dtype=np.uint64
        )
        for tbl in self._reader.iterate_tables(filter=datums):
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
            confusion_matrix[0, 0] += counts[
                mask_null_gts & mask_null_pds
            ].sum()
            for idx in range(n_labels):
                mask_gts = ids[:, 1] == idx
                for pidx in range(n_labels):
                    mask_pds = ids[:, 2] == pidx
                    confusion_matrix[idx + 1, pidx + 1] += counts[
                        mask_gts & mask_pds
                    ].sum()

                mask_unmatched_gts = mask_gts & mask_null_pds
                confusion_matrix[idx + 1, 0] += counts[
                    mask_unmatched_gts
                ].sum()
                mask_unmatched_pds = mask_null_gts & (ids[:, 2] == idx)
                confusion_matrix[0, idx + 1] += counts[
                    mask_unmatched_pds
                ].sum()
        return confusion_matrix

    def compute_precision_recall_iou(
        self, datums: pc.Expression | None = None
    ) -> dict[MetricType, list]:
        """
        Performs an evaluation and returns metrics.

        Parameters
        ----------
        datums : pyarrow.compute.Expression, optional
            Option to filter datums by an expression.

        Returns
        -------
        dict[MetricType, list]
            A dictionary mapping MetricType enumerations to lists of computed metrics.
        """
        confusion_matrix = self._compute_confusion_matrix_intermediate(
            datums=datums
        )
        results = compute_metrics(confusion_matrix=confusion_matrix)
        return unpack_precision_recall_iou_into_metric_lists(
            results=results,
            index_to_label=self._index_to_label,
        )
