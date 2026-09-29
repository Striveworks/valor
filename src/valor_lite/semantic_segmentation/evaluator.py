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
    annotation_validity,
    decode_metadata_fields,
    encode_metadata_fields,
    extract_counts,
    extract_labels,
    generate_cache_path,
    generate_metadata_path,
    generate_schema,
    mask_annotations,
)
from valor_lite.semantic_segmentation.utilities import (
    unpack_precision_recall_iou_into_metric_lists,
)


class Builder:
    def __init__(
        self,
        writer: MemoryCacheWriter | FileCacheWriter,
        metadata_fields: list[tuple[str, str | pa.DataType]] | None = None,
    ):
        self._writer = writer
        self._metadata_fields = metadata_fields or None

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
        self._index_to_label = dict(sorted(index_to_label.items()))
        self._metadata_fields = metadata_fields or None

    @property
    def info(self) -> EvaluatorInfo:
        return self.get_info()

    def get_info(
        self,
        datums: pc.Expression | None = None,
        groundtruths: pc.Expression | None = None,
        predictions: pc.Expression | None = None,
    ) -> EvaluatorInfo:
        """Count evaluated pixels after masking each annotation side independently."""
        info = EvaluatorInfo()
        if datums is None and groundtruths is None and predictions is None:
            info.number_of_rows = self._reader.count_rows()
            info.number_of_labels = len(self._index_to_label)
        else:
            label_ids = set()
            for tbl in self._reader.iterate_tables(filter=datums):
                tbl = mask_annotations(tbl, groundtruths, predictions)
                info.number_of_rows += tbl.num_rows
                for col in ("gt_label_id", "pd_label_id"):
                    label_ids.update(tbl[col].to_pylist())
            label_ids.discard(-1)
            info.number_of_labels = len(label_ids)
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
        Mask ground truth and prediction annotations independently.

        Datum filters discard complete rows. An excluded annotation is masked
        while the remaining side contributes a false positive or false negative.
        Rows excluded on both sides are discarded from all counts and metrics.
        Retained rows keep their labels and IDs, with gt_valid/pd_valid masks.
        Raises EmptyCacheError if no rows remain.

        Parameters
        ----------
        datums : pc.Expression | None = None
            A filter expression used to filter datums.
        groundtruths : pc.Expression | None = None
            A filter expression used to filter ground truth annotations.
        predictions : pc.Expression | None = None
            A filter expression used to filter predictions.
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
                metadata_fields=self.info.metadata_fields,
            )
        else:
            builder = Builder.in_memory(
                batch_size=self._reader.batch_size,
                metadata_fields=self.info.metadata_fields,
            )

        for tbl in self._reader.iterate_tables(filter=datums):
            tbl = mask_annotations(tbl, groundtruths, predictions)
            if tbl.num_rows:
                builder._writer.write_table(
                    tbl.select(builder._writer.schema.names)
                )

        return builder.finalize()

    def _compute_confusion_matrix_intermediate(
        self,
        datums: pc.Expression | None = None,
        index_to_label: dict[int, str] | None = None,
    ) -> NDArray[np.uint64]:
        """
        Accumulate cached counts into a matrix with dense class positions.

        The extra row and column accommodate missing annotations in legacy
        caches. ``index_to_label`` optionally selects the vocabulary used for
        a datum-filtered evaluation.
        """
        index_to_label = (
            self._index_to_label if index_to_label is None else index_to_label
        )
        n_labels = len(index_to_label)
        confusion_matrix = np.zeros(
            (n_labels + 1, n_labels + 1), dtype=np.uint64
        )
        label_ids = np.array(sorted(index_to_label), dtype=np.int64)
        columns = ["gt_label_id", "pd_label_id", "count"]
        columns.extend(
            col
            for col in ("gt_valid", "pd_valid")
            if col in self._reader.schema.names
        )
        for tbl in self._reader.iterate_tables(columns=columns, filter=datums):
            tbl = mask_annotations(tbl)
            ids = np.column_stack(
                [tbl[col].to_numpy() for col in ("gt_label_id", "pd_label_id")]
            ).astype(np.int64)
            # Mask only working indices; cached label IDs remain unchanged.
            ids[~annotation_validity(tbl, "gt"), 0] = -1
            ids[~annotation_validity(tbl, "pd"), 1] = -1
            # Slot zero represents a masked or legacy missing annotation.
            # Retained cache IDs may be noncontiguous after row filtering.
            valid = ids != -1
            positions = np.searchsorted(label_ids, ids[valid])
            if np.any(positions >= n_labels) or np.any(
                label_ids[positions] != ids[valid]
            ):
                raise ValueError(
                    "cache contains label IDs missing from the vocabulary"
                )
            indices = np.zeros_like(ids)
            indices[valid] = positions + 1
            np.add.at(
                confusion_matrix,
                (indices[:, 0], indices[:, 1]),
                tbl["count"].to_numpy(),
            )
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
        index_to_label = self._index_to_label
        if datums is not None:
            index_to_label = dict(
                sorted(extract_labels(self._reader, filter=datums).items())
            )
        confusion_matrix = self._compute_confusion_matrix_intermediate(
            datums=datums, index_to_label=index_to_label
        )
        results = compute_metrics(confusion_matrix=confusion_matrix)
        return unpack_precision_recall_iou_into_metric_lists(
            results=results,
            index_to_label=dict(enumerate(index_to_label.values())),
        )
