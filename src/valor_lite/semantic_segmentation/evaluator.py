from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray

from valor_lite.cache import (
    FileCacheReader,
    FileCacheWriter,
    MemoryCacheReader,
    MemoryCacheWriter,
)
from valor_lite.exceptions import EmptyCacheError
from valor_lite.filtering import DataType, Expression
from valor_lite.semantic_segmentation.computation import compute_metrics
from valor_lite.semantic_segmentation.metric import MetricType
from valor_lite.semantic_segmentation.shared import (
    EvaluatorInfo,
    decode_metadata_fields,
    encode_metadata_fields,
    extract_counts,
    extract_labels,
    generate_cache_path,
    generate_metadata_path,
    generate_schema,
)
from valor_lite.semantic_segmentation.utilities import (
    unpack_precision_recall_iou_into_metric_lists,
)


class Builder:
    def __init__(
        self,
        writer: MemoryCacheWriter | FileCacheWriter,
        metadata_fields: list[tuple[str, str | DataType]] | None = None,
    ):
        self._writer = writer
        self._metadata_fields = metadata_fields

    @classmethod
    def in_memory(
        cls,
        batch_size: int = 10_000,
        metadata_fields: list[tuple[str, str | DataType]] | None = None,
    ):
        """
        Create an in-memory evaluator cache.

        Parameters
        ----------
        batch_size : int, default=10_000
            The target number of rows to buffer before writing to the cache. Defaults to 10_000.
        metadata_fields : list[tuple[str, str | DataType]], optional
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
        metadata_fields: list[tuple[str, str | DataType]] | None = None,
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
        metadata_fields : list[tuple[str, str | DataType]], optional
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
        metadata_fields: list[tuple[str, str | DataType]] | None = None,
    ):
        self._reader = reader
        self._index_to_label = index_to_label
        self._metadata_fields = metadata_fields

    @property
    def info(self) -> EvaluatorInfo:
        return self.get_info()

    def get_info(
        self,
        datums: Expression | None = None,
        groundtruths: Expression | None = None,
        predictions: Expression | None = None,
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
        datums: Expression | None = None,
        groundtruths: Expression | None = None,
        predictions: Expression | None = None,
        path: str | Path | None = None,
    ) -> Evaluator:
        """
        Filter evaluator cache.

        Parameters
        ----------
        datums : Expression | None = None
            A filter expression used to filter datums.
        groundtruths : Expression | None = None
            A filter expression used to filter ground truth annotations.
        predictions : Expression | None = None
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

        datum_filter = datums.to_arrow() if datums is not None else None
        for tbl in self._reader.iterate_tables(filter=datum_filter):
            columns = (
                "datum_id",
                "gt_label_id",
                "pd_label_id",
            )
            pairs = np.column_stack([tbl[col].to_numpy() for col in columns])

            n_pairs = pairs.shape[0]
            gt_ids = pairs[:, (0, 1)].astype(np.int64)
            pd_ids = pairs[:, (0, 2)].astype(np.int64)

            if groundtruths is not None:
                mask_valid_gt = np.zeros(n_pairs, dtype=np.bool_)
                gt_tbl = tbl.filter(groundtruths.to_arrow())
                gt_pairs = np.column_stack(
                    [
                        gt_tbl[col].to_numpy()
                        for col in ("datum_id", "gt_label_id")
                    ]
                ).astype(np.int64)
                for gt in np.unique(gt_pairs, axis=0):
                    mask_valid_gt |= (gt_ids == gt).all(axis=1)
            else:
                mask_valid_gt = np.ones(n_pairs, dtype=np.bool_)

            if predictions is not None:
                mask_valid_pd = np.zeros(n_pairs, dtype=np.bool_)
                pd_tbl = tbl.filter(predictions.to_arrow())
                pd_pairs = np.column_stack(
                    [
                        pd_tbl[col].to_numpy()
                        for col in ("datum_id", "pd_label_id")
                    ]
                ).astype(np.int64)
                for pd in np.unique(pd_pairs, axis=0):
                    mask_valid_pd |= (pd_ids == pd).all(axis=1)
            else:
                mask_valid_pd = np.ones(n_pairs, dtype=np.bool_)

            mask_valid = mask_valid_gt | mask_valid_pd
            mask_valid_gt &= mask_valid
            mask_valid_pd &= mask_valid

            pairs[~mask_valid_gt, 1] = -1
            pairs[~mask_valid_pd, 2] = -1

            for idx, col in enumerate(columns):
                tbl = tbl.set_column(
                    tbl.schema.names.index(col), col, pa.array(pairs[:, idx])
                )
            builder._writer.write_table(tbl)

        return builder.finalize(index_to_label_override=self._index_to_label)

    def _compute_confusion_matrix_intermediate(
        self, datums: Expression | None = None
    ) -> NDArray[np.uint64]:
        """
        Performs an evaluation and returns metrics.

        Parameters
        ----------
        datums : Expression, optional
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
        datum_filter = datums.to_arrow() if datums is not None else None
        for tbl in self._reader.iterate_tables(filter=datum_filter):
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
        self, datums: Expression | None = None
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
