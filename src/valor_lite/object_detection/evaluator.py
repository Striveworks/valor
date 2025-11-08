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
from valor_lite.object_detection.computation import (
    compute_average_precision,
    compute_average_recall,
    compute_confusion_matrix,
    compute_counts,
    compute_pair_classifications,
    compute_precision_recall_f1,
)
from valor_lite.object_detection.metric import Metric, MetricType
from valor_lite.object_detection.shared import (
    EvaluatorInfo,
    generate_detailed_cache_path,
    generate_detailed_schema,
    generate_meta,
    generate_metadata_path,
    generate_ranked_cache_path,
    generate_ranked_schema,
    generate_temporary_cache_path,
)
from valor_lite.object_detection.utilities import (
    create_empty_confusion_matrix_with_examples,
    create_mapping,
    unpack_confusion_matrix,
    unpack_confusion_matrix_with_examples,
    unpack_examples,
    unpack_precision_recall_into_metric_lists,
)


class Builder:
    def __init__(
        self,
        detailed_writer: MemoryCacheWriter | FileCacheWriter,
        ranked_writer: MemoryCacheWriter | FileCacheWriter,
        metadata_fields: list[tuple[str, pa.DataType]] | None = None,
        groundtruth_metadata_fields: list[tuple[str, pa.DataType]]
        | None = None,
        prediction_metadata_fields: list[tuple[str, pa.DataType]]
        | None = None,
        path: str | Path | None = None,
    ):
        self._path = Path(path) if path else None
        self._detailed_writer = detailed_writer
        self._ranked_writer = ranked_writer
        self._metadata_fields = metadata_fields
        self._groundtruth_metadata_fields = groundtruth_metadata_fields
        self._prediction_metadata_fields = prediction_metadata_fields

        # internal state
        self._labels = {}
        self._datum_count = 0
        self._groundtruth_count = 0
        self._prediction_count = 0

    @classmethod
    def in_memory(
        cls,
        batch_size: int = 10_000,
        metadata_fields: list[tuple[str, pa.DataType]] | None = None,
    ):
        """
        Create an in-memory evaluator cache.

        Parameters
        ----------
        batch_size : int, default=10_000
            The target number of rows to buffer before writing to the cache. Defaults to 10_000.
        metadata_fields : list[tuple[str, pa.DataType]], optional
            Optional datum metadata field definitions.
        """
        # create cache
        detailed_writer = MemoryCacheWriter.create(
            schema=generate_detailed_schema(metadata_fields),
            batch_size=batch_size,
        )
        ranked_writer = MemoryCacheWriter.create(
            schema=generate_ranked_schema(
                metadata_fields=metadata_fields,
            ),
            batch_size=batch_size,
        )

        return cls(
            detailed_writer=detailed_writer,
            ranked_writer=ranked_writer,
            metadata_fields=metadata_fields,
        )

    @classmethod
    def persistent(
        cls,
        path: str | Path,
        batch_size: int = 10_000,
        rows_per_file: int = 100_000,
        compression: str = "snappy",
        metadata_fields: list[tuple[str, pa.DataType]] | None = None,
        groundtruth_metadata_fields: list[tuple[str, pa.DataType]]
        | None = None,
        prediction_metadata_fields: list[tuple[str, pa.DataType]]
        | None = None,
        delete_if_exists: bool = False,
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
        metadata_fields : list[tuple[str, pa.DataType]], optional
            Optional datum metadata field definition.
        groundtruth_metadata_fields : list[tuple[str, pa.DataType]], optional
            Optional ground truth annotation metadata field definition.
        prediction_metadata_fields : list[tuple[str, pa.DataType]], optional
            Optional prediction metadata field definition.
        delete_if_exists : bool, default=False
            Option to delete any pre-exisiting cache at the given path.
        """
        path = Path(path)
        if delete_if_exists and path.exists():
            cls.delete_at_path(path)

        # create caches
        detailed_writer = FileCacheWriter.create(
            path=cls._generate_detailed_cache_path(path),
            schema=cls._generate_detailed_schema(
                metadata_fields=metadata_fields,
                groundtruth_metadata_fields=groundtruth_metadata_fields,
                prediction_metadata_fields=prediction_metadata_fields,
            ),
            batch_size=batch_size,
            rows_per_file=rows_per_file,
            compression=compression,
        )
        ranked_writer = FileCacheWriter.create(
            path=cls._generate_ranked_cache_path(path),
            schema=cls._generate_ranked_schema(
                metadata_fields=metadata_fields,
            ),
            batch_size=batch_size,
            rows_per_file=rows_per_file,
            compression=compression,
        )

        # write metadata
        metadata_path = cls._generate_metadata_path(path)
        with open(metadata_path, "w") as f:
            encoded_types = cls._encode_metadata_fields(
                metadata_fields=metadata_fields,
                groundtruth_metadata_fields=groundtruth_metadata_fields,
                prediction_metadata_fields=prediction_metadata_fields,
            )
            json.dump(encoded_types, f, indent=2)

        return cls(
            detailed_writer=detailed_writer,
            ranked_writer=ranked_writer,
            metadata_fields=metadata_fields,
            groundtruth_metadata_fields=groundtruth_metadata_fields,
            prediction_metadata_fields=prediction_metadata_fields,
            path=path,
        )

    def _rank(
        self,
        n_labels: int,
        batch_size: int = 1_000,
    ):
        """Perform pair ranking over the detailed cache."""

        detailed_reader = self._detailed_writer.to_reader()
        cache.sort(
            source=detailed_reader,
            sink=self._ranked_writer,
            batch_size=batch_size,
            sorting=[
                ("score", "descending"),
                ("iou", "descending"),
            ],
            columns=[
                field.name
                for field in self._ranked_writer.schema
                if field.name not in {"high_score", "iou_prev"}
            ],
            table_sort_override=lambda tbl: rank_table(
                tbl, number_of_labels=n_labels
            ),
        )
        self._ranked_writer.flush()

    def finalize(
        self,
        batch_size: int = 1_000,
        index_to_label_override: dict[int, str] | None = None,
    ):
        """
        Performs data finalization and preprocessing.

        Parameters
        ----------
        batch_size : int, default=1_000
            Sets the batch size for reading. Defaults to 1_000.
        index_to_label_override : dict[int, str], optional
            Pre-configures label mapping. Used when operating over filtered subsets.
        """
        self._detailed_writer.flush()
        if self._detailed_writer.count_rows() == 0:
            raise EmptyCacheError()

        detailed_reader = self._detailed_writer.to_reader()

        # build evaluator meta
        (
            index_to_label,
            number_of_groundtruths_per_label,
            info,
        ) = self._generate_meta(detailed_reader, index_to_label_override)
        info.metadata_fields = self._metadata_fields
        info.groundtruth_metadata_fields = self._groundtruth_metadata_fields
        info.prediction_metadata_fields = self._prediction_metadata_fields

        # populate ranked cache
        self._rank(
            n_labels=len(index_to_label),
            batch_size=batch_size,
        )

        ranked_reader = self._ranked_writer.to_reader()
        return Evaluator(
            detailed_reader=detailed_reader,
            ranked_reader=ranked_reader,
            info=info,
            index_to_label=index_to_label,
            number_of_groundtruths_per_label=number_of_groundtruths_per_label,
            path=self._path,
        )


class Evaluator(Base):
    def __init__(
        self,
        detailed_reader: MemoryCacheReader | FileCacheReader,
        ranked_reader: MemoryCacheReader | FileCacheReader,
        info: EvaluatorInfo,
        index_to_label: dict[int, str],
        number_of_groundtruths_per_label: NDArray[np.uint64],
        path: str | Path | None = None,
    ):
        self._path = Path(path) if path else None
        self._detailed_reader = detailed_reader
        self._ranked_reader = ranked_reader
        self._info = info
        self._index_to_label = index_to_label
        self._number_of_groundtruths_per_label = (
            number_of_groundtruths_per_label
        )

    @property
    def info(self) -> EvaluatorInfo:
        return self._info

    @classmethod
    def load(
        cls,
        path: str | Path,
        index_to_label_override: dict[int, str] | None = None,
    ):
        # validate path
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Directory does not exist: {path}")
        elif not path.is_dir():
            raise NotADirectoryError(
                f"Path exists but is not a directory: {path}"
            )

        detailed_reader = FileCacheReader.load(
            cls._generate_detailed_cache_path(path)
        )
        ranked_reader = FileCacheReader.load(
            cls._generate_ranked_cache_path(path)
        )

        # build evaluator meta
        (
            index_to_label,
            number_of_groundtruths_per_label,
            info,
        ) = cls._generate_meta(detailed_reader, index_to_label_override)

        # read config
        metadata_path = cls._generate_metadata_path(path)
        with open(metadata_path, "r") as f:
            encoded_metadata_types = json.load(f)
            (
                info.metadata_fields,
                info.groundtruth_metadata_fields,
                info.prediction_metadata_fields,
            ) = cls._decode_metadata_fields(encoded_metadata_types)

        return cls(
            path=path,
            detailed_reader=detailed_reader,
            ranked_reader=ranked_reader,
            info=info,
            index_to_label=index_to_label,
            number_of_groundtruths_per_label=number_of_groundtruths_per_label,
        )

    def filter(
        self,
        datums: pc.Expression | None = None,
        groundtruths: pc.Expression | None = None,
        predictions: pc.Expression | None = None,
        batch_size: int = 1_000,
        path: str | Path | None = None,
    ) -> "Evaluator":
        """
        Filter evaluator cache.

        Parameters
        ----------
        datums : pc.Expression | None = None
            A filter expression used to filter datums.
        groundtruths : pc.Expression | None = None
            A filter expression used to filter ground truth annotations.
        predictions : pc.Expression | None = None
            A filter expression used to filter predictions.
        batch_size : int
            The maximum number of rows read into memory per file.
        path : str | Path
            Where to store the filtered cache if storing on disk.

        Returns
        -------
        Evaluator
            A new evaluator object containing the filtered cache.
        """
        from valor_lite.object_detection.loader import Loader

        if isinstance(self._detailed_reader, FileCacheReader):
            if not path:
                raise ValueError(
                    "expected path to be defined for file-based loader"
                )
            loader = Loader.persistent(
                path=path,
                batch_size=self._detailed_reader.batch_size,
                rows_per_file=self._detailed_reader.rows_per_file,
                compression=self._detailed_reader.compression,
                metadata_fields=self.info.metadata_fields,
                groundtruth_metadata_fields=self.info.groundtruth_metadata_fields,
                prediction_metadata_fields=self.info.prediction_metadata_fields,
            )
        else:
            loader = Loader.in_memory(
                batch_size=self._detailed_reader.batch_size,
                metadata_fields=self.info.metadata_fields,
                groundtruth_metadata_fields=self.info.groundtruth_metadata_fields,
                prediction_metadata_fields=self.info.prediction_metadata_fields,
            )

        for tbl in self._detailed_reader.iterate_tables(filter=datums):
            columns = (
                "datum_id",
                "gt_id",
                "pd_id",
                "gt_label_id",
                "pd_label_id",
                "iou",
                "score",
            )
            pairs = np.column_stack([tbl[col].to_numpy() for col in columns])

            n_pairs = pairs.shape[0]
            gt_ids = pairs[:, (0, 1)].astype(np.int64)
            pd_ids = pairs[:, (0, 2)].astype(np.int64)

            if groundtruths is not None:
                mask_valid_gt = np.zeros(n_pairs, dtype=np.bool_)
                gt_tbl = tbl.filter(groundtruths)
                gt_pairs = np.column_stack(
                    [gt_tbl[col].to_numpy() for col in ("datum_id", "gt_id")]
                ).astype(np.int64)
                for gt in np.unique(gt_pairs, axis=0):
                    mask_valid_gt |= (gt_ids == gt).all(axis=1)
            else:
                mask_valid_gt = np.ones(n_pairs, dtype=np.bool_)

            if predictions is not None:
                mask_valid_pd = np.zeros(n_pairs, dtype=np.bool_)
                pd_tbl = tbl.filter(predictions)
                pd_pairs = np.column_stack(
                    [pd_tbl[col].to_numpy() for col in ("datum_id", "pd_id")]
                ).astype(np.int64)
                for pd in np.unique(pd_pairs, axis=0):
                    mask_valid_pd |= (pd_ids == pd).all(axis=1)
            else:
                mask_valid_pd = np.ones(n_pairs, dtype=np.bool_)

            mask_valid = mask_valid_gt | mask_valid_pd
            mask_valid_gt &= mask_valid
            mask_valid_pd &= mask_valid

            pairs[np.ix_(~mask_valid_gt, (1, 3))] = -1.0  # type: ignore - numpy ix_
            pairs[np.ix_(~mask_valid_pd, (2, 4, 6))] = -1.0  # type: ignore - numpy ix_
            pairs[~mask_valid_pd | ~mask_valid_gt, 5] = 0.0

            for idx, col in enumerate(columns):
                column = pairs[:, idx]
                if col not in {"iou", "score"}:
                    column = column.astype(np.int64)

                col_idx = tbl.schema.names.index(col)
                tbl = tbl.set_column(
                    col_idx, tbl.schema[col_idx], pa.array(column)
                )

            mask_invalid = ~mask_valid | (pairs[:, (1, 2)] < 0).all(axis=1)
            filtered_tbl = tbl.filter(pa.array(~mask_invalid))
            loader._detailed_writer.write_table(filtered_tbl)

        return loader.finalize(
            batch_size=batch_size,
            index_to_label_override=self._index_to_label,
        )

    def compute_precision_recall(
        self,
        iou_thresholds: list[float],
        score_thresholds: list[float],
    ) -> dict[MetricType, list[Metric]]:
        """
        Computes all metrics except for ConfusionMatrix

        Parameters
        ----------
        iou_thresholds : list[float]
            A list of IOU thresholds to compute metrics over.
        score_thresholds : list[float]
            A list of score thresholds to compute metrics over.

        Returns
        -------
        dict[MetricType, list]
            A dictionary mapping MetricType enumerations to lists of computed metrics.
        """
        if not iou_thresholds:
            raise ValueError("At least one IOU threshold must be passed.")
        elif not score_thresholds:
            raise ValueError("At least one score threshold must be passed.")

        n_ious = len(iou_thresholds)
        n_scores = len(score_thresholds)
        n_labels = len(self._index_to_label)

        counts = np.zeros((n_ious, n_scores, 3, n_labels), dtype=np.uint64)
        pr_curve = np.zeros((n_ious, n_labels, 101, 2), dtype=np.float64)
        running_counts = np.ones((n_ious, n_labels, 2), dtype=np.uint64)
        for pairs in self._ranked_reader.iterate_arrays(
            numeric_columns=[
                "datum_id",
                "gt_id",
                "pd_id",
                "gt_label_id",
                "pd_label_id",
                "iou",
                "score",
                "iou_prev",
                "high_score",
            ],
        ):
            if pairs.size == 0:
                continue

            (batch_counts, batch_pr_curve) = compute_counts(
                ranked_pairs=pairs,
                iou_thresholds=np.array(iou_thresholds),
                score_thresholds=np.array(score_thresholds),
                number_of_groundtruths_per_label=self._number_of_groundtruths_per_label,
                number_of_labels=len(self._index_to_label),
                running_counts=running_counts,
            )
            counts += batch_counts
            pr_curve = np.maximum(batch_pr_curve, pr_curve)

        # fn count
        counts[:, :, 2, :] = (
            self._number_of_groundtruths_per_label - counts[:, :, 0, :]
        )

        precision_recall_f1 = compute_precision_recall_f1(
            counts=counts,
            number_of_groundtruths_per_label=self._number_of_groundtruths_per_label,
        )
        (
            average_precision,
            mean_average_precision,
            pr_curve,
        ) = compute_average_precision(pr_curve=pr_curve)
        average_recall, mean_average_recall = compute_average_recall(
            prec_rec_f1=precision_recall_f1
        )

        return unpack_precision_recall_into_metric_lists(
            counts=counts,
            precision_recall_f1=precision_recall_f1,
            average_precision=average_precision,
            mean_average_precision=mean_average_precision,
            average_recall=average_recall,
            mean_average_recall=mean_average_recall,
            pr_curve=pr_curve,
            iou_thresholds=iou_thresholds,
            score_thresholds=score_thresholds,
            index_to_label=self._index_to_label,
        )

    def compute_confusion_matrix(
        self,
        iou_thresholds: list[float],
        score_thresholds: list[float],
    ) -> list[Metric]:
        """
        Computes confusion matrices at various thresholds.

        Parameters
        ----------
        iou_thresholds : list[float]
            A list of IOU thresholds to compute metrics over.
        score_thresholds : list[float]
            A list of score thresholds to compute metrics over.

        Returns
        -------
        list[Metric]
            List of confusion matrices per threshold pair.
        """
        if not iou_thresholds:
            raise ValueError("At least one IOU threshold must be passed.")
        elif not score_thresholds:
            raise ValueError("At least one score threshold must be passed.")

        n_ious = len(iou_thresholds)
        n_scores = len(score_thresholds)
        n_labels = len(self._index_to_label)

        confusion_matrices = np.zeros(
            (n_ious, n_scores, n_labels, n_labels), dtype=np.uint64
        )
        unmatched_groundtruths = np.zeros(
            (n_ious, n_scores, n_labels), dtype=np.uint64
        )
        unmatched_predictions = np.zeros_like(unmatched_groundtruths)
        for pairs in self._detailed_reader.iterate_arrays(
            numeric_columns=[
                "datum_id",
                "gt_id",
                "pd_id",
                "gt_label_id",
                "pd_label_id",
                "iou",
                "score",
            ],
        ):
            if pairs.size == 0:
                continue

            (
                batch_mask_tp,
                batch_mask_fp_fn_misclf,
                batch_mask_fp_unmatched,
                batch_mask_fn_unmatched,
            ) = compute_pair_classifications(
                detailed_pairs=pairs,
                iou_thresholds=np.array(iou_thresholds),
                score_thresholds=np.array(score_thresholds),
            )
            (
                batch_confusion_matrices,
                batch_unmatched_groundtruths,
                batch_unmatched_predictions,
            ) = compute_confusion_matrix(
                detailed_pairs=pairs,
                mask_tp=batch_mask_tp,
                mask_fp_fn_misclf=batch_mask_fp_fn_misclf,
                mask_fp_unmatched=batch_mask_fp_unmatched,
                mask_fn_unmatched=batch_mask_fn_unmatched,
                number_of_labels=n_labels,
                iou_thresholds=np.array(iou_thresholds),
                score_thresholds=np.array(score_thresholds),
            )
            confusion_matrices += batch_confusion_matrices
            unmatched_groundtruths += batch_unmatched_groundtruths
            unmatched_predictions += batch_unmatched_predictions

        return unpack_confusion_matrix(
            confusion_matrices=confusion_matrices,
            unmatched_groundtruths=unmatched_groundtruths,
            unmatched_predictions=unmatched_predictions,
            index_to_label=self._index_to_label,
            iou_thresholds=iou_thresholds,
            score_thresholds=score_thresholds,
        )

    def compute_examples(
        self,
        iou_thresholds: list[float],
        score_thresholds: list[float],
    ) -> list[Metric]:
        """
        Computes examples at various thresholds.

        This function can use a lot of memory with larger or high density datasets. Please use it with filters.

        Parameters
        ----------
        iou_thresholds : list[float]
            A list of IOU thresholds to compute metrics over.
        score_thresholds : list[float]
            A list of score thresholds to compute metrics over.

        Returns
        -------
        list[Metric]
            List of confusion matrices per threshold pair.
        """
        if not iou_thresholds:
            raise ValueError("At least one IOU threshold must be passed.")
        elif not score_thresholds:
            raise ValueError("At least one score threshold must be passed.")

        metrics = []
        tbl_columns = [
            "datum_uid",
            "gt_uid",
            "pd_uid",
        ]
        numeric_columns = [
            "datum_id",
            "gt_id",
            "pd_id",
            "gt_label_id",
            "pd_label_id",
            "iou",
            "score",
        ]
        for tbl, pairs in self._detailed_reader.iterate_tables_with_arrays(
            columns=tbl_columns + numeric_columns,
            numeric_columns=numeric_columns,
        ):
            if pairs.size == 0:
                continue

            index_to_datum_id = {}
            index_to_groundtruth_id = {}
            index_to_prediction_id = {}

            # extract external identifiers
            index_to_datum_id = create_mapping(
                tbl, pairs, 0, "datum_id", "datum_uid"
            )
            index_to_groundtruth_id = create_mapping(
                tbl, pairs, 1, "gt_id", "gt_uid"
            )
            index_to_prediction_id = create_mapping(
                tbl, pairs, 2, "pd_id", "pd_uid"
            )

            (
                mask_tp,
                mask_fp_fn_misclf,
                mask_fp_unmatched,
                mask_fn_unmatched,
            ) = compute_pair_classifications(
                detailed_pairs=pairs,
                iou_thresholds=np.array(iou_thresholds),
                score_thresholds=np.array(score_thresholds),
            )

            mask_fn = mask_fp_fn_misclf | mask_fn_unmatched
            mask_fp = mask_fp_fn_misclf | mask_fp_unmatched

            batch_examples = unpack_examples(
                detailed_pairs=pairs,
                mask_tp=mask_tp,
                mask_fp=mask_fp,
                mask_fn=mask_fn,
                index_to_datum_id=index_to_datum_id,
                index_to_groundtruth_id=index_to_groundtruth_id,
                index_to_prediction_id=index_to_prediction_id,
                iou_thresholds=iou_thresholds,
                score_thresholds=score_thresholds,
            )
            metrics.extend(batch_examples)

        return metrics

    def compute_confusion_matrix_with_examples(
        self,
        iou_thresholds: list[float],
        score_thresholds: list[float],
    ) -> list[Metric]:
        """
        Computes confusion matrix with examples at various thresholds.

        This function can use a lot of memory with larger or high density datasets. Please use it with filters.

        Parameters
        ----------
        iou_thresholds : list[float]
            A list of IOU thresholds to compute metrics over.
        score_thresholds : list[float]
            A list of score thresholds to compute metrics over.

        Returns
        -------
        list[Metric]
            List of confusion matrices per threshold pair.
        """
        if not iou_thresholds:
            raise ValueError("At least one IOU threshold must be passed.")
        elif not score_thresholds:
            raise ValueError("At least one score threshold must be passed.")

        metrics = {
            iou_idx: {
                score_idx: create_empty_confusion_matrix_with_examples(
                    iou_threhsold=iou_thresh,
                    score_threshold=score_thresh,
                    index_to_label=self._index_to_label,
                )
                for score_idx, score_thresh in enumerate(score_thresholds)
            }
            for iou_idx, iou_thresh in enumerate(iou_thresholds)
        }
        tbl_columns = [
            "datum_uid",
            "gt_uid",
            "pd_uid",
        ]
        numeric_columns = [
            "datum_id",
            "gt_id",
            "pd_id",
            "gt_label_id",
            "pd_label_id",
            "iou",
            "score",
        ]
        for tbl, pairs in self._detailed_reader.iterate_tables_with_arrays(
            columns=tbl_columns + numeric_columns,
            numeric_columns=numeric_columns,
        ):
            if pairs.size == 0:
                continue

            index_to_datum_id = {}
            index_to_groundtruth_id = {}
            index_to_prediction_id = {}

            # extract external identifiers
            index_to_datum_id = create_mapping(
                tbl, pairs, 0, "datum_id", "datum_uid"
            )
            index_to_groundtruth_id = create_mapping(
                tbl, pairs, 1, "gt_id", "gt_uid"
            )
            index_to_prediction_id = create_mapping(
                tbl, pairs, 2, "pd_id", "pd_uid"
            )

            (
                mask_tp,
                mask_fp_fn_misclf,
                mask_fp_unmatched,
                mask_fn_unmatched,
            ) = compute_pair_classifications(
                detailed_pairs=pairs,
                iou_thresholds=np.array(iou_thresholds),
                score_thresholds=np.array(score_thresholds),
            )

            unpack_confusion_matrix_with_examples(
                metrics=metrics,
                detailed_pairs=pairs,
                mask_tp=mask_tp,
                mask_fp_fn_misclf=mask_fp_fn_misclf,
                mask_fp_unmatched=mask_fp_unmatched,
                mask_fn_unmatched=mask_fn_unmatched,
                index_to_datum_id=index_to_datum_id,
                index_to_groundtruth_id=index_to_groundtruth_id,
                index_to_prediction_id=index_to_prediction_id,
                index_to_label=self._index_to_label,
            )

        return [m for inner in metrics.values() for m in inner.values()]

    def delete(self):
        """Delete any cached files."""
        if self._path and self._path.exists():
            self.delete_at_path(self._path)
