from __future__ import annotations

import json
from operator import index
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from numpy.typing import NDArray

from valor_lite.exceptions import EmptyCacheError
from valor_lite.cache import compute
from valor_lite.cache import (
    MemoryCacheReader,
    MemoryCacheWriter,
    FileCacheReader,
    FileCacheWriter,
)
from valor_lite.classification.computation import (
    compute_accuracy,
    compute_confusion_matrix,
    compute_counts,
    compute_f1_score,
    compute_pair_classifications,
    compute_precision,
    compute_recall,
    compute_rocauc,
)
from valor_lite.classification.metric import Metric, MetricType
from valor_lite.classification.shared import (
    EvaluatorInfo,
    generate_cache_path,
    generate_meta,
    generate_metadata_path,
    generate_rocauc_cache_path,
    generate_rocauc_schema,
    generate_schema,
    encode_metadata_fields,
    decode_metadata_fields,
)
from valor_lite.classification.utilities import (
    create_empty_confusion_matrix_with_examples,
    create_mapping,
    unpack_confusion_matrix,
    unpack_confusion_matrix_with_examples,
    unpack_examples,
    unpack_precision_recall,
    unpack_rocauc,
)


class Builder:
    def __init__(
        self,
        writer: MemoryCacheWriter | FileCacheWriter,
        sorted_writer: MemoryCacheWriter | FileCacheWriter,
        metadata_fields: list[tuple[str, pa.DataType]] | None = None,
    ):
        self._writer = writer
        self._rocauc_writer = sorted_writer
        self._metadata_fields = metadata_fields

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
            Optional metadata field definitions.
        """
        writer = MemoryCacheWriter.create(
            schema=generate_schema(metadata_fields),
            batch_size=batch_size,
        )
        sorted_writer = MemoryCacheWriter.create(
            schema=generate_rocauc_schema(),
            batch_size=batch_size,
        )
        return cls(
            writer=writer,
            sorted_writer=sorted_writer,
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
    ):
        """
        Create a persistent file-based evaluator cache.

        Parameters
        ----------
        path : str | Path
            Where to store file-based cache.
        batch_size : int, default=10_000
            Sets the batch size for writing to file.
        rows_per_file : int, default=100_000
            Sets the maximum number of rows per file. This may be exceeded as files are datum aligned.
        compression : str, default="snappy"
            Sets the pyarrow compression method.
        metadata_fields : list[tuple[str, pa.DataType]], optional
            Optionally sets metadata description for use in filtering.
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
        sorted_writer = FileCacheWriter.create(
            path=generate_rocauc_cache_path(path),
            schema=generate_rocauc_schema(),
            batch_size=batch_size,
            rows_per_file=rows_per_file,
            compression=compression,
        )

        # write metadatata config
        metadata_path = generate_metadata_path(path)
        with open(metadata_path, "w") as f:
            encoded_types = encode_metadata_fields(metadata_fields)
            json.dump(encoded_types, f, indent=2)

        return cls(
            writer=writer,
            sorted_writer=sorted_writer,
            metadata_fields=metadata_fields,
        )

    def finalize(
        self,
        batch_size: int = 1_000,
        index_to_label_override: dict[int, str] | None = None,
    ):
        """
        Performs data finalization and some preprocessing steps.

        Parameters
        ----------
        batch_size : int, default=1_000
            Sets the maximum number of elements read into memory per-file when performing merge sort.
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
        elif self._rocauc_writer.count_rows() > 0:
            raise RuntimeError("data already finalized")

        # sort in-place and locally
        self._writer.sort_by(
            [
                ("score", "descending"),
                ("datum_id", "ascending"),
                ("gt_label_id", "ascending"),
                ("pd_label_id", "ascending"),
            ]
        )

        # post-process into sorted writer
        reader = self._writer.to_reader()

        # generate evaluator meta
        (
            index_to_label, 
            label_counts, 
            info,
        ) = generate_meta(reader=reader, index_to_label_override=index_to_label_override)
        n_labels = len(index_to_label)

        # def accumulate(batch: pa.RecordBatch, prev: np.ndarray | None) -> pa.RecordBatch:
        #     pd_label_id = batch["pd_label_id"].as_py()
        #     matched = batch["match"][0].as_py()
        #     if prev is None:
        #         prev = np.zeros(n_labels, dtype=np.uint64)
        #         return None, ()

        compute.sort(
            source=reader,
            sink=self._rocauc_writer,
            batch_size=batch_size,
            sorting=[
                ("score", "descending"),
                # ("match", "descending"),
                # ("pd_label_id", "ascending"),
            ],
            columns=[
                "pd_label_id",
                "score",
                "match",
            ],
        )
        rocauc_reader = self._rocauc_writer.to_reader()

        return Evaluator(
            reader=reader,
            rocauc_reader=rocauc_reader,
            info=info,
            label_counts=label_counts,
            index_to_label=index_to_label,
        )


class Evaluator:
    def __init__(
        self,
        reader: MemoryCacheReader | FileCacheReader,
        rocauc_reader: MemoryCacheReader | FileCacheReader,
        info: EvaluatorInfo,
        label_counts: NDArray[np.uint64],
        index_to_label: dict[int, str],
    ):
        self._reader = reader
        self._rocauc_reader = rocauc_reader
        self._info = info
        self._label_counts = label_counts
        self._index_to_label = index_to_label

    @property
    def info(self) -> EvaluatorInfo:
        return self._info

    @classmethod
    def load(
        cls,
        path: str | Path,
        index_to_label_override: dict[int, str] | None = None,
    ):
        """
        Load from an existing classification cache.

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
        rocauc_reader = FileCacheReader.load(
            generate_rocauc_cache_path(path)
        )

        # build evaluator meta
        (
            index_to_label,
            label_counts,
            info,
        ) = generate_meta(reader, index_to_label_override)

        # read config
        metadata_path = generate_metadata_path(path)
        with open(metadata_path, "r") as f:
            info.metadata_fields = json.load(f)

        return cls(
            reader=reader,
            rocauc_reader=rocauc_reader,
            info=info,
            label_counts=label_counts,
            index_to_label=index_to_label,
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
        from valor_lite.classification.loader import Loader

        if isinstance(self._reader, FileCacheReader):
            if not path:
                raise ValueError(
                    "expected path to be defined for file-based loader"
                )
            loader = Loader.persistent(
                path=path,
                batch_size=self._reader.batch_size,
                rows_per_file=self._reader.rows_per_file,
                compression=self._reader.compression,
                metadata_fields=self.info.metadata_fields,
            )
        else:
            loader = Loader.in_memory(
                batch_size=self._reader.batch_size,
                metadata_fields=self.info.metadata_fields,
            )

        for tbl in self._reader.iterate_tables(filter=datums):
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
                gt_tbl = tbl.filter(groundtruths)
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
                pd_tbl = tbl.filter(predictions)
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
            # TODO (c.zaloom) - improve write strategy, filtered data could be small
            loader._writer.write_table(tbl)

        return loader.finalize(index_to_label_override=self._index_to_label)

    def iterate_values(self):
        columns = [
            "datum_id",
            "gt_label_id",
            "pd_label_id",
            "score",
            "winner",
            "match",
        ]
        for tbl in self._reader.iterate_tables(columns=columns):
            ids = np.column_stack(
                [
                    tbl[col].to_numpy()
                    for col in [
                        "datum_id",
                        "gt_label_id",
                        "pd_label_id",
                    ]
                ]
            )
            scores = tbl["score"].to_numpy()
            winners = tbl["winner"].to_numpy()
            matches = tbl["match"].to_numpy()
            yield ids, scores, winners, matches

    def iterate_values_with_tables(self):
        for tbl in self._reader.iterate_tables():
            ids = np.column_stack(
                [
                    tbl[col].to_numpy()
                    for col in [
                        "datum_id",
                        "gt_label_id",
                        "pd_label_id",
                    ]
                ]
            )
            scores = tbl["score"].to_numpy()
            winners = tbl["winner"].to_numpy()
            matches = tbl["match"].to_numpy()
            yield ids, scores, winners, matches, tbl

    def compute_rocauc(self) -> dict[MetricType, list[Metric]]:
        """
        Compute ROCAUC.

        Returns
        -------
        dict[MetricType, list[Metric]]
            A dictionary mapping MetricType enumerations to lists of computed metrics.
        """
        n_labels = self.info.number_of_labels

        rocauc = np.zeros(n_labels, dtype=np.float64)
        accumulated_fp = np.zeros(n_labels, dtype=np.uint64)
        accumulated_tp = np.zeros(n_labels, dtype=np.uint64)

        prev_tpr = np.ones(n_labels, dtype=np.float64) * -1
        prev_fpr = np.ones(n_labels, dtype=np.float64) * -1

        for loopid, array in enumerate(self._rocauc_reader.iterate_arrays(
            numeric_columns=[
                "pd_label_id",
                "score",
                "match",
            ]
        )):
            # for id_row, score in zip(ids, scores):
            #     glabel = id_row[1]
            #     plabel = id_row[2]
            #     if glabel < 0 or plabel < 0:
            #         continue
            #     elif glabel == plabel:
            #         cumulative_tp[plabel] += 1
            #         tpr_new = cumulative_tp[plabel] / positive_count[plabel]
            #         tpr[plabel]
            #     else:
            #         cumulative_fp[plabel] += 1
            rocauc, accumulated_fp, accumulated_tp = compute_rocauc(
                rocauc=rocauc,
                array=array,
                gt_count_per_label=self._label_counts[:, 0],
                pd_count_per_label=self._label_counts[:, 1],
                n_labels=self.info.number_of_labels,
                accumulated_fp=accumulated_fp,
                accumulated_tp=accumulated_tp,
            )

        mean_rocauc = rocauc.mean()

        return unpack_rocauc(
            rocauc=rocauc,
            mean_rocauc=mean_rocauc,
            index_to_label=self._index_to_label,
        )

    def compute_precision_recall(
        self,
        score_thresholds: list[float] = [0.0],
        hardmax: bool = True,
    ) -> dict[MetricType, list]:
        """
        Performs an evaluation and returns metrics.

        Parameters
        ----------
        score_thresholds : list[float]
            A list of score thresholds to compute metrics over.
        hardmax : bool
            Toggles whether a hardmax is applied to predictions.
        rows_per_chunk : int, default=10_000
            The number of sorted rows to return in each chunk.
        read_batch_size : int, default=1_000
            The maximum number of rows to load in-memory per file.

        Returns
        -------
        dict[MetricType, list]
            A dictionary mapping MetricType enumerations to lists of computed metrics.
        """
        if not score_thresholds:
            raise ValueError("At least one score threshold must be passed.")

        n_scores = len(score_thresholds)
        n_datums = self.info.number_of_datums
        n_labels = self.info.number_of_labels

        # intermediates
        counts = np.zeros((n_scores, n_labels, 4), dtype=np.uint64)

        for ids, scores, winners, _ in self.iterate_values():
            batch_counts = compute_counts(
                ids=ids,
                scores=scores,
                winners=winners,
                score_thresholds=np.array(score_thresholds),
                hardmax=hardmax,
                n_labels=n_labels,
            )
            counts += batch_counts

        precision = compute_precision(counts)
        recall = compute_recall(counts)
        f1_score = compute_f1_score(precision, recall)
        accuracy = compute_accuracy(counts, n_datums=n_datums)

        return unpack_precision_recall(
            counts=counts,
            precision=precision,
            recall=recall,
            accuracy=accuracy,
            f1_score=f1_score,
            score_thresholds=score_thresholds,
            hardmax=hardmax,
            index_to_label=self._index_to_label,
        )

    def compute_confusion_matrix(
        self,
        score_thresholds: list[float] = [0.0],
        hardmax: bool = True,
    ) -> list[Metric]:
        """
        Compute a confusion matrix.

        Parameters
        ----------
        score_thresholds : list[float]
            A list of score thresholds to compute metrics over.
        hardmax : bool
            Toggles whether a hardmax is applied to predictions.

        Returns
        -------
        list[Metric]
            A list of confusion matrices.
        """
        if not score_thresholds:
            raise ValueError("At least one score threshold must be passed.")

        n_scores = len(score_thresholds)
        n_labels = len(self._index_to_label)
        confusion_matrices = np.zeros(
            (n_scores, n_labels, n_labels), dtype=np.uint64
        )
        unmatched_groundtruths = np.zeros(
            (n_scores, n_labels), dtype=np.uint64
        )
        for ids, scores, winners, matches in self.iterate_values():
            (
                mask_tp,
                mask_fp_fn_misclf,
                mask_fn_unmatched,
            ) = compute_pair_classifications(
                ids=ids,
                scores=scores,
                winners=winners,
                score_thresholds=np.array(score_thresholds),
                hardmax=hardmax,
            )

            batch_cm, batch_ugt = compute_confusion_matrix(
                ids=ids,
                mask_tp=mask_tp,
                mask_fp_fn_misclf=mask_fp_fn_misclf,
                mask_fn_unmatched=mask_fn_unmatched,
                score_thresholds=np.array(score_thresholds),
                n_labels=n_labels,
            )
            confusion_matrices += batch_cm
            unmatched_groundtruths += batch_ugt

        return unpack_confusion_matrix(
            confusion_matrices=confusion_matrices,
            unmatched_groundtruths=unmatched_groundtruths,
            index_to_label=self._index_to_label,
            score_thresholds=score_thresholds,
            hardmax=hardmax,
        )

    def compute_examples(
        self,
        score_thresholds: list[float] = [0.0],
        hardmax: bool = True,
    ) -> list[Metric]:
        """
        Compute examples per datum.

        Note: This function should be used with filtering to reduce response size.

        Parameters
        ----------
        score_thresholds : list[float]
            A list of score thresholds to compute metrics over.
        hardmax : bool
            Toggles whether a hardmax is applied to predictions.

        Returns
        -------
        list[Metric]
            A list of confusion matrices.
        """
        if not score_thresholds:
            raise ValueError("At least one score threshold must be passed.")

        metrics = []
        for (
            ids,
            scores,
            winners,
            matches,
            tbl,
        ) in self.iterate_values_with_tables():
            if ids.size == 0:
                continue

            # extract external identifiers
            index_to_datum_id = create_mapping(
                tbl, ids, 0, "datum_id", "datum_uid"
            )

            (
                mask_tp,
                mask_fp_fn_misclf,
                mask_fn_unmatched,
            ) = compute_pair_classifications(
                ids=ids,
                scores=scores,
                winners=winners,
                score_thresholds=np.array(score_thresholds),
                hardmax=hardmax,
            )

            mask_fn = mask_fp_fn_misclf | mask_fn_unmatched
            mask_fp = mask_fp_fn_misclf

            batch_examples = unpack_examples(
                ids=ids,
                mask_tp=mask_tp,
                mask_fp=mask_fp,
                mask_fn=mask_fn,
                index_to_datum_id=index_to_datum_id,
                score_thresholds=score_thresholds,
                hardmax=hardmax,
                index_to_label=self._index_to_label,
            )
            metrics.extend(batch_examples)

        return metrics

    def compute_confusion_matrix_with_examples(
        self,
        score_thresholds: list[float] = [0.0],
        hardmax: bool = True,
    ) -> list[Metric]:
        """
        Compute confusion matrix with examples.

        Note: This function should be used with filtering to reduce response size.

        Parameters
        ----------
        metrics : dict[int, Metric]
            Mapping of score threshold index to cached metric.
        score_thresholds : list[float]
            A list of score thresholds to compute metrics over.
        hardmax : bool
            Toggles whether a hardmax is applied to predictions.

        Returns
        -------
        list[Metric]
            A list of confusion matrices.
        """
        if not score_thresholds:
            raise ValueError("At least one score threshold must be passed.")

        metrics = {
            score_idx: create_empty_confusion_matrix_with_examples(
                score_threshold=score_thresh,
                hardmax=hardmax,
                index_to_label=self._index_to_label,
            )
            for score_idx, score_thresh in enumerate(score_thresholds)
        }
        for (
            ids,
            scores,
            winners,
            matches,
            tbl,
        ) in self.iterate_values_with_tables():
            if ids.size == 0:
                continue

            # extract external identifiers
            index_to_datum_id = create_mapping(
                tbl, ids, 0, "datum_id", "datum_uid"
            )

            (
                mask_tp,
                mask_fp_fn_misclf,
                mask_fn_unmatched,
            ) = compute_pair_classifications(
                ids=ids,
                scores=scores,
                winners=winners,
                score_thresholds=np.array(score_thresholds),
                hardmax=hardmax,
            )

            mask_matched = mask_tp | mask_fp_fn_misclf
            mask_unmatched_fn = mask_fn_unmatched

            unpack_confusion_matrix_with_examples(
                metrics=metrics,
                ids=ids,
                scores=scores,
                winners=winners,
                mask_matched=mask_matched,
                mask_unmatched_fn=mask_unmatched_fn,
                index_to_datum_id=index_to_datum_id,
                index_to_label=self._index_to_label,
            )

        return list(metrics.values())
