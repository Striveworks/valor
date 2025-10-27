from __future__ import annotations

import heapq
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from numpy.typing import NDArray

from valor_lite.cache import CacheReader, DataType
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
from valor_lite.classification.format import PathFormatter
from valor_lite.classification.metric import Metric, MetricType
from valor_lite.classification.utilities import (
    create_empty_confusion_matrix_with_examples,
    create_mapping,
    unpack_confusion_matrix,
    unpack_confusion_matrix_with_examples,
    unpack_examples,
    unpack_precision_recall_rocauc_into_metric_lists,
)
from valor_lite.exceptions import EmptyCacheError


@dataclass
class EvaluatorInfo:
    number_of_rows: int = 0
    number_of_datums: int = 0
    number_of_labels: int = 0
    datum_metadata_types: dict[str, DataType] | None = None


@dataclass
class Filter:
    datums: pc.Expression | None = None
    groundtruths: pc.Expression | None = None
    predictions: pc.Expression | None = None


class Evaluator(PathFormatter):
    def __init__(
        self,
        path: str | Path,
        reader: CacheReader,
        info: EvaluatorInfo,
        label_counts: NDArray[np.uint64],
        index_to_label: dict[int, str],
    ):
        self._path = Path(path)
        self._cache = reader
        self._info = info
        self._label_counts = label_counts
        self._index_to_label = index_to_label

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
        cache_path = cls._generate_cache_path(path)
        cache = CacheReader.load(cache_path)

        # build evaluator meta
        (
            index_to_label,
            label_counts,
            info,
        ) = cls.generate_meta(cache.dataset, index_to_label_override)

        # read config
        metadata_path = cls._generate_metadata_path(path)
        with open(metadata_path, "r") as f:
            types = json.load(f)
            info.datum_metadata_types = types["datum"]

        return cls(
            path=path,
            reader=cache,
            info=info,
            label_counts=label_counts,
            index_to_label=index_to_label,
        )

    def filter(
        self,
        path: str | Path,
        filter_expr: Filter,
    ) -> Evaluator:
        """
        Filter classification cache.

        Parameters
        ----------
        path : str | Path
            Where to write the filtered cache.
        filter_expr : Filter
            A collection of filter expressions.

        Returns
        -------
        Evaluator
            A new evaluator object containing the filtered cache.
        """
        from valor_lite.classification.loader import Loader

        loader = Loader.create(
            path=path,
            batch_size=self.cache.batch_size,
            rows_per_file=self.cache.rows_per_file,
            compression=self.cache.compression,
            datum_metadata_types=self.info.datum_metadata_types,
        )
        for fragment in self.cache.dataset.get_fragments():
            tbl = fragment.to_table(filter=filter_expr.datums)

            columns = (
                "datum_id",
                "gt_label_id",
                "pd_label_id",
            )
            pairs = np.column_stack([tbl[col].to_numpy() for col in columns])

            n_pairs = pairs.shape[0]
            gt_ids = pairs[:, (0, 1)].astype(np.int64)
            pd_ids = pairs[:, (0, 2)].astype(np.int64)

            if filter_expr.groundtruths is not None:
                print(filter_expr.groundtruths)
                mask_valid_gt = np.zeros(n_pairs, dtype=np.bool_)
                gt_tbl = tbl.filter(filter_expr.groundtruths)
                print(gt_tbl)
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

            print(gt_ids)
            print(mask_valid_gt)

            if filter_expr.predictions is not None:
                mask_valid_pd = np.zeros(n_pairs, dtype=np.bool_)
                pd_tbl = tbl.filter(filter_expr.predictions)
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
            loader._cache.write_table(tbl)

        loader._cache.flush()
        if loader._cache.dataset.count_rows() == 0:
            raise EmptyCacheError()

        return Evaluator.load(
            path=loader.path,
            index_to_label_override=self._index_to_label,
        )

    def delete(self):
        """
        Delete classification cache.
        """
        from valor_lite.classification.loader import Loader

        Loader.delete(self.path)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def cache(self) -> CacheReader:
        return self._cache

    @property
    def info(self) -> EvaluatorInfo:
        return self._info

    @staticmethod
    def generate_meta(
        dataset: ds.Dataset,
        labels_override: dict[int, str] | None,
    ) -> tuple[dict[int, str], NDArray[np.uint64], EvaluatorInfo]:
        """
        Generate cache statistics.

        Parameters
        ----------
        dataset : Dataset
            Valor cache.
        labels_override : dict[int, str], optional
            Optional labels override. Use when operating over filtered data.

        Returns
        -------
        labels : dict[int, str]
            Mapping of label ID's to label values.
        label_counts : NDArray[np.uint64]
            Array of size (n_labels, 2) containing counts of ground truths and predictions per label.
        info : EvaluatorInfo
            Evaluator cache details.
        """
        labels = labels_override if labels_override else {}
        info = EvaluatorInfo()

        for fragment in dataset.get_fragments():
            tbl = fragment.to_table()
            columns = (
                "datum_id",
                "gt_label_id",
                "pd_label_id",
            )
            ids = np.column_stack(
                [tbl[col].to_numpy() for col in columns]
            ).astype(np.int64)

            # count number of rows
            info.number_of_rows += int(tbl.shape[0])

            # count unique datums
            datum_ids = np.unique(ids[:, 0])
            info.number_of_datums += int(datum_ids.size)

            # get gt labels
            gt_label_ids = ids[:, 1]
            gt_label_ids, gt_indices = np.unique(
                gt_label_ids, return_index=True
            )
            gt_labels = tbl["gt_label"].take(gt_indices).to_pylist()
            gt_labels = dict(zip(gt_label_ids.astype(int).tolist(), gt_labels))
            gt_labels.pop(-1, None)
            labels.update(gt_labels)

            # get pd labels
            pd_label_ids = ids[:, 2]
            pd_label_ids, pd_indices = np.unique(
                pd_label_ids, return_index=True
            )
            pd_labels = tbl["pd_label"].take(pd_indices).to_pylist()
            pd_labels = dict(zip(pd_label_ids.astype(int).tolist(), pd_labels))
            pd_labels.pop(-1, None)
            labels.update(pd_labels)

        # post-process
        labels.pop(-1, None)

        # count ground truth and prediction label occurences
        n_labels = len(labels)
        label_counts = np.zeros((n_labels, 2), dtype=np.uint64)
        for fragment in dataset.get_fragments():
            tbl = fragment.to_table()
            columns = (
                "datum_id",
                "gt_label_id",
                "pd_label_id",
            )
            ids = np.column_stack(
                [tbl[col].to_numpy() for col in columns]
            ).astype(np.int64)

            # count unique gt labels
            unique_gts = np.unique(ids[:, (0, 1)], axis=0)
            unique_gt_labels, gt_label_counts = np.unique(
                unique_gts[:, 1], return_counts=True
            )
            label_counts[unique_gt_labels, 0] += gt_label_counts.astype(
                np.uint64
            )

            # count unique pd labels
            unique_pds = np.unique(ids[:, (0, 2)], axis=0)
            unique_pd_labels, pd_label_counts = np.unique(
                unique_pds[:, 1], return_counts=True
            )
            label_counts[unique_pd_labels, 1] += pd_label_counts.astype(
                np.uint64
            )

        # complete info object
        info.number_of_labels = len(labels)

        return labels, label_counts, info

    def iterate_fragments(self):
        columns = [
            "datum_id",
            "gt_label_id",
            "pd_label_id",
            "score",
            "winner",
        ]
        for fragment in self.cache.dataset.get_fragments():
            tbl = fragment.to_table(columns=columns)
            ids = np.column_stack([tbl[col].to_numpy() for col in columns[:3]])
            scores = tbl["score"].to_numpy()
            winners = tbl["winner"].to_numpy()
            yield ids, scores, winners

    def iterate_fragments_with_table(self):
        columns = [
            "datum_id",
            "gt_label_id",
            "pd_label_id",
            "score",
            "winner",
        ]
        for fragment in self.cache.dataset.get_fragments():
            tbl = fragment.to_table()
            ids = np.column_stack([tbl[col].to_numpy() for col in columns[:3]])
            scores = tbl["score"].to_numpy()
            winners = tbl["winner"].to_numpy()
            yield tbl, ids, scores, winners

    def iterate_sorted_chunks(
        self,
        rows_per_chunk: int,
        read_batch_size: int,
    ):
        """
        Iterate through sorted pairs in chunks.

        Parameters
        ----------
        rows_per_chunk : int
            The number of sorted rows to return in each chunk.
        read_batch_size : int
            The maximum number of rows to load in-memory per file.

        Yields
        ------
        NDArray[float64]
            The next chunk of pairs in a sequence sorted by descending score.
        """
        id_columns = [
            "datum_id",
            "gt_label_id",
            "pd_label_id",
        ]
        if self.cache.num_dataset_files == 1:
            pf = pq.ParquetFile(self.cache.dataset_files[0])
            tbl = pf.read()
            ids = np.column_stack([tbl[col].to_numpy() for col in id_columns])
            scores = tbl["score"].to_numpy()
            winners = tbl["winner"].to_numpy()

            n_pairs = ids.shape[0]
            n_chunks = n_pairs // rows_per_chunk
            for i in range(n_chunks):
                lb = i * rows_per_chunk
                ub = i * (rows_per_chunk + 1)
                yield ids[lb:ub], scores[lb:ub], winners[lb:ub]
            lb = n_chunks * rows_per_chunk
            yield ids[lb:], scores[lb:], winners[lb:]
        else:

            def generate_heap_item(batches, batch_idx, row_idx):
                score = batches[batch_idx]["score"][row_idx].as_py()
                gidx = batches[batch_idx]["gt_label_id"][row_idx].as_py()
                pidx = batches[batch_idx]["pd_label_id"][row_idx].as_py()
                return (
                    -score,
                    pidx,
                    gidx,
                    batch_idx,
                    row_idx,
                )

            # merge sorted rows
            heap = []
            batch_iterators = []
            batches = []
            for batch_idx, path in enumerate(self.cache.dataset_files):
                pf = pq.ParquetFile(path)
                batch_iter = pf.iter_batches(batch_size=read_batch_size)
                batch_iterators.append(batch_iter)
                batches.append(next(batch_iterators[batch_idx], None))
                if (
                    batches[batch_idx] is not None
                    and len(batches[batch_idx]) > 0
                ):
                    heapq.heappush(
                        heap, generate_heap_item(batches, batch_idx, 0)
                    )

            ids_buffer = []
            scores_buffer = []
            winners_buffer = []
            while heap:
                _, _, _, batch_idx, row_idx = heapq.heappop(heap)
                row_table = batches[batch_idx].slice(row_idx, 1)

                ids_buffer.append(
                    np.column_stack(
                        [row_table[col].to_numpy() for col in id_columns]
                    )
                )
                scores_buffer.append(row_table["score"].to_numpy())
                winners_buffer.append(
                    row_table["winner"].to_numpy(zero_copy_only=False)
                )
                if len(ids_buffer) >= rows_per_chunk:
                    ids = np.concatenate(ids_buffer, axis=0)
                    scores = np.concatenate(scores_buffer, axis=0)
                    winners = np.concatenate(winners_buffer, axis=0)
                    yield ids, scores, winners
                    ids_buffer, scores_buffer, winners_buffer = [], [], []

                row_idx += 1
                if row_idx < len(batches[batch_idx]):
                    heapq.heappush(
                        heap,
                        generate_heap_item(batches, batch_idx, row_idx),
                    )
                else:
                    batches[batch_idx] = next(batch_iterators[batch_idx], None)
                    if (
                        batches[batch_idx] is not None
                        and len(batches[batch_idx]) > 0
                    ):
                        heapq.heappush(
                            heap, generate_heap_item(batches, batch_idx, 0)
                        )
            if len(ids_buffer) > 0:
                ids = np.concatenate(ids_buffer, axis=0)
                scores = np.concatenate(scores_buffer, axis=0)
                winners = np.concatenate(winners_buffer, axis=0)
                yield ids, scores, winners

    def compute_precision_recall_rocauc(
        self,
        score_thresholds: list[float] = [0.0],
        hardmax: bool = True,
        *_,
        rows_per_chunk: int = 10_000,
        read_batch_size: int = 1_000,
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
        cumulative_fp = np.zeros((n_labels, 1), dtype=np.uint64)
        cumulative_tp = np.zeros((n_labels, 1), dtype=np.uint64)
        rocauc = np.zeros(n_labels, dtype=np.float64)
        counts = np.zeros((n_scores, n_labels, 4), dtype=np.uint64)

        for ids, scores, winners in self.iterate_sorted_chunks(
            rows_per_chunk=rows_per_chunk,
            read_batch_size=read_batch_size,
        ):
            batch_rocauc, cumulative_fp, cumulative_tp = compute_rocauc(
                ids=ids,
                scores=scores,
                gt_count_per_label=self._label_counts[:, 0],
                pd_count_per_label=self._label_counts[:, 1],
                n_datums=self.info.number_of_datums,
                n_labels=self.info.number_of_labels,
                prev_cumulative_fp=cumulative_fp,
                prev_cumulative_tp=cumulative_tp,
            )
            rocauc += batch_rocauc

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
        mean_rocauc = rocauc.mean()

        return unpack_precision_recall_rocauc_into_metric_lists(
            counts=counts,
            precision=precision,
            recall=recall,
            accuracy=accuracy,
            f1_score=f1_score,
            rocauc=rocauc,
            mean_rocauc=mean_rocauc,
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
        for ids, scores, winners in self.iterate_fragments():
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
        for tbl, ids, scores, winners in self.iterate_fragments_with_table():
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
        for tbl, ids, scores, winners in self.iterate_fragments_with_table():
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
