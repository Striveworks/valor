import heapq
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
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
from valor_lite.classification.metric import Metric, MetricType
from valor_lite.classification.utilities import (
    create_empty_confusion_matrix_with_examples,
    create_mapping,
    unpack_confusion_matrix,
    unpack_confusion_matrix_with_examples,
    unpack_examples,
    unpack_precision_recall_rocauc_into_metric_lists,
)


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


class Evaluator:
    def __init__(
        self,
        name: str = "default",
        directory: str | Path = ".valor",
        labels_override: dict[int, str] | None = None,
    ):
        self._directory = Path(directory)
        self._name = name
        self._path = self._directory / name
        self._cache_path = self._path / "cache"
        self._metadata_path = self._path / "metadata.json"

        # link cache
        self._dataset = ds.dataset(self._cache_path, format="parquet")

        # build evaluator meta
        (
            self._index_to_label,
            self._label_counts,
            self._info,
        ) = self.generate_meta(self._dataset, labels_override)

        # read config
        with open(self._metadata_path, "r") as f:
            types = json.load(f)
            self._info.datum_metadata_types = types["datum"]
        with open(self._cache_path / ".cfg", "r") as f:
            cfg = json.load(f)
            self._detailed_batch_size = cfg["batch_size"]
            self._detailed_rows_per_file = cfg["rows_per_file"]
            self._detailed_compression = cfg["compression"]

    @property
    def dataset(self) -> ds.Dataset:
        return self._dataset

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

        # create confusion matrix
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
            unique_gts = np.unique(ids[:, (0, 1)], axis=0)
            unique_pds = np.unique(ids[:, (0, 2)], axis=0)
            unique_gt_labels, gt_label_counts = np.unique(
                unique_gts[:, 1], return_counts=True
            )
            unique_pd_labels, pd_label_counts = np.unique(
                unique_pds[:, 1], return_counts=True
            )
            label_counts[unique_gt_labels, 0] = gt_label_counts
            label_counts[unique_pd_labels, 1] = pd_label_counts

        # complete info object
        info.number_of_labels = len(labels)

        return labels, label_counts, info

    def filter(
        self,
        filter_expr: Filter,
        name: str | None = None,
        directory: str | Path | None = None,
    ) -> "Evaluator":
        """
        Filter evaluator cache.

        Parameters
        ----------
        filter_expr : Filter
            An object containing filter expressions.
        name : str, optional
            Filtered cache name.
        directory : str | Path, optional
            The directory to store the filtered cache.

        Returns
        -------
        Evaluator
            A new evaluator object containing the filtered cache.
        """
        name = name if name else "filtered"
        directory = directory if directory else self._directory
        from valor_lite.classification.loader import Loader

        return Loader.filter(
            name=name,
            directory=directory,
            evaluator=self,
            filter_expr=filter_expr,
        )

    def iterate_fragments(self):
        columns = [
            "datum_id",
            "gt_label_id",
            "pd_label_id",
            "score",
            "winner",
        ]
        for fragment in self.dataset.get_fragments():
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
        for fragment in self.dataset.get_fragments():
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
        with CacheReader(self._cache_path) as cache:
            if cache.num_dataset_files == 1:
                pf = pq.ParquetFile(cache.dataset_files[0])
                tbl = pf.read()
                ids = np.column_stack(
                    [tbl[col].to_numpy() for col in id_columns]
                )
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
                for batch_idx, path in enumerate(cache.dataset_files):
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
                    winners_buffer.append(row_table["winner"].to_numpy())
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
                        batches[batch_idx] = next(
                            batch_iterators[batch_idx], None
                        )
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
