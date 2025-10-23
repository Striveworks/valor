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
    compute_confusion_matrix,
    compute_precision_recall,
    compute_rocauc,
)
from valor_lite.classification.metric import Metric, MetricType
from valor_lite.classification.utilities import (
    unpack_confusion_matrix_into_metric_list,
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
        self._cache_path = self._path / "counts"
        self._metadata_path = self._path / "metadata.json"

        # link cache
        self._dataset = ds.dataset(self._cache_path, format="parquet")

        # build evaluator meta
        (
            self._index_to_label,
            self._counts_per_label,
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
        confusion_matrix : NDArray[np.uint64]
            Array of size (n_labels + 1, n_labels + 1) containing pair counts.
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
        matrix = np.zeros((n_labels + 1, n_labels + 1), dtype=np.uint64)
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
            counts = tbl["count"].to_numpy()

            mask_null_gts = ids[:, 1] == -1
            mask_null_pds = ids[:, 2] == -1
            matrix[0, 0] = counts[mask_null_gts & mask_null_pds].sum()
            for idx in range(n_labels):
                mask_gts = ids[:, 1] == idx
                for pidx in range(n_labels):
                    mask_pds = ids[:, 2] == pidx
                    matrix[idx + 1, pidx + 1] = counts[
                        mask_gts & mask_pds
                    ].sum()

                mask_unmatched_gts = mask_gts & mask_null_pds
                matrix[idx + 1, 0] = counts[mask_unmatched_gts].sum()
                mask_unmatched_pds = mask_null_gts & (ids[:, 2] == idx)
                matrix[0, idx + 1] = counts[mask_unmatched_pds].sum()

        # complete info object
        info.number_of_labels = len(labels)
        info.number_of_pixels = matrix.sum()
        info.number_of_groundtruth_pixels = matrix[1:, :].sum()
        info.number_of_prediction_pixels = matrix[:, 1:].sum()

        return labels, matrix, info

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
        cumulative_fp = 0
        cumulative_tp = 0

        for ids, scores, winners in self.iterate_sorted_chunks(
            rows_per_chunk=rows_per_chunk,
            read_batch_size=read_batch_size,
        ):
            # calculate ROCAUC
            rocauc, mean_rocauc = compute_rocauc(
                scores=scores,
                gt_count_per_label=label_metadata[:, 0],
                pd_count_per_label=label_metadata[:, 1],
                n_datums=n_datums,
                n_labels=n_labels,
                mask_matching_labels=mask_matching_labels,
                pd_labels=pd_labels,
            )

            results = compute_precision_recall_rocauc(
                pairs=detailed_pairs,
                label_metadata=label_metadata,
                score_thresholds=np.array(score_thresholds),
                hardmax=hardmax,
                n_datums=n_datums,
            )
        return unpack_precision_recall_rocauc_into_metric_lists(
            results=results,
            score_thresholds=score_thresholds,
            hardmax=hardmax,
            label_metadata=label_metadata,
            index_to_label=self.index_to_label,
        )

    def compute_confusion_matrix(
        self,
        score_thresholds: list[float] = [0.0],
        hardmax: bool = True,
        filter_: Filter | None = None,
    ) -> list[Metric]:
        """
        Computes a detailed confusion matrix..

        Parameters
        ----------
        score_thresholds : list[float]
            A list of score thresholds to compute metrics over.
        hardmax : bool
            Toggles whether a hardmax is applied to predictions.
        filter_ : Filter, optional
            Applies a filter to the internal cache.

        Returns
        -------
        list[Metric]
            A list of confusion matrices.
        """
        # apply filters
        if filter_ is not None:
            detailed_pairs, _ = self.filter(filter_=filter_)
        else:
            detailed_pairs = self._detailed_pairs

        if detailed_pairs.size == 0:
            return list()

        result = compute_confusion_matrix(
            detailed_pairs=detailed_pairs,
            score_thresholds=np.array(score_thresholds),
            hardmax=hardmax,
        )
        return unpack_confusion_matrix_into_metric_list(
            detailed_pairs=detailed_pairs,
            result=result,
            score_thresholds=score_thresholds,
            index_to_datum_id=self.index_to_datum_id,
            index_to_label=self.index_to_label,
        )

    def evaluate(
        self,
        score_thresholds: list[float] = [0.0],
        hardmax: bool = True,
        filter_: Filter | None = None,
    ) -> dict[MetricType, list[Metric]]:
        """
        Computes a detailed confusion matrix..

        Parameters
        ----------
        score_thresholds : list[float]
            A list of score thresholds to compute metrics over.
        hardmax : bool
            Toggles whether a hardmax is applied to predictions.
        filter_ : Filter, optional
            Applies a filter to the internal cache.

        Returns
        -------
        dict[MetricType, list[Metric]]
            Lists of metrics organized by metric type.
        """
        metrics = self.compute_precision_recall_rocauc(
            score_thresholds=score_thresholds,
            hardmax=hardmax,
            filter_=filter_,
        )
        metrics[MetricType.ConfusionMatrix] = self.compute_confusion_matrix(
            score_thresholds=score_thresholds,
            hardmax=hardmax,
            filter_=filter_,
        )
        return metrics
