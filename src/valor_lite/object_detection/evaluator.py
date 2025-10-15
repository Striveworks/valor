import heapq
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from numpy.typing import NDArray

from valor_lite.cache import CacheReader, CacheWriter, DataType
from valor_lite.object_detection.computation import (
    calculate_ranking_boundaries,
    compute_average_precision,
    compute_average_recall,
    compute_confusion_matrix,
    compute_counts,
    compute_pair_classifications,
    compute_precision_recall_f1,
    rank_pairs_returning_indices,
)
from valor_lite.object_detection.metric import Metric, MetricType
from valor_lite.object_detection.utilities import (
    unpack_confusion_matrix,
    unpack_examples,
    unpack_precision_recall_into_metric_lists,
)


def create_mapping(
    tbl: pa.Table,
    pairs: NDArray[np.float64],
    index: int,
    id_col: str,
    uid_col: str,
) -> dict[int, str]:
    col = pairs[:, index].astype(np.int64)
    values, indices = np.unique(col, return_index=True)
    indices = indices[values >= 0]
    return {
        tbl[id_col][idx].as_py(): tbl[uid_col][idx].as_py() for idx in indices
    }


@dataclass
class EvaluatorInfo:
    number_of_datums: int
    number_of_groundtruth_annotations: int
    number_of_prediction_annotations: int
    number_of_labels: int
    number_of_rows: int
    datum_metadata_types: dict[str, DataType] | None
    groundtruth_metadata_types: dict[str, DataType] | None
    prediction_metadata_types: dict[str, DataType] | None

    def __getitem__(self, key: str):
        return getattr(self, key)


class Evaluator:
    def __init__(
        self,
        directory: str | Path,
    ):
        self._dir = Path(directory)
        self._detailed_path = self._dir / Path("detailed")
        self._ranked_path = self._dir / Path("ranked")
        self._labels_path = self._dir / Path("labels.json")
        self._info_path = self._dir / Path("info.json")
        self._number_of_groundtruths_per_label_path = self._dir / Path(
            "groundtruths_per_label.json"
        )

        # read from file
        self._dataset = ds.dataset(self._detailed_path, format="parquet")

        # load id to label mapping
        with open(self._labels_path, "r") as f:
            labels = json.load(f)
            self._index_to_label = {int(k): v for k, v in labels.items()}

        # load label id to ground truth count mapping
        with open(self._number_of_groundtruths_per_label_path, "r") as f:
            self._number_of_groundtruths_per_label = np.zeros(
                len(self._index_to_label), dtype=np.uint64
            )
            for k, v in json.load(f).items():
                self._number_of_groundtruths_per_label[int(k)] = v

        # load evaluator info
        with open(self._info_path, "r") as f:
            self._info = EvaluatorInfo(**json.load(f))

        # create ranked cache schema
        annotation_metadata_keys = {
            *(
                set(self.info.groundtruth_metadata_types.keys())
                if self.info.groundtruth_metadata_types
                else {}
            ),
            *(
                set(self.info.prediction_metadata_types.keys())
                if self.info.prediction_metadata_types
                else {}
            ),
        }
        self._pruned_schema = pa.schema(
            [
                field
                for field in self._dataset.schema
                if field.name not in annotation_metadata_keys
            ]
        )
        self.ranked_schema = self._pruned_schema.append(
            pa.field("iou_prev", pa.float64())
        )
        self.ranked_schema = self.ranked_schema.append(
            pa.field("high_score", pa.bool_())
        )

    @property
    def _ranked(self):
        return ds.dataset(self._ranked_path, format="parquet")

    @property
    def info(self) -> EvaluatorInfo:
        return self._info

    def create_ranked_cache(
        self,
        where: str | Path,
    ):
        detailed_cache = CacheReader(self._detailed_path)

        cache = CacheWriter(
            where=where,
            schema=self.ranked_schema,
            batch_size=detailed_cache.batch_size,
            rows_per_file=detailed_cache.rows_per_file,
            compression=detailed_cache.compression,
        )
        sorting_args = [
            ("score", "descending"),
            ("iou", "descending"),
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

        if detailed_cache.num_files == 1:
            pf = pq.ParquetFile(detailed_cache.files[0])
            tbl = pf.read()
            sorted_tbl = tbl.sort_by(sorting_args)

            pairs = np.column_stack(
                [sorted_tbl[col].to_numpy() for col in numeric_columns]
            )
            pairs, indices = rank_pairs_returning_indices(pairs)
            ranked_tbl = sorted_tbl.take(indices)

            lower_iou_bound, winners = calculate_ranking_boundaries(
                pairs, number_of_labels=len(self._index_to_label)
            )

            column_data = pa.array(lower_iou_bound, type=pa.float64())
            column_field = pa.field("iou_prev", pa.float64())
            ranked_tbl = ranked_tbl.append_column(column_field, column_data)

            column_data = pa.array(winners, type=pa.bool_())
            column_field = pa.field("winner", pa.bool_())
            ranked_tbl = ranked_tbl.append_column(column_field, column_data)

            ranked_tbl = ranked_tbl.sort_by(sorting_args)
            cache.write_table(ranked_tbl)

        pruned_detailed_columns = [field.name for field in self._pruned_schema]

        with tempfile.TemporaryDirectory() as tmpdir:

            # sort individual files
            tmpfiles = []
            for idx, fragment in enumerate(self._dataset.get_fragments()):
                tbl = fragment.to_table(columns=pruned_detailed_columns)

                sorted_tbl = tbl.sort_by(sorting_args)

                path = Path(tmpdir) / f"{idx:06d}.parquet"

                pairs = np.column_stack(
                    [sorted_tbl[col].to_numpy() for col in numeric_columns]
                )
                pairs, indices = rank_pairs_returning_indices(pairs)
                ranked_tbl = sorted_tbl.take(indices)

                (
                    lower_iou_bound,
                    winning_predictions,
                ) = calculate_ranking_boundaries(
                    pairs, number_of_labels=len(self._index_to_label)
                )

                column_data = pa.array(lower_iou_bound, type=pa.float64())
                column_field = pa.field("iou_prev", pa.float64())
                ranked_tbl = ranked_tbl.append_column(
                    column_field, column_data
                )

                column_data = pa.array(winning_predictions, type=pa.bool_())
                column_field = pa.field("high_score", pa.bool_())
                ranked_tbl = ranked_tbl.append_column(
                    column_field, column_data
                )

                ranked_tbl = ranked_tbl.sort_by(sorting_args)
                pq.write_table(ranked_tbl, path)
                tmpfiles.append(path)

            # merge sorted rows
            heap = []
            batch_iterators = []
            batches = []
            for idx, path in enumerate(tmpfiles):
                pf = pq.ParquetFile(path)
                batch_iter = pf.iter_batches(
                    batch_size=detailed_cache.batch_size
                )
                batch_iterators.append(batch_iter)
                batches.append(next(batch_iterators[idx], None))
                if batches[idx] is not None and len(batches[idx]) > 0:
                    sort_args = [
                        -batches[idx][col][0].as_py()
                        if order == "descending"
                        else batches[idx][col][0].as_py()
                        for col, order in sorting_args
                    ]
                    heapq.heappush(heap, (*sort_args, idx, 0))

            while heap:
                item = heapq.heappop(heap)
                idx, row_idx = item[-2], item[-1]
                row_table = batches[idx].slice(row_idx, 1)
                cache.write_batch(row_table)

                row_idx += 1
                if row_idx < len(batches[idx]):
                    sort_args = [
                        batches[idx][col][row_idx].as_py()
                        if order == "ascending"
                        else -batches[idx][col][row_idx].as_py()
                        for col, order in sorting_args
                    ]
                    heapq.heappush(heap, (*sort_args, idx, row_idx))
                else:
                    batches[idx] = next(batch_iterators[idx], None)
                    if batches[idx] is not None and len(batches[idx]) > 0:
                        sort_args = [
                            batches[idx][col][0].as_py()
                            if order == "ascending"
                            else -batches[idx][col][0].as_py()
                            for col, order in sorting_args
                        ]
                        heapq.heappush(heap, (*sort_args, idx, 0))

    def iterate_pairs(
        self,
        dataset: ds.Dataset,
        columns=None,
        filter_expr=None,
    ):
        for fragment in dataset.get_fragments():
            yield fragment.to_table(
                columns=columns,
                filter=filter_expr,
            )

    def compute_precision_recall(
        self,
        iou_thresholds: list[float],
        score_thresholds: list[float],
        filter_expr=None,
    ) -> dict[MetricType, list[Metric]]:
        """
        Computes all metrics except for ConfusionMatrix

        Parameters
        ----------
        iou_thresholds : list[float]
            A list of IOU thresholds to compute metrics over.
        score_thresholds : list[float]
            A list of score thresholds to compute metrics over.
        filter_ : Filter, optional
            A collection of filter parameters and masks.

        Returns
        -------
        dict[MetricType, list]
            A dictionary mapping MetricType enumerations to lists of computed metrics.
        """
        if not iou_thresholds:
            raise ValueError("At least one IOU threshold must be passed.")
        elif not score_thresholds:
            raise ValueError("At least one score threshold must be passed.")

        # if filter_expr is None:
        ranked_dataset = ds.dataset(self._ranked_path, format="parquet")

        columns = [
            "datum_id",
            "gt_id",
            "pd_id",
            "gt_label_id",
            "pd_label_id",
            "iou",
            "score",
            "iou_prev",
            "winner",
        ]
        n_ious = len(iou_thresholds)
        n_scores = len(score_thresholds)
        n_labels = len(self._index_to_label)

        counts = np.zeros((n_ious, n_scores, 3, n_labels), dtype=np.uint64)
        pr_curve = np.zeros((n_ious, n_labels, 101, 2), dtype=np.float64)
        running_counts = np.ones((n_ious, n_labels, 2), dtype=np.uint64)
        for tbl in self.iterate_pairs(
            dataset=ranked_dataset,
            columns=columns,
            filter_expr=filter_expr,
        ):
            pairs = np.column_stack([tbl[col].to_numpy() for col in columns])
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
        filter_expr=None,
    ) -> list[Metric]:
        """
        Computes confusion matrices at various thresholds.

        Parameters
        ----------
        iou_thresholds : list[float]
            A list of IOU thresholds to compute metrics over.
        score_thresholds : list[float]
            A list of score thresholds to compute metrics over.
        filter_ : Filter, optional
            A collection of filter parameters and masks.

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
        columns = [
            "datum_id",
            "gt_id",
            "pd_id",
            "gt_label_id",
            "pd_label_id",
            "iou",
            "score",
        ]
        for tbl in self.iterate_pairs(
            dataset=self._dataset,
            columns=columns,
            filter_expr=filter_expr,
        ):
            pairs = np.column_stack(
                [tbl.column(i).to_numpy() for i in range(tbl.num_columns)]
            )
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
        filter_expr=None,
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
        filter_ : Filter, optional
            A collection of filter parameters and masks.

        Returns
        -------
        list[Metric]
            List of confusion matrices per threshold pair.
        """
        if not iou_thresholds:
            raise ValueError("At least one IOU threshold must be passed.")
        elif not score_thresholds:
            raise ValueError("At least one score threshold must be passed.")

        pairs = None
        mask_tp = None
        mask_fp = None
        mask_fn = None
        index_to_datum_id = {}
        index_to_groundtruth_id = {}
        index_to_prediction_id = {}

        columns = [
            "datum_id",
            "gt_id",
            "pd_id",
            "gt_label_id",
            "pd_label_id",
            "iou",
            "score",
        ]
        for tbl in self.iterate_pairs(
            dataset=self._dataset,
            columns=columns,
            filter_expr=filter_expr,
        ):
            batch_pairs = np.column_stack(
                [tbl.column(i).to_numpy() for i in range(tbl.num_columns)]
            )
            if batch_pairs.size == 0:
                continue

            # extract UIDs
            index_to_datum_id.update(
                create_mapping(tbl, batch_pairs, 0, "datum_id", "datum_uid")
            )
            index_to_groundtruth_id.update(
                create_mapping(tbl, batch_pairs, 1, "gt_id", "gt_uid")
            )
            index_to_prediction_id.update(
                create_mapping(tbl, batch_pairs, 2, "pd_id", "pd_uid")
            )

            (
                batch_mask_tp,
                batch_mask_fp_fn_misclf,
                batch_mask_fp_unmatched,
                batch_mask_fn_unmatched,
            ) = compute_pair_classifications(
                detailed_pairs=batch_pairs,
                iou_thresholds=np.array(iou_thresholds),
                score_thresholds=np.array(score_thresholds),
            )

            batch_mask_fn = batch_mask_fp_fn_misclf | batch_mask_fn_unmatched
            batch_mask_fp = batch_mask_fp_fn_misclf | batch_mask_fp_unmatched

            pairs = (
                np.concatenate([pairs, batch_pairs]) if pairs else batch_pairs
            )
            mask_tp = (
                np.concatenate([mask_tp, batch_mask_tp])
                if mask_tp
                else batch_mask_tp
            )
            mask_fp = (
                np.concatenate([mask_fp, batch_mask_fp])
                if mask_fp
                else batch_mask_fp
            )
            mask_fn = (
                np.concatenate([mask_fn, batch_mask_fn])
                if mask_fn
                else batch_mask_fn
            )

        if (
            pairs is None
            or mask_tp is None
            or mask_fp is None
            or mask_fn is None
        ):
            raise Exception

        return unpack_examples(
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

    def evaluate(
        self,
        iou_thresholds: list[float] = [0.1, 0.5, 0.75],
        score_thresholds: list[float] = [0.5, 0.75, 0.9],
        filter_expr=None,
    ) -> dict[MetricType, list[Metric]]:
        metrics = self.compute_precision_recall(
            iou_thresholds=iou_thresholds,
            score_thresholds=score_thresholds,
            filter_expr=filter_expr,
        )
        metrics[MetricType.ConfusionMatrix] = self.compute_confusion_matrix(
            iou_thresholds=iou_thresholds,
            score_thresholds=score_thresholds,
            filter_expr=filter_expr,
        )
        return metrics
