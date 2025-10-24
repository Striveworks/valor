import heapq
import json
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from numpy.typing import NDArray

from valor_lite.cache import CacheReader, CacheWriter, DataType
from valor_lite.object_detection.computation import (
    compute_average_precision,
    compute_average_recall,
    compute_confusion_matrix,
    compute_counts,
    compute_pair_classifications,
    compute_precision_recall_f1,
    rank_table,
)
from valor_lite.object_detection.format import PathFormatter
from valor_lite.object_detection.metric import Metric, MetricType
from valor_lite.object_detection.utilities import (
    create_empty_confusion_matrix_with_examples,
    create_mapping,
    unpack_confusion_matrix,
    unpack_confusion_matrix_with_examples,
    unpack_examples,
    unpack_precision_recall_into_metric_lists,
)


@dataclass
class EvaluatorInfo:
    number_of_datums: int = 0
    number_of_groundtruth_annotations: int = 0
    number_of_prediction_annotations: int = 0
    number_of_labels: int = 0
    number_of_rows: int = 0
    datum_metadata_types: dict[str, DataType] | None = None
    groundtruth_metadata_types: dict[str, DataType] | None = None
    prediction_metadata_types: dict[str, DataType] | None = None


@dataclass
class Filter:
    datums: pc.Expression | None = None
    groundtruths: pc.Expression | None = None
    predictions: pc.Expression | None = None


class Evaluator(PathFormatter):
    def __init__(
        self,
        path: str | Path,
        detailed_cache: CacheReader,
        ranked_cache: CacheReader,
        info: EvaluatorInfo,
        index_to_label: dict[int, str],
        number_of_groundtruths_per_label: NDArray[np.uint64],
    ):
        self._path = Path(path)
        self._detailed_cache = detailed_cache
        self._ranked_cache = ranked_cache
        self._info = info
        self._index_to_label = index_to_label
        self._number_of_groundtruths_per_label = (
            number_of_groundtruths_per_label
        )

    @classmethod
    def create(
        cls,
        path: str | Path,
        batch_size: int = 1_000,
        index_to_label_override: dict[int, str] | None = None,
    ):
        """
        Create a ranked pair cache.

        Parameters
        ----------
        path : str | Path
            Where to store the evaluator cache.
        batch_size : int, default=1_000
            Sets the batch size for reading. Defaults to 1_000.
        """
        detailed_cache = CacheReader(cls._generate_detailed_cache_path(path))

        # build evaluator meta
        (
            index_to_label,
            number_of_groundtruths_per_label,
            info,
        ) = cls.generate_meta(detailed_cache.dataset, index_to_label_override)

        # read config
        metadata_path = cls._generate_metadata_path(path)
        with open(metadata_path, "r") as f:
            types = json.load(f)
            info.datum_metadata_types = types["datum_metadata_types"]
            info.groundtruth_metadata_types = types[
                "groundtruth_metadata_types"
            ]
            info.prediction_metadata_types = types["prediction_metadata_types"]

        # create ranked cache schema
        annotation_metadata_keys = {
            *(
                set(info.groundtruth_metadata_types.keys())
                if info.groundtruth_metadata_types
                else {}
            ),
            *(
                set(info.prediction_metadata_types.keys())
                if info.prediction_metadata_types
                else {}
            ),
        }
        pruned_schema = pa.schema(
            [
                field
                for field in detailed_cache.schema
                if field.name not in annotation_metadata_keys
            ]
        )
        ranked_schema = pruned_schema.append(
            pa.field("iou_prev", pa.float64())
        )
        ranked_schema = ranked_schema.append(
            pa.field("high_score", pa.bool_())
        )

        n_labels = len(index_to_label)

        with CacheWriter.create(
            path=cls._generate_ranked_cache_path(path),
            schema=ranked_schema,
            batch_size=detailed_cache.batch_size,
            rows_per_file=detailed_cache.rows_per_file,
            compression=detailed_cache.compression,
        ) as ranked_cache:
            if detailed_cache.num_dataset_files == 1:
                pf = pq.ParquetFile(detailed_cache.dataset_files[0])
                tbl = pf.read()
                ranked_tbl = rank_table(tbl, n_labels)
                ranked_cache.write_table(ranked_tbl)
            else:
                pruned_detailed_columns = [
                    field.name for field in pruned_schema
                ]
                with tempfile.TemporaryDirectory() as tmpdir:

                    # rank individual files
                    tmpfiles = []
                    for idx, fragment in enumerate(
                        detailed_cache.dataset.get_fragments()
                    ):
                        fragment_path = Path(tmpdir) / f"{idx:06d}.parquet"
                        tbl = fragment.to_table(
                            columns=pruned_detailed_columns
                        )
                        ranked_tbl = rank_table(tbl, n_labels)
                        pq.write_table(ranked_tbl, fragment_path)
                        tmpfiles.append(fragment_path)

                    def generate_heap_item(batches, batch_idx, row_idx):
                        score = batches[batch_idx]["score"][row_idx].as_py()
                        iou = batches[batch_idx]["iou"][row_idx].as_py()
                        return (
                            -score,
                            -iou,
                            batch_idx,
                            row_idx,
                        )

                    # merge sorted rows
                    heap = []
                    batch_iterators = []
                    batches = []
                    for batch_idx, batch_path in enumerate(tmpfiles):
                        pf = pq.ParquetFile(batch_path)
                        batch_iter = pf.iter_batches(batch_size=batch_size)
                        batch_iterators.append(batch_iter)
                        batches.append(next(batch_iterators[batch_idx], None))
                        if (
                            batches[batch_idx] is not None
                            and len(batches[batch_idx]) > 0
                        ):
                            heapq.heappush(
                                heap, generate_heap_item(batches, batch_idx, 0)
                            )

                    while heap:
                        _, _, batch_idx, row_idx = heapq.heappop(heap)
                        row_table = batches[batch_idx].slice(row_idx, 1)
                        ranked_cache.write_batch(row_table)
                        row_idx += 1
                        if row_idx < len(batches[batch_idx]):
                            heapq.heappush(
                                heap,
                                generate_heap_item(
                                    batches, batch_idx, row_idx
                                ),
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
                                    heap,
                                    generate_heap_item(batches, batch_idx, 0),
                                )

        ranked_cache = CacheReader(cls._generate_ranked_cache_path(path))
        return cls(
            path=path,
            detailed_cache=detailed_cache,
            ranked_cache=ranked_cache,
            info=info,
            index_to_label=index_to_label,
            number_of_groundtruths_per_label=number_of_groundtruths_per_label,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        index_to_label_override: dict[int, str] | None = None,
    ):
        detailed_cache = CacheReader(cls._generate_detailed_cache_path(path))
        ranked_cache = CacheReader(cls._generate_ranked_cache_path(path))

        # build evaluator meta
        (
            index_to_label,
            number_of_groundtruths_per_label,
            info,
        ) = cls.generate_meta(detailed_cache.dataset, index_to_label_override)

        # read config
        metadata_path = cls._generate_metadata_path(path)
        with open(metadata_path, "r") as f:
            types = json.load(f)
            info.datum_metadata_types = types["datum_metadata_types"]
            info.groundtruth_metadata_types = types[
                "groundtruth_metadata_types"
            ]
            info.prediction_metadata_types = types["prediction_metadata_types"]

        return cls(
            path=path,
            detailed_cache=detailed_cache,
            ranked_cache=ranked_cache,
            info=info,
            index_to_label=index_to_label,
            number_of_groundtruths_per_label=number_of_groundtruths_per_label,
        )

    @property
    def path(self) -> Path:
        return self._path

    @property
    def detailed(self) -> CacheReader:
        return self._detailed_cache

    @property
    def ranked(self) -> CacheReader:
        return self._ranked_cache

    @property
    def info(self) -> EvaluatorInfo:
        return self._info

    @staticmethod
    def generate_meta(
        dataset: ds.Dataset,
        labels_override: dict[int, str] | None = None,
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
        number_of_groundtruths_per_label : NDArray[np.uint64]
            Array of size (n_labels,) containing ground truth counts.
        info : EvaluatorInfo
            Evaluator cache details.
        """
        gt_counts_per_lbl = defaultdict(int)
        labels = labels_override if labels_override else {}
        info = EvaluatorInfo()

        for fragment in dataset.get_fragments():
            tbl = fragment.to_table()
            columns = (
                "datum_id",
                "gt_id",
                "pd_id",
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

            # count unique groundtruths
            gt_ids = ids[:, 1]
            gt_ids = np.unique(gt_ids[gt_ids >= 0])
            info.number_of_groundtruth_annotations += int(gt_ids.shape[0])

            # count unique predictions
            pd_ids = ids[:, 2]
            pd_ids = np.unique(pd_ids[pd_ids >= 0])
            info.number_of_prediction_annotations += int(pd_ids.shape[0])

            # get gt labels
            gt_label_ids = ids[:, 3]
            gt_label_ids, gt_indices = np.unique(
                gt_label_ids, return_index=True
            )
            gt_labels = tbl["gt_label"].take(gt_indices).to_pylist()
            gt_labels = dict(zip(gt_label_ids.astype(int).tolist(), gt_labels))
            gt_labels.pop(-1, None)
            labels.update(gt_labels)

            # get pd labels
            pd_label_ids = ids[:, 4]
            pd_label_ids, pd_indices = np.unique(
                pd_label_ids, return_index=True
            )
            pd_labels = tbl["pd_label"].take(pd_indices).to_pylist()
            pd_labels = dict(zip(pd_label_ids.astype(int).tolist(), pd_labels))
            pd_labels.pop(-1, None)
            labels.update(pd_labels)

            # count gts per label
            gts = ids[:, (1, 3)].astype(np.int64)
            unique_ann = np.unique(gts[gts[:, 0] >= 0], axis=0)
            unique_labels, label_counts = np.unique(
                unique_ann[:, 1], return_counts=True
            )
            for label_id, count in zip(unique_labels, label_counts):
                gt_counts_per_lbl[int(label_id)] += int(count)

        # post-process
        labels.pop(-1, None)

        # complete info object
        info.number_of_labels = len(labels)

        # convert gt counts to numpy
        number_of_groundtruths_per_label = np.zeros(
            len(labels), dtype=np.uint64
        )
        for k, v in gt_counts_per_lbl.items():
            number_of_groundtruths_per_label[int(k)] = v

        return labels, number_of_groundtruths_per_label, info

    @staticmethod
    def iterate_pairs(
        dataset: ds.Dataset,
        columns: list[str] | None = None,
    ):
        for fragment in dataset.get_fragments():
            tbl = fragment.to_table(columns=columns)
            yield np.column_stack(
                [tbl.column(i).to_numpy() for i in range(tbl.num_columns)]
            )

    @staticmethod
    def iterate_pairs_with_table(
        dataset: ds.Dataset,
        columns: list[str] | None = None,
    ):
        for fragment in dataset.get_fragments():
            tbl = fragment.to_table()
            columns = columns if columns else tbl.columns
            yield tbl, np.column_stack(
                [tbl[col].to_numpy() for col in columns]
            )

    def filter(
        self,
        path: str | Path,
        filter_expr: Filter,
    ) -> "Evaluator":
        """
        Filter evaluator cache.

        Parameters
        ----------
        path : str | Path
            Where to store the filtered cache.
        filter_expr : Filter
            An object containing filter expressions.

        Returns
        -------
        Evaluator
            A new evaluator object containing the filtered cache.
        """
        from valor_lite.object_detection.loader import Loader

        return Loader.filter(
            path=path,
            evaluator=self,
            filter_expr=filter_expr,
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
        for pairs in self.iterate_pairs(
            dataset=self.ranked.dataset,
            columns=[
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
        for pairs in self.iterate_pairs(
            dataset=self.detailed.dataset,
            columns=[
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
        for tbl, pairs in self.iterate_pairs_with_table(
            dataset=self.detailed.dataset,
            columns=[
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
        for tbl, pairs in self.iterate_pairs_with_table(
            dataset=self.detailed.dataset,
            columns=[
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
