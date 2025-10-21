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

    def __getitem__(self, key: str):
        return getattr(self, key)


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
        self._detailed_path = self._path / "detailed"
        self._ranked_path = self._path / "ranked"
        self._metadata_path = self._path / "metadata.json"

        # link cache
        self._dataset = ds.dataset(self._detailed_path, format="parquet")

        # build evaluator meta
        (
            self._index_to_label,
            self._number_of_groundtruths_per_label,
            self._info,
        ) = self.generate_meta(self._dataset, labels_override)

        # read config
        with open(self._metadata_path, "r") as f:
            types = json.load(f)
            self._info.datum_metadata_types = types["datum"]
            self._info.groundtruth_metadata_types = types["groundtruth"]
            self._info.prediction_metadata_types = types["prediction"]
        with open(self._detailed_path / ".cfg", "r") as f:
            cfg = json.load(f)
            self._detailed_batch_size = cfg["batch_size"]
            self._detailed_rows_per_file = cfg["rows_per_file"]
            self._detailed_compression = cfg["compression"]

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
    def detailed(self) -> ds.Dataset:
        return self._dataset

    @property
    def ranked(self) -> ds.Dataset:
        return ds.dataset(self._ranked_path, format="parquet")

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
        from valor_lite.object_detection.loader import Loader

        return Loader.filter(
            directory=directory,
            name=name,
            evaluator=self,
            filter_expr=filter_expr,
        )

    def rank(
        self,
        where: str | Path,
        rows_per_file: int | None = None,
        compression: str | None = None,
        write_batch_size: int | None = None,
        read_batch_size: int = 1_000,
    ):
        """
        Create a ranked pair cache.

        Parameters
        ----------
        where : str | Path
            Where to store the ranked cache.
        rows_per_file : int, optional
            Sets the maximum number of rows per file. Defaults to value from detailed cache.
        compression : str, optional
            Sets the compression method. Defaults to value from detailed cache.
        write_batch_size : int, optional
            Sets the batch size for writing. Defaults to value from detailed cache.
        read_batch_size : int, default=1_000
            Sets the batch size for reading. Defaults to 1_000.
        """

        n_labels = len(self._index_to_label)
        detailed_cache = CacheReader(self._detailed_path)

        write_batch_size = (
            write_batch_size
            if write_batch_size is not None
            else detailed_cache.batch_size
        )
        rows_per_file = (
            rows_per_file
            if rows_per_file is not None
            else detailed_cache.rows_per_file
        )
        compression = (
            compression
            if compression is not None
            else detailed_cache.compression
        )

        with CacheWriter(
            where=where,
            schema=self.ranked_schema,
            batch_size=write_batch_size,
            rows_per_file=rows_per_file,
            compression=compression,
        ) as ranked_cache:
            if detailed_cache.num_dataset_files == 1:
                pf = pq.ParquetFile(detailed_cache.dataset_files[0])
                tbl = pf.read()
                ranked_tbl = rank_table(tbl, n_labels)
                ranked_cache.write_table(ranked_tbl)
                return

            pruned_detailed_columns = [
                field.name for field in self._pruned_schema
            ]
            with tempfile.TemporaryDirectory() as tmpdir:

                # rank individual files
                tmpfiles = []
                for idx, fragment in enumerate(self._dataset.get_fragments()):
                    path = Path(tmpdir) / f"{idx:06d}.parquet"
                    tbl = fragment.to_table(columns=pruned_detailed_columns)
                    ranked_tbl = rank_table(tbl, n_labels)
                    pq.write_table(ranked_tbl, path)
                    tmpfiles.append(path)

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
                for batch_idx, path in enumerate(tmpfiles):
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

                while heap:
                    _, _, batch_idx, row_idx = heapq.heappop(heap)
                    row_table = batches[batch_idx].slice(row_idx, 1)
                    ranked_cache.write_batch(row_table)
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
            dataset=self.ranked,
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
            dataset=self._dataset,
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
            dataset=self._dataset,
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
            dataset=self._dataset,
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
