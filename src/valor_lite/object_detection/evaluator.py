import heapq
import json
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from numpy.typing import NDArray

from valor_lite.cache import Cache
from valor_lite.object_detection.computation import (
    compute_average_precision,
    compute_average_recall,
    compute_confusion_matrix,
    compute_counts,
    compute_label_metadata,
    compute_precision_recall_f1,
    prune_pairs,
    rank_pairs,
)
from valor_lite.object_detection.metric import Metric, MetricType
from valor_lite.object_detection.utilities import (
    unpack_confusion_matrix_into_metric_list,
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


class Evaluator:
    def __init__(self, directory: str | Path):
        self._dir = Path(directory)
        self._detailed_path = self._dir / Path("detailed")
        self._ranked_path = self._dir / Path("ranked")
        self._labels_path = self._dir / Path("labels.json")

        self._detailed = ds.dataset(self._detailed_path, format="parquet")
        self._ranked = ds.dataset(self._ranked_path, format="parquet")
        with open(self._labels_path, "r") as f:
            labels = json.load(f)
            self._index_to_label = {int(k): v for k, v in labels.items()}

        print(json.dumps(self.info, indent=2))

    @property
    def schema(self) -> pa.Schema:
        return self._detailed.schema

    @property
    def info(self) -> dict[str, int]:
        return {
            "number_of_detailed_rows": self._detailed.count_rows(),
            "number_of_ranked_rows": self._ranked.count_rows(),
            "number_of_labels": len(self._index_to_label),
        }

    def iterate_detailed_pairs(self, columns=None, filter_expr=None):
        for fragment in self._detailed.get_fragments():
            yield fragment.to_table(
                columns=columns,
                filter=filter_expr,
            )

    def iterate_ranked_pairs(self, columns=None, filter_expr=None):
        for fragment in self._ranked.get_fragments():
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

        columns = [
            "datum_id",
            "gt_id",
            "pd_id",
            "gt_label_id",
            "pd_label_id",
            "iou",
            "score",
        ]
        n_ious = len(iou_thresholds)
        n_scores = len(score_thresholds)
        n_labels = len(self._index_to_label)

        counts = np.zeros((n_ious, n_scores, 3, n_labels), dtype=np.uint64)
        pr_curve = np.zeros((n_ious, n_labels, 101, 2))
        for tbl in self.iterate_ranked_pairs(
            columns=columns,
            filter_expr=filter_expr,
        ):
            pairs = np.column_stack(
                [tbl.column(i).to_numpy() for i in range(tbl.num_columns)]
            )
            if pairs.size == 0:
                continue

            ranked_pairs = rank_pairs(pairs)
            (batch_counts, batch_pr_curve) = compute_counts(
                ranked_pairs=ranked_pairs,
                iou_thresholds=np.array(iou_thresholds),
                score_thresholds=np.array(score_thresholds),
                number_of_groundtruths_per_label=label_metadata[:, 0],
                number_of_labels=len(self._index_to_label),
            )
            counts += batch_counts
            pr_curve += batch_pr_curve

        precision_recall_f1 = compute_precision_recall_f1(
            counts=counts,
            number_of_groundtruths_per_label=label_metadata[:, 0],
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
            label_metadata=label_metadata,
            iou_thresholds=iou_thresholds,
            score_thresholds=score_thresholds,
            index_to_label=self._index_to_label,
        )

    def compute_confusion_matrix(
        self,
        tbl: pa.Table,
        detailed: np.ndarray,
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

        numeric_cols = [
            "datum_id",
            "gt_id",
            "pd_id",
            "gt_label_id",
            "pd_label_id",
            "iou",
            "score",
        ]
        metrics = []
        for tbl in self._iterate_fragments():
            detailed = np.column_stack(
                [tbl[col].to_numpy() for col in numeric_cols]
            )
            if detailed.size == 0:
                continue

            # extract UIDs
            index_to_datum_id = create_mapping(
                tbl, detailed, 0, "datum_id", "datum_uid"
            )
            index_to_groundtruth_id = create_mapping(
                tbl, detailed, 1, "gt_id", "gt_uid"
            )
            index_to_prediction_id = create_mapping(
                tbl, detailed, 2, "pd_id", "pd_uid"
            )

            results = compute_confusion_matrix(
                detailed_pairs=detailed,
                iou_thresholds=np.array(iou_thresholds),
                score_thresholds=np.array(score_thresholds),
            )

            metrics.append(
                unpack_confusion_matrix_into_metric_list(
                    results=results,
                    detailed_pairs=detailed,
                    iou_thresholds=iou_thresholds,
                    score_thresholds=score_thresholds,
                    index_to_datum_id=index_to_datum_id,
                    index_to_groundtruth_id=index_to_groundtruth_id,
                    index_to_prediction_id=index_to_prediction_id,
                    index_to_label=self._index_to_label,
                )
            )

        return metrics

    def evaluate(
        self,
        iou_thresholds: list[float] = [0.1, 0.5, 0.75],
        score_thresholds: list[float] = [0.5, 0.75, 0.9],
        filter_expr=None,
    ):
        numeric_cols = [
            "datum_id",
            "gt_id",
            "pd_id",
            "gt_label_id",
            "pd_label_id",
            "iou",
            "score",
        ]
        metrics = defaultdict(list)
        for tbl in self._iterate_fragments():
            detailed = np.column_stack(
                [tbl[col].to_numpy() for col in numeric_cols]
            )
            if detailed.size == 0:
                continue

            # base metrics
            ranked = prune_pairs(detailed)

            # extract UIDs
            index_to_datum_id = create_mapping(
                tbl, detailed, 0, "datum_id", "datum_uid"
            )
            index_to_groundtruth_id = create_mapping(
                tbl, detailed, 1, "gt_id", "gt_uid"
            )
            index_to_prediction_id = create_mapping(
                tbl, detailed, 2, "pd_id", "pd_uid"
            )

            cm = compute_confusion_matrix(
                detailed_pairs=detailed,
                iou_thresholds=np.array(iou_thresholds),
                score_thresholds=np.array(score_thresholds),
            )

            metrics[MetricType.ConfusionMatrix].extend(
                unpack_confusion_matrix_into_metric_list(
                    results=cm,
                    detailed_pairs=detailed,
                    iou_thresholds=iou_thresholds,
                    score_thresholds=score_thresholds,
                    index_to_datum_id=index_to_datum_id,
                    index_to_groundtruth_id=index_to_groundtruth_id,
                    index_to_prediction_id=index_to_prediction_id,
                    index_to_label=self._index_to_label,
                )
            )

        return metrics


if __name__ == "__main__":

    e = Evaluator("bench")

    e.evaluate([], [], None)
