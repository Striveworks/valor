import numpy as np
import pyarrow as pa
import pyarrow as pc
import pyarrow.dataset as ds

from numpy.typing import NDArray
from collections import defaultdict
from pathlib import Path

from valor_lite.object_detection.metric import Metric, MetricType
from valor_lite.object_detection.utilities import (
    unpack_confusion_matrix_into_metric_list,
    unpack_precision_recall_into_metric_lists,
)
from valor_lite.object_detection.computation import compute_confusion_matrix


def prune_pairs(
    pairs: NDArray[np.float64]
):
    # remove null predictions
    pairs = pairs[pairs[:, 2] >= 0.0]

    # find best fits for prediction
    mask_label_match = np.isclose(pairs[:, 3], pairs[:, 4])
    matched_predictions = np.unique(pairs[mask_label_match, 2])
    mask_unmatched_predictions = ~np.isin(pairs[:, 2], matched_predictions)
    pairs = pairs[mask_label_match | mask_unmatched_predictions]

    # only keep the highest ranked pair
    _, indices = np.unique(pairs[:, [0, 2, 4]], axis=0, return_index=True)
    pairs = pairs[indices]



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
        tbl[id_col][idx].as_py(): tbl[uid_col][idx].as_py()
        for idx in indices
    }


class Evaluator:

    def __init__(self, cache_dir: str | Path):
        self._cache_dir = cache_dir
        self._dataset = ds.dataset(
            str(self._cache_dir),
            format="parquet",
        )
    
    @property
    def schema(self) -> pa.Schema:
        return self._dataset.schema

    def _walk(
        self,
        columns=None,
        datum_filter_expr=None,
        annotation_filter_expr=None,
    ):
        """
        Generator that yields Tables from a PyArrow dataset.
        
        Yields
        ------
        pyarrow.Table
            Batches of data as Tables
        """
        scanner = self._dataset.scanner(
            columns=columns,
            filter=datum_filter_expr,
        )
        for batch in scanner.to_batches():
            yield pa.Table.from_batches([batch])

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

        if detailed.size == 0:
            return []
        
        # extract UIDs
        index_to_datum_id = create_mapping(tbl, detailed, 0, "datum_id", "datum_uid")
        index_to_groundtruth_id = create_mapping(tbl, detailed, 1, "gt_id", "gt_uid")
        index_to_prediction_id = create_mapping(tbl, detailed, 2, "pd_id", "pd_uid")
        index_to_gt_label = create_mapping(tbl, detailed, 3, "gt_label_id", "gt_label")
        index_to_pd_label = create_mapping(tbl, detailed, 4, "pd_label_id", "pd_label")
        index_to_label = index_to_gt_label | index_to_pd_label

        results = compute_confusion_matrix(
            detailed_pairs=detailed,
            iou_thresholds=np.array(iou_thresholds),
            score_thresholds=np.array(score_thresholds),
        )
        return unpack_confusion_matrix_into_metric_list(
            results=results,
            detailed_pairs=detailed,
            iou_thresholds=iou_thresholds,
            score_thresholds=score_thresholds,
            index_to_datum_id=index_to_datum_id,
            index_to_groundtruth_id=index_to_groundtruth_id,
            index_to_prediction_id=index_to_prediction_id,
            index_to_label=index_to_label,
        )

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
        for i, tbl in enumerate(self._walk()):
            detailed = np.column_stack([tbl[col].to_numpy() for col in numeric_cols])
            ranked = prune_pairs(detailed)

            metrics[MetricType.ConfusionMatrix].extend(
                self.compute_confusion_matrix(
                    tbl=tbl,
                    detailed=detailed,
                    iou_thresholds=iou_thresholds,
                    score_thresholds=score_thresholds,
                )
            )
        
        return metrics


if __name__ == "__main__":

    e = Evaluator("bench")

    e.evaluate([], [], None)

