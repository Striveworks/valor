import json
import heapq
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pyarrow.dataset as ds
from numpy.typing import NDArray

from valor_lite.cache import Cache
from valor_lite.object_detection.computation import (
    compute_label_metadata,
    compute_confusion_matrix,
    compute_counts,
    compute_average_precision,
    compute_average_recall,
    compute_precision_recall_f1,
    rank_pairs,
    prune_pairs,
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
        self._dataset_path = self._dir / Path("cache")
        self._labels_path = self._dir / Path("labels.json")

        self._dataset = ds.dataset(self._dataset_path, format="parquet")
        with open(self._labels_path, "r") as f:
            labels = json.load(f)
            self._index_to_label = {int(k): v for k, v in labels.items()}

    @property
    def schema(self) -> pa.Schema:
        return self._dataset.schema

    def _iterate_fragments(
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
        for fragment in self._dataset.get_fragments():
            yield fragment.to_table()

    def create_ranked_cache(
        self,
        where: str | Path,
        batch_size: int = 1000,
        rows_per_file: int = 10000,
        compression: str = "snappy",
        datum_filter_expr=None,
        annotation_filter_expr=None,
    ):
        path = self._dir / where
        schema = pa.schema([
            ("datum_id", pa.int64()),
            # groundtruth
            ("gt_id", pa.int64()),
            ("gt_label_id", pa.int64()),
            # prediction
            ("pd_id", pa.int64()),
            ("pd_label_id", pa.int64()),
            ("score", pa.float64()),
            # pair
            ("iou", pa.float64()),
        ])
        numeric_cols = [
            "datum_id",
            "gt_id",
            "pd_id",
            "gt_label_id",
            "pd_label_id",
            "iou",
            "score",
        ]

        # TODO (c.zaloom) - this is slow
        with Cache(
            path,
            schema=schema,
            batch_size=batch_size,
            rows_per_file=rows_per_file,
            compression=compression,
        ) as cache:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpcache = Cache(                
                    path,
                    schema=schema,
                    batch_size=batch_size,
                    rows_per_file=rows_per_file,
                    compression=compression,
                )
                for tbl in self._iterate_fragments(
                    datum_filter_expr=datum_filter_expr,
                    annotation_filter_expr=annotation_filter_expr,
                ):
                    detailed = np.column_stack(
                        [tbl[col].to_numpy() for col in numeric_cols]
                    )
                    if detailed.size == 0:
                        continue
                    ranked = prune_pairs(detailed)
                    tmpcache.write_batch({
                        "datum_id": pa.array(ranked[:, 0]),
                        "gt_id": pa.array(ranked[:, 1]),
                        "pd_id": pa.array(ranked[:, 2]),
                        "gt_label_id": pa.array(ranked[:, 3]),
                        "pd_label_id": pa.array(ranked[:, 4]),
                        "iou": pa.array(ranked[:, 5]),
                        "score": pa.array(ranked[:, 6]),
                    })
                tmpcache.flush()

                # file sort
                sort_args = [
                    ("score", "descending"),
                    ("iou", "descending"),
                ]
                for fragment in tmpcache.dataset.get_fragments():
                    tbl = fragment.to_table()
                    sorted_tbl = tbl.sort_by(sort_args)
                    pq.write_table(sorted_tbl, fragment.path)

                # merge
                import itertools
                counter = itertools.count()
                heap = []
                for idx, path in enumerate(tmpcache.files):
                    try:
                        pf = pq.ParquetFile(path)
                        batch_iter = pf.iter_batches(batch_size=batch_size)
                        batch = next(batch_iter)
                        table = pa.Table.from_batches([batch])
                        if len(table) > 0:
                            sort_key = (-table[col][0].as_py() for col in ["iou", "score"])
                            heapq.heappush(heap, (sort_key, next(counter), idx, table, 0, batch_iter, pf))
                    except StopIteration:
                        pass
                
                while heap:
                    sort_key, file_idx, table, row_idx, batch_iter, pf = heapq.heappop(heap)
                    row_table = table.slice(row_idx, 1)
                    cache.write_batch(row_table)
                    
                    row_idx += 1
                    if row_idx < len(table):
                        sort_key = tuple(-table[col][row_idx].as_py() for col in ["iou", "score"])
                        heapq.heappush(heap, (sort_key, next(counter), file_idx, table, row_idx, batch_iter, pf))
                    else:
                        # Need next batch
                        try:
                            batch = next(batch_iter)
                            table = pa.Table.from_batches([batch])
                            if len(table) > 0:
                                sort_key = tuple(-table[col][0].as_py() for col in ["iou", "score"])
                                heapq.heappush(heap, (sort_key, next(counter), file_idx, table, 0, batch_iter, pf))
                        except StopIteration:
                            pass

    # def compute_precision_recall(
    #     self,
    #     iou_thresholds: list[float],
    #     score_thresholds: list[float],
    # ) -> dict[MetricType, list[Metric]]:
    #     """
    #     Computes all metrics except for ConfusionMatrix

    #     Parameters
    #     ----------
    #     iou_thresholds : list[float]
    #         A list of IOU thresholds to compute metrics over.
    #     score_thresholds : list[float]
    #         A list of score thresholds to compute metrics over.
    #     filter_ : Filter, optional
    #         A collection of filter parameters and masks.

    #     Returns
    #     -------
    #     dict[MetricType, list]
    #         A dictionary mapping MetricType enumerations to lists of computed metrics.
    #     """
    #     if not iou_thresholds:
    #         raise ValueError("At least one IOU threshold must be passed.")
    #     elif not score_thresholds:
    #         raise ValueError("At least one score threshold must be passed.")

    #     results = compute_precion_recall(
    #         ranked_pairs=ranked,
    #         label_metadata=label_metadata,
    #         iou_thresholds=np.array(iou_thresholds),
    #         score_thresholds=np.array(score_thresholds),
    #     )
    #     return unpack_precision_recall_into_metric_lists(
    #         results=results,
    #         label_metadata=label_metadata,
    #         iou_thresholds=iou_thresholds,
    #         score_thresholds=score_thresholds,
    #         index_to_label=self._index_to_label,
    #     )

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
        for tbl in self._iterate_dataset():
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
