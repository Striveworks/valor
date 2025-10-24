import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.dataset as ds
from numpy.typing import NDArray

from valor_lite.cache import DataType
from valor_lite.semantic_segmentation.computation import compute_metrics
from valor_lite.semantic_segmentation.metric import Metric, MetricType
from valor_lite.semantic_segmentation.utilities import (
    unpack_precision_recall_iou_into_metric_lists,
)


@dataclass
class EvaluatorInfo:
    number_of_rows: int = 0
    number_of_datums: int = 0
    number_of_labels: int = 0
    number_of_pixels: int = 0
    number_of_groundtruth_pixels: int = 0
    number_of_prediction_pixels: int = 0
    datum_metadata_types: dict[str, DataType] | None = None
    groundtruth_metadata_types: dict[str, DataType] | None = None
    prediction_metadata_types: dict[str, DataType] | None = None


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
            self._confusion_matrix,
            self._info,
        ) = self.generate_meta(self._dataset, labels_override)

        # read config
        with open(self._metadata_path, "r") as f:
            types = json.load(f)
            self._info.datum_metadata_types = types["datum"]
            self._info.groundtruth_metadata_types = types["groundtruth"]
            self._info.prediction_metadata_types = types["prediction"]
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
        loader = cls(
            directory=directory,
            name=name,
            batch_size=evaluator._detailed_batch_size,
            rows_per_file=evaluator._detailed_rows_per_file,
            compression=evaluator._detailed_compression,
            datum_metadata_types=evaluator.info.datum_metadata_types,
            groundtruth_metadata_types=evaluator.info.groundtruth_metadata_types,
            prediction_metadata_types=evaluator.info.prediction_metadata_types,
        )
        for fragment in evaluator.dataset.get_fragments():
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
                mask_valid_gt = np.zeros(n_pairs, dtype=np.bool_)
                gt_tbl = tbl.filter(filter_expr.groundtruths)
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
            loader._cache.write_table(tbl)

        loader._cache.flush()
        if loader._cache.dataset.count_rows() == 0:
            raise EmptyCacheError()

        evaluator = Evaluator(
            directory=loader._directory,
            name=loader._name,
            labels_override=evaluator._index_to_label,
        )
        return evaluator

    def compute_precision_recall_iou(self) -> dict[MetricType, list]:
        """
        Performs an evaluation and returns metrics.

        Returns
        -------
        dict[MetricType, list]
            A dictionary mapping MetricType enumerations to lists of computed metrics.
        """
        results = compute_metrics(counts=self._confusion_matrix)
        return unpack_precision_recall_iou_into_metric_lists(
            results=results,
            index_to_label=self._index_to_label,
        )

    def evaluate(self) -> dict[MetricType, list[Metric]]:
        """
        Computes all available metrics.

        Returns
        -------
        dict[MetricType, list[Metric]]
            Lists of metrics organized by metric type.
        """
        return self.compute_precision_recall_iou()
