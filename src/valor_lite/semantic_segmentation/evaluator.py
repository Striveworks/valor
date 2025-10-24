import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
from numpy.typing import NDArray

from valor_lite.cache import CacheReader, DataType
from valor_lite.exceptions import EmptyCacheError
from valor_lite.semantic_segmentation.computation import compute_metrics
from valor_lite.semantic_segmentation.format import PathFormatter
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


class Evaluator(PathFormatter):
    def __init__(
        self,
        path: str | Path,
        cache: CacheReader,
        info: EvaluatorInfo,
        index_to_label: dict[int, str],
        confusion_matrix: NDArray[np.uint64],
    ):
        self._path = Path(path)
        self._cache = cache
        self._info = info
        self._index_to_label = index_to_label
        self._confusion_matrix = confusion_matrix

    @classmethod
    def load(
        cls,
        path: str | Path,
        index_to_label_override: dict[int, str] | None = None,
    ):
        # validate path
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Directory does not exist: {path}")
        elif not path.is_dir():
            raise NotADirectoryError(
                f"Path exists but is not a directory: {path}"
            )

        # load cache
        cache = CacheReader.load(cls._generate_cache_path(path))

        # build evaluator meta
        (
            index_to_label,
            confusion_matrix,
            info,
        ) = cls.generate_meta(cache.dataset, index_to_label_override)

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
            cache=cache,
            info=info,
            index_to_label=index_to_label,
            confusion_matrix=confusion_matrix,
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
        from valor_lite.semantic_segmentation.loader import Loader

        loader = Loader.create(
            path=path,
            batch_size=self.cache.batch_size,
            rows_per_file=self.cache.rows_per_file,
            compression=self.cache.compression,
            datum_metadata_types=self.info.datum_metadata_types,
            groundtruth_metadata_types=self.info.groundtruth_metadata_types,
            prediction_metadata_types=self.info.prediction_metadata_types,
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

        return loader.finalize()

    def delete(self):
        """
        Delete evaluator cache.
        """
        from valor_lite.semantic_segmentation.loader import Loader

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
