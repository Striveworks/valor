from dataclasses import asdict, dataclass

import json
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

from pyarrow import pa
from pyarrow.compute import pc
from pyarrow.dataset import ds

from valor_lite.exceptions import EmptyCacheError, EmptyFilterError
from valor_lite.semantic_segmentation.annotation import Segmentation
from valor_lite.semantic_segmentation.computation import (
    compute_intermediates,
    compute_label_metadata,
    compute_metrics,
    filter_cache,
)
from valor_lite.semantic_segmentation.metric import Metric, MetricType
from valor_lite.semantic_segmentation.utilities import (
    unpack_precision_recall_iou_into_metric_lists,
)
from valor_lite.cache import CacheReader, DataType, convert_type_mapping_to_schema


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
            self._number_of_groundtruths_per_label,
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
        from valor_lite.semantic_segmentation.loader import Loader

        return Loader.filter(
            directory=directory,
            name=name,
            evaluator=self,
            filter_expr=filter_expr,
        )