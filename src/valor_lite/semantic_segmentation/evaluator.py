import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from numpy.typing import NDArray

from valor_lite.cache import FileCacheReader, MemoryCacheReader
from valor_lite.semantic_segmentation.computation import compute_metrics
from valor_lite.semantic_segmentation.metric import MetricType
from valor_lite.semantic_segmentation.shared import Base, EvaluatorInfo
from valor_lite.semantic_segmentation.utilities import (
    unpack_precision_recall_iou_into_metric_lists,
)


class Evaluator(Base):
    def __init__(
        self,
        reader: MemoryCacheReader | FileCacheReader,
        info: EvaluatorInfo,
        index_to_label: dict[int, str],
        confusion_matrix: NDArray[np.uint64],
        path: str | Path | None = None,
    ):
        self._path = Path(path) if path else None
        self._reader = reader
        self._info = info
        self._index_to_label = index_to_label
        self._confusion_matrix = confusion_matrix

    @property
    def info(self) -> EvaluatorInfo:
        return self._info

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
        reader = FileCacheReader.load(cls._generate_cache_path(path))

        # build evaluator meta
        (
            index_to_label,
            confusion_matrix,
            info,
        ) = cls._generate_meta(reader, index_to_label_override)

        # read config
        metadata_path = cls._generate_metadata_path(path)
        with open(metadata_path, "r") as f:
            metadata_types = json.load(f)
            (
                info.datum_metadata_fields,
                info.groundtruth_metadata_fields,
                info.prediction_metadata_fields,
            ) = cls._decode_metadata_fields(metadata_types)

        return cls(
            reader=reader,
            info=info,
            index_to_label=index_to_label,
            confusion_matrix=confusion_matrix,
            path=path,
        )

    def filter(
        self,
        datums: pc.Expression | None = None,
        groundtruths: pc.Expression | None = None,
        predictions: pc.Expression | None = None,
        path: str | Path | None = None,
    ) -> "Evaluator":
        """
        Filter evaluator cache.

        Parameters
        ----------
        datums : pc.Expression | None = None
            A filter expression used to filter datums.
        groundtruths : pc.Expression | None = None
            A filter expression used to filter ground truth annotations.
        predictions : pc.Expression | None = None
            A filter expression used to filter predictions.
        path : str | Path
            Where to store the filtered cache if storing on disk.

        Returns
        -------
        Evaluator
            A new evaluator object containing the filtered cache.
        """
        from valor_lite.semantic_segmentation.loader import Loader

        if isinstance(self._reader, FileCacheReader):
            if not path:
                raise ValueError(
                    "expected path to be defined for file-based loader"
                )
            loader = Loader.persistent(
                path=path,
                batch_size=self._reader.batch_size,
                rows_per_file=self._reader.rows_per_file,
                compression=self._reader.compression,
                datum_metadata_fields=self.info.datum_metadata_fields,
                groundtruth_metadata_fields=self.info.groundtruth_metadata_fields,
                prediction_metadata_fields=self.info.prediction_metadata_fields,
            )
        else:
            loader = Loader.in_memory(
                batch_size=self._reader.batch_size,
                datum_metadata_fields=self.info.datum_metadata_fields,
                groundtruth_metadata_fields=self.info.groundtruth_metadata_fields,
                prediction_metadata_fields=self.info.prediction_metadata_fields,
            )

        for tbl in self._reader.iterate_tables(filter=datums):
            columns = (
                "datum_id",
                "gt_label_id",
                "pd_label_id",
            )
            pairs = np.column_stack([tbl[col].to_numpy() for col in columns])

            n_pairs = pairs.shape[0]
            gt_ids = pairs[:, (0, 1)].astype(np.int64)
            pd_ids = pairs[:, (0, 2)].astype(np.int64)

            if groundtruths is not None:
                mask_valid_gt = np.zeros(n_pairs, dtype=np.bool_)
                gt_tbl = tbl.filter(groundtruths)
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

            if predictions is not None:
                mask_valid_pd = np.zeros(n_pairs, dtype=np.bool_)
                pd_tbl = tbl.filter(predictions)
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
            loader._writer.write_table(tbl)

        return loader.finalize(index_to_label_override=self._index_to_label)

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

    def delete(self):
        """Delete any cached files."""
        if self._path and self._path.exists():
            self.delete_at_path(self._path)
