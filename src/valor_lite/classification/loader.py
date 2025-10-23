import heapq
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from valor_lite.cache import (
    CacheWriter,
    DataType,
    convert_type_mapping_to_schema,
)
from valor_lite.classification.annotation import Classification
from valor_lite.classification.evaluator import Evaluator, Filter
from valor_lite.exceptions import EmptyCacheError


class Loader:
    def __init__(
        self,
        name: str = "default",
        directory: str | Path = ".valor",
        batch_size: int = 1_000,
        rows_per_file: int = 10_000,
        compression: str = "snappy",
        datum_metadata_types: dict[str, DataType] | None = None,
    ):
        self._directory = Path(directory)
        self._name = name

        self._path = self._directory / self._name
        self._path.mkdir(parents=True, exist_ok=True)
        self._cache_path = self._path / "cache"
        self._metadata_path = self._path / "metadata.json"

        self._labels: dict[str, int] = {}
        self._index_to_label: dict[int, str] = {}
        self._datum_count = 0

        with open(self._metadata_path, "w") as f:
            types = {
                "datum": datum_metadata_types,
                "groundtruth": None,
                "prediction": None,
            }
            json.dump(types, f, indent=2)

        datum_metadata_schema = convert_type_mapping_to_schema(
            datum_metadata_types
        )

        self._schema = pa.schema(
            [
                ("datum_uid", pa.string()),
                ("datum_id", pa.int64()),
                *datum_metadata_schema,
                # groundtruth
                ("gt_label", pa.string()),
                ("gt_label_id", pa.int64()),
                # prediction
                ("pd_label", pa.string()),
                ("pd_label_id", pa.int64()),
                # pair
                ("score", pa.float64()),
                ("winner", pa.bool_()),
            ]
        )
        self._cache = CacheWriter(
            where=self._cache_path,
            schema=self._schema,
            batch_size=batch_size,
            rows_per_file=rows_per_file,
            compression=compression,
        )

    @property
    def schema(self) -> pa.Schema:
        return self._schema

    def _add_label(self, value: str) -> int:
        idx = self._labels.get(value, None)
        if idx is None:
            idx = len(self._labels)
            self._labels[value] = idx
            self._index_to_label[idx] = value
        return idx

    def add_data(
        self,
        classifications: list[Classification],
        show_progress: bool = False,
    ):
        """
        Adds classifications to the cache.

        Parameters
        ----------
        classifications : list[Classification]
            A list of Classification objects.
        show_progress : bool, default=False
            Toggle for tqdm progress bar.
        """

        disable_tqdm = not show_progress
        for classification in tqdm(classifications, disable=disable_tqdm):
            if len(classification.predictions) == 0:
                raise ValueError(
                    "Classifications must contain at least one prediction."
                )

            # prepare metadata
            datum_metadata = (
                classification.metadata if classification.metadata else {}
            )

            # write to cache
            rows = list()
            gidx = self._add_label(classification.groundtruth)
            max_score_idx = np.argmax(np.array(classification.scores))
            for idx, (plabel, score) in enumerate(
                zip(classification.predictions, classification.scores)
            ):
                pidx = self._add_label(plabel)
                rows.append(
                    {
                        # datum
                        "datum_uid": classification.uid,
                        "datum_id": self._datum_count,
                        **datum_metadata,
                        # groundtruth
                        "gt_label": classification.groundtruth,
                        "gt_label_id": gidx,
                        # prediction
                        "pd_label": plabel,
                        "pd_label_id": pidx,
                        # pair
                        "score": float(score),
                        "winner": max_score_idx == idx,
                    }
                )
            self._cache.write_rows(rows)

            # update datum count
            self._datum_count += 1

    def finalize(self):
        """
        Performs data finalization and some preprocessing steps.

        Returns
        -------
        Evaluator
            A ready-to-use evaluator object.
        """
        self._cache.flush()
        if self._cache.dataset.count_rows() == 0:
            raise EmptyCacheError()

        # sort files individually
        for path in self._cache.dataset_files:
            pf = pq.ParquetFile(path)
            tbl = pf.read()
            sorted_tbl = tbl.sort_by(
                [
                    ("score", "descending"),
                    ("pd_label_id", "ascending"),
                    ("gt_label_id", "ascending"),
                ]
            )
            pq.write_table(
                sorted_tbl, path, compression=self._cache.compression
            )

        return Evaluator(
            directory=self._directory,
            name=self._name,
        )

    @classmethod
    def filter(
        cls,
        directory: str | Path,
        name: str,
        evaluator: Evaluator,
        filter_expr: Filter,
    ) -> Evaluator:
        loader = cls(
            directory=directory,
            name=name,
            batch_size=evaluator._detailed_batch_size,
            rows_per_file=evaluator._detailed_rows_per_file,
            compression=evaluator._detailed_compression,
            datum_metadata_types=evaluator.info.datum_metadata_types,
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
