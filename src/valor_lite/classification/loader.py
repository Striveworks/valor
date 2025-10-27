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
from valor_lite.classification.evaluator import Evaluator
from valor_lite.classification.format import PathFormatter
from valor_lite.exceptions import EmptyCacheError


class Loader(PathFormatter):
    def __init__(
        self,
        path: str | Path,
        writer: CacheWriter,
        datum_metadata_types: dict[str, DataType] | None = None,
    ):
        self._path = Path(path)
        self._cache = writer
        self._datum_metadata_types = datum_metadata_types

        # internal state
        self._labels: dict[str, int] = {}
        self._index_to_label: dict[int, str] = {}
        self._datum_count = 0

    @property
    def path(self) -> Path:
        return self._path

    @classmethod
    def create(
        cls,
        path: str | Path,
        batch_size: int = 1,
        rows_per_file: int = 1,
        compression: str = "snappy",
        datum_metadata_types: dict[str, DataType] | None = None,
        delete_if_exists: bool = False,
    ):
        path = Path(path)
        if delete_if_exists:
            cls.delete(path)

        # create cache schema
        datum_metadata_schema = convert_type_mapping_to_schema(
            datum_metadata_types
        )
        schema = pa.schema(
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

        # create cache
        cache_path = cls._generate_cache_path(path)
        cache = CacheWriter.create(
            path=cache_path,
            schema=schema,
            batch_size=batch_size,
            rows_per_file=rows_per_file,
            compression=compression,
        )

        # write metadatata config
        metadata_path = cls._generate_metadata_path(path)
        with open(metadata_path, "w") as f:
            types = {
                "datum": datum_metadata_types,
                "groundtruth": None,
                "prediction": None,
            }
            json.dump(types, f, indent=2)

        return cls(
            path=path,
            writer=cache,
            datum_metadata_types=datum_metadata_types,
        )

    @classmethod
    def delete(cls, path: str | Path):
        path = Path(path)
        if not path.exists():
            return
        CacheWriter.delete(cls._generate_cache_path(path))
        metadata_path = cls._generate_metadata_path(path)
        if metadata_path.exists() and metadata_path.is_file():
            metadata_path.unlink()
        path.rmdir()

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
            rows = []
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
                    ("datum_id", "ascending"),
                    ("gt_label_id", "ascending"),
                    ("pd_label_id", "ascending"),
                ]
            )
            pq.write_table(
                sorted_tbl, path, compression=self._cache.compression
            )
        return Evaluator.load(self.path)
