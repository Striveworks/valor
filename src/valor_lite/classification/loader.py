import json
from pathlib import Path

import pyarrow as pa
import numpy as np
from pyarrow import DataType
from tqdm import tqdm

from valor_lite.cache.compute import sort
from valor_lite.cache.ephemeral import MemoryCacheWriter
from valor_lite.cache.persistent import FileCacheWriter
from valor_lite.classification.annotation import Classification
from valor_lite.classification.evaluator import Evaluator
from valor_lite.classification.shared import Base
from valor_lite.exceptions import EmptyCacheError


class Loader(Base):
    def __init__(
        self,
        writer: MemoryCacheWriter | FileCacheWriter,
        sorted_writer: MemoryCacheWriter | FileCacheWriter,
        datum_metadata_fields: list[tuple[str, DataType]] | None = None,
        path: str | Path | None = None,
    ):
        self._path = Path(path) if path else None
        self._writer = writer
        self._rocauc_writer = sorted_writer
        self._datum_metadata_fields = datum_metadata_fields

        # internal state
        self._labels: dict[str, int] = {}
        self._index_to_label: dict[int, str] = {}
        self._datum_count = 0

    @classmethod
    def in_memory(
        cls,
        batch_size: int = 10_000,
        datum_metadata_fields: list[tuple[str, DataType]] | None = None,
    ):
        """
        Create an in-memory evaluator cache.

        Parameters
        ----------
        batch_size : int, default=10_000
            The target number of rows to buffer before writing to the cache. Defaults to 10_000.
        datum_metadata_fields : list[tuple[str, DataType]], optional
            Optional datum metadata field definition.
        """
        writer = MemoryCacheWriter.create(
            schema=cls._generate_schema(datum_metadata_fields),
            batch_size=batch_size,
        )
        sorted_writer = MemoryCacheWriter.create(
            schema=cls._generate_rocauc_schema(),
            batch_size=batch_size,
        )
        return cls(
            writer=writer,
            sorted_writer=sorted_writer,
            datum_metadata_fields=datum_metadata_fields,
            path=None,
        )

    @classmethod
    def persistent(
        cls,
        path: str | Path,
        batch_size: int = 10_000,
        rows_per_file: int = 100_000,
        compression: str = "snappy",
        datum_metadata_fields: list[tuple[str, DataType]] | None = None,
        delete_if_exists: bool = False,
    ):
        """
        Create a persistent file-based evaluator cache.

        Parameters
        ----------
        path : str | Path
            Where to store file-based cache.
        batch_size : int, default=10_000
            Sets the batch size for writing to file.
        rows_per_file : int, default=100_000
            Sets the maximum number of rows per file. This may be exceeded as files are datum aligned.
        compression : str, default="snappy"
            Sets the pyarrow compression method.
        datum_metadata_fields : list[tuple[str, DataType]], optional
            Optionally sets metadata description for use in filtering.
        delete_if_exists : bool, default=False
            Option to delete if cache already exists.
        """
        path = Path(path)
        if delete_if_exists:
            cls.delete_at_path(path)

        # create cache
        writer = FileCacheWriter.create(
            path=cls._generate_cache_path(path),
            schema=cls._generate_schema(datum_metadata_fields),
            batch_size=batch_size,
            rows_per_file=rows_per_file,
            compression=compression,
        )
        sorted_writer = FileCacheWriter.create(
            path=cls._generate_rocauc_cache_path(path),
            schema=cls._generate_rocauc_schema(),
            batch_size=batch_size,
            rows_per_file=rows_per_file,
            compression=compression,
        )

        # write metadatata config
        metadata_path = cls._generate_metadata_path(path)
        with open(metadata_path, "w") as f:
            encoded_types = cls._encode_metadata_fields(datum_metadata_fields)
            json.dump(encoded_types, f, indent=2)

        return cls(
            path=path,
            writer=writer,
            sorted_writer=sorted_writer,
            datum_metadata_fields=datum_metadata_fields,
        )

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
                        "match": (gidx == pidx) and pidx >= 0,
                    }
                )
            self._writer.write_rows(rows)

            # update datum count
            self._datum_count += 1

    def finalize(
        self,
        batch_size: int = 1_000,
        index_to_label_override: dict[int, str] | None = None,
    ):
        """
        Performs data finalization and some preprocessing steps.

        Parameters
        ----------
        batch_size : int, default=1_000
            Sets the maximum number of elements read into memory per-file when performing merge sort.
        index_to_label_override : dict[int, str], optional
            Pre-configures label mapping. Used when operating over filtered subsets.

        Returns
        -------
        Evaluator
            A ready-to-use evaluator object.
        """
        self._writer.flush()
        if self._writer.count_rows() == 0:
            raise EmptyCacheError()
        elif self._rocauc_writer.count_rows() > 0:
            raise RuntimeError("data already finalized")

        # sort in-place and locally
        self._writer.sort_by(
            [
                ("score", "descending"),
                ("datum_id", "ascending"),
                ("gt_label_id", "ascending"),
                ("pd_label_id", "ascending"),
            ]
        )

        # post-process into sorted writer
        reader = self._writer.to_reader()

        n_labels = len(self._index_to_label)

        def accumulate(batch: pa.RecordBatch, prev: np.ndarray | None) -> pa.RecordBatch:
            pd_label_id = batch["pd_label_id"].as_py()
            matched = batch["match"][0].as_py()
            if prev is None:
                prev = np.zeros(n_labels, dtype=np.uint64)
                return None, ()

        sort(
            source=reader,
            sink=self._rocauc_writer,
            batch_size=batch_size,
            sorting=[
                ("score", "descending"),
                # ("match", "descending"),
                # ("pd_label_id", "ascending"),
            ],
            columns=[
                "pd_label_id",
                "score",
                "match",
            ],
        )
        rocauc_reader = self._rocauc_writer.to_reader()

        # generate evaluator meta
        (index_to_label, label_counts, info,) = self.generate_meta(
            reader=reader, index_to_label_override=index_to_label_override
        )

        return Evaluator(
            reader=reader,
            rocauc_reader=rocauc_reader,
            info=info,
            label_counts=label_counts,
            index_to_label=index_to_label,
            path=self._path,
        )

    def delete(self):
        """
        Delete the classification evaluator cache.
        """
        if self._path and self._path.exists():
            self.delete_at_path(self._path)
