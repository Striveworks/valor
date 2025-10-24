import glob
import json
import os
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.lib as pl
import pyarrow.parquet as pq


class DataType(StrEnum):
    INTEGER = "int"
    FLOAT = "float"
    STRING = "string"
    TIMESTAMP = "timestamp"

    def to_py(self):
        match self:
            case DataType.INTEGER:
                return int
            case DataType.FLOAT:
                return float
            case DataType.STRING:
                return str
            case DataType.TIMESTAMP:
                return datetime

    def to_arrow(self):
        match self:
            case DataType.INTEGER:
                return pa.int64()
            case DataType.FLOAT:
                return pa.float64()
            case DataType.STRING:
                return pa.string()
            case DataType.TIMESTAMP:
                return pa.timestamp("us")


def convert_type_mapping_to_schema(
    type_mapping: dict[str, DataType] | None
) -> list[tuple[str, pl.DataType]]:
    """
    Convert type mapping to a pyarrow schema input.

    Parameters
    ----------
    type_mapping : dict[str, DataType] | None
        A map from string key to datatype. Treats input of `None` as empty mapping.

    Returns
    -------
    list[tuple[str, pyarrow.lib.DataType]]
        A list of field name, field type pairs that can be used as input to pyarrow.schema.
    """
    if not type_mapping:
        return []
    return [(k, DataType(v).to_arrow()) for k, v in type_mapping.items()]


class CacheFiles:
    def __init__(self, path: str | Path):
        self._path = Path(path)

    @property
    def files(self) -> list[str]:
        files = []
        for entry in os.listdir(self._path):
            full_path = os.path.join(self._path, entry)
            if os.path.isfile(full_path):
                files.append(full_path)
        return files

    @property
    def num_files(self) -> int:
        return len(self.files)

    @property
    def dataset_files(self) -> list[str]:
        return glob.glob(f"{self._path}/*.parquet")

    @property
    def num_dataset_files(self) -> int:
        return len(self.dataset_files)

    @staticmethod
    def _generate_config_path(path: str | Path) -> Path:
        return Path(path) / ".cfg"

    @staticmethod
    def _get_dataset_from_path(path: str | Path) -> ds.Dataset:
        return ds.dataset(path, format="parquet")


class CacheReader(CacheFiles):
    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._cfg = None
        self._dataset = None

        # validate path
        if not self._path.exists():
            raise FileNotFoundError(f"Directory does not exist: {self._path}")
        elif not self._path.is_dir():
            raise NotADirectoryError(
                f"Path exists but is not a directory: {self._path}"
            )

    @property
    def dataset(self) -> ds.Dataset:
        if not self._dataset:
            self._dataset = ds.dataset(
                self._path,
                format="parquet",
            )
        return self._dataset

    @property
    def schema(self) -> pa.Schema:
        return self.dataset.schema

    @property
    def config(self) -> dict:
        if self._cfg is None:
            cfg_path = self._generate_config_path(self._path)
            with open(cfg_path, "r") as f:
                self._cfg = json.load(f)
        return self._cfg

    def _read_config(self, key: str):
        if value := self.config.get(key, None):
            return value
        raise KeyError(
            f"'{key}' is not defined within {self._generate_config_path(self._path)}"
        )

    @property
    def batch_size(self) -> int:
        return int(self._read_config("batch_size"))

    @property
    def rows_per_file(self) -> int:
        return int(self._read_config("rows_per_file"))

    @property
    def compression(self) -> str:
        return str(self._read_config("compression"))


class CacheWriter(CacheFiles):
    def __init__(
        self,
        path: str | Path,
        schema: pa.Schema,
        batch_size: int,
        rows_per_file: int,
        compression: str,
    ):
        self._path = Path(path)
        self._schema = schema
        self._batch_size = batch_size
        self._rows_per_file = rows_per_file
        self._compression = compression

        # validate path
        if not self._path.exists():
            raise FileNotFoundError(f"Directory does not exist: {self._path}")
        elif not self._path.is_dir():
            raise NotADirectoryError(
                f"Path exists but is not a directory: {self._path}"
            )

        # internal state
        self._writer = None
        self._buffer = []
        self._count = 0

    @classmethod
    def create(
        cls,
        path: str | Path,
        schema: pa.Schema,
        batch_size: int = 1000,
        rows_per_file: int = 10000,
        compression: str = "snappy",
    ):
        Path(path).mkdir(parents=True, exist_ok=False)
        cfg_path = cls._generate_config_path(path)
        with open(cfg_path, "w") as f:
            cfg = dict(
                batch_size=batch_size,
                rows_per_file=rows_per_file,
                compression=compression,
            )
            json.dump(cfg, f, indent=2)
        return cls(
            path=path,
            schema=schema,
            batch_size=batch_size,
            rows_per_file=rows_per_file,
            compression=compression,
        )

    @classmethod
    def load(cls, path: str | Path):
        cfg_path = cls._generate_config_path(path)
        dataset = cls._get_dataset_from_path(path)
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        return cls(
            path=path,
            schema=dataset.schema,
            **cfg,
        )

    def write_rows(
        self,
        rows: list[dict[str, Any]],
    ):
        if not rows:
            return
        batch = pa.RecordBatch.from_pylist(rows, schema=self.schema)
        self.write_batch(batch)

    def write_batch(
        self,
        batch: pa.RecordBatch | dict[str, list | np.ndarray | pa.Array],
    ):
        if isinstance(batch, dict):
            batch = pa.RecordBatch.from_pydict(batch)

        size = batch.num_rows  # type: ignore - pyarrow typing
        if self._buffer:
            size += sum([b.num_rows for b in self._buffer])

        # check size
        if size < self.batch_size and self._count < self.rows_per_file:
            self._buffer.append(batch)
            return

        if self._buffer:
            self._buffer.append(batch)
            combined_arrays = [
                pa.concat_arrays([b.column(name) for b in self._buffer])
                for name in self.schema.names
            ]
            batch = pa.RecordBatch.from_arrays(
                combined_arrays, schema=self.schema
            )
            self._buffer = []

        # write batch
        writer = self._get_or_create_writer()
        writer.write_batch(batch)

        # check file size
        self._count += size
        if self._count >= self.rows_per_file:
            self.flush()

    def write_table(
        self,
        table: pa.Table,
    ):
        self.flush()
        pq.write_table(table, where=self._next_filename())

    def flush(self):
        if self._buffer:
            combined_arrays = [
                pa.concat_arrays([b.column(name) for b in self._buffer])
                for name in self.schema.names
            ]
            batch = pa.RecordBatch.from_arrays(
                combined_arrays, schema=self.schema
            )
            self._buffer = []
            writer = self._get_or_create_writer()
            writer.write_batch(batch)
        self._buffer = []
        self._count = 0
        self._close_writer()

    def delete(self):
        for file in self.files:
            Path(file).unlink()

    def _next_filename(self) -> Path:
        files = self.dataset_files
        if not files:
            next_index = 0
        else:
            next_index = max([int(Path(f).stem) for f in files]) + 1
        return self._path / f"{next_index:06d}.parquet"

    def _get_or_create_writer(self) -> pq.ParquetWriter:
        """Open a new parquet file for writing."""
        if self._writer is not None:
            return self._writer
        self._writer = pq.ParquetWriter(
            where=self._next_filename(),
            schema=self.schema,
            compression=self.compression,
        )
        return self._writer

    def _close_writer(self) -> None:
        """Close the current parquet file."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures data is flushed."""
        self.flush()

    @property
    def schema(self) -> pa.Schema:
        return self._schema

    @property
    def dataset(self) -> ds.Dataset:
        return ds.dataset(
            self._path,
            format="parquet",
            schema=self.schema,
        )

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def rows_per_file(self) -> int:
        return self._rows_per_file

    @property
    def compression(self) -> str:
        return self._compression
