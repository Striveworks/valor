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
        """Get python type."""
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
        """Get arrow type."""
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
    def path(self) -> Path:
        """Cache directory path."""
        return self._path

    @staticmethod
    def _generate_config_path(path: str | Path) -> Path:
        """
        Generate cache configuration path.

        Parameters
        ----------
        path : str | Path
            Where the cache is stored.

        Returns
        -------
        Path
            Configuration filepath.
        """
        return Path(path) / ".cfg"

    def get_files(self) -> list[Path]:
        """
        Retrieve all files.

        Returns
        -------
        list[Path]
            A list of paths to files in the cache.
        """
        if not self.path.exists():
            return []
        files = []
        for entry in os.listdir(self._path):
            full_path = os.path.join(self._path, entry)
            if os.path.isfile(full_path):
                files.append(Path(full_path))
        return files

    def get_dataset_files(self) -> list[Path]:
        """
        Retrieve all dataset files.

        Returns
        -------
        list[Path]
            A list of paths to dataset files in the cache.
        """
        if not self.path.exists():
            return []
        return [
            Path(filepath) for filepath in glob.glob(f"{self._path}/*.parquet")
        ]


class CacheReader(CacheFiles):
    def __init__(
        self,
        path: str | Path,
        batch_size: int,
        rows_per_file: int,
        compression: str,
    ):
        self._path = Path(path)
        self._batch_size = batch_size
        self._rows_per_file = rows_per_file
        self._compression = compression

    @classmethod
    def load(cls, path: str | Path):
        """
        Load cache from disk.

        Parameters
        ----------
        path : str | Path
            Where the cache is stored.
        """

        # validate path
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Directory does not exist: {path}")
        elif not path.is_dir():
            raise NotADirectoryError(
                f"Path exists but is not a directory: {path}"
            )

        def _retrieve(config: dict, key: str):
            if value := config.get(key, None):
                return value
            raise KeyError(
                f"'{key}' is not defined within {cls._generate_config_path(path)}"
            )

        cfg_path = cls._generate_config_path(path)
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
            batch_size = _retrieve(cfg, "batch_size")
            rows_per_file = _retrieve(cfg, "rows_per_file")
            compression = _retrieve(cfg, "compression")

        return cls(
            path=path,
            batch_size=batch_size,
            rows_per_file=rows_per_file,
            compression=compression,
        )

    @property
    def dataset(self) -> ds.Dataset:
        """Cache dataset."""
        return ds.dataset(self._path, format="parquet")

    @property
    def schema(self) -> pa.Schema:
        """Cache schema."""
        return self.dataset.schema

    @property
    def batch_size(self) -> int:
        """Batch size for writes."""
        return self._batch_size

    @property
    def rows_per_file(self) -> int:
        """Target number of rows per file."""
        return self._rows_per_file

    @property
    def compression(self) -> str:
        """File compression method."""
        return self._compression


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

        # internal state
        self._writer = None
        self._buffer = []
        self._count = 0

    @classmethod
    def create(
        cls,
        path: str | Path,
        schema: pa.Schema,
        batch_size: int = 1_000,
        rows_per_file: int = 10_000,
        compression: str = "snappy",
    ):
        """
        Create a cache.

        Parameters
        ----------
        path : str | Path
            Where to write the cache.
        schema : pa.Schema
            Cache schema.
        batch_size : int, default=1_000
            Target batch size when writing chunks.
        rows_per_file : int, default=10_000
            Target number of rows to store per file.
        compression : str, default="snappy"
            Compression method to use when storing on disk.
        """
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
        """
        Load cache from disk.

        Parameters
        ----------
        path : str | Path
            Where the cache is stored.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Directory does not exist: {path}")
        elif not path.is_dir():
            raise NotADirectoryError(
                f"Path exists but is not a directory: {path}"
            )

        cfg_path = cls._generate_config_path(path)
        dataset = ds.dataset(path, format="parquet")
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        return cls(
            path=path,
            schema=dataset.schema,
            **cfg,
        )

    @classmethod
    def delete(cls, path: str | Path):
        """
        Delete a cache from disk.

        Parameters
        ----------
        path : str | Path
            Where the cache is stored.
        """
        path = Path(path)
        if not path.exists():
            return
        cache = cls.load(path)
        # delete config file
        cfg_path = cls._generate_config_path(path)
        if cfg_path.exists() and cfg_path.is_file():
            cfg_path.unlink()
        # delete parquet files
        for file in cache.get_dataset_files():
            if file.exists() and file.is_file() and file.suffix == ".parquet":
                file.unlink()
        # delete empty cache directory
        path.rmdir()

    @property
    def dataset(self) -> ds.Dataset:
        """Cache dataset."""
        return ds.dataset(
            self._path,
            format="parquet",
            schema=self.schema,
        )

    @property
    def schema(self) -> pa.Schema:
        """Cache schema."""
        return self._schema

    @property
    def batch_size(self) -> int:
        """Batch size for writes."""
        return self._batch_size

    @property
    def rows_per_file(self) -> int:
        """Target number of rows per file."""
        return self._rows_per_file

    @property
    def compression(self) -> str:
        """File compression method."""
        return self._compression

    def write_rows(
        self,
        rows: list[dict[str, Any]],
    ):
        """
        Write rows to cache.

        Parameters
        ----------
        rows : list[dict[str, Any]]
            A list of rows represented by dictionaries mapping fields to values.
        """
        if not rows:
            return
        batch = pa.RecordBatch.from_pylist(rows, schema=self.schema)
        self.write_batch(batch)

    def write_batch(
        self,
        batch: pa.RecordBatch | dict[str, list | np.ndarray | pa.Array],
    ):
        """
        Write a batch to cache.

        Parameters
        ----------
        batch : pa.RecordBatch | dict[str, list | np.ndarray | pa.Array]
            A batch of columnar data.
        """
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
        """
        Write a table directly to cache.

        Parameters
        ----------
        table : pa.Table
            A populated table.
        """
        self.flush()
        pq.write_table(table, where=self._generate_next_filename())

    def flush(self):
        """Flush the cache buffer."""
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

    def _generate_next_filename(self) -> Path:
        """Generates next dataset filepath."""
        files = self.get_dataset_files()
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
            where=self._generate_next_filename(),
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
