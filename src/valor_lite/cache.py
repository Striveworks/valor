import glob
import json
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


class DataType(StrEnum):
    INTEGER = "int"
    FLOAT = "float"
    STRING = "string"
    TIMESTAMP = "timestamp"

    def to_type(self):
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
                return pa.timestamp()


class CacheReader:
    def __init__(self, where: str | Path):
        self._dir = Path(where)
        self._cfg = self._dir / ".cfg"

        with open(self._cfg, "r") as f:
            cfg = json.load(f)
            self._batch_size = cfg.get("batch_size")
            self._rows_per_file = cfg.get("rows_per_file")
            self._compression = cfg.get("compression")

    @property
    def files(self) -> list[str]:
        return glob.glob(f"{self._dir}/*.parquet")

    @property
    def num_files(self) -> int:
        return len(self.files)

    @property
    def dataset(self):
        return ds.dataset(
            self._dir,
            format="parquet",
        )

    @property
    def schema(self):
        return self.dataset.schema

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def rows_per_file(self) -> int:
        return self._rows_per_file

    @property
    def compression(self) -> str:
        return self._compression


class CacheWriter(CacheReader):
    def __init__(
        self,
        where: str | Path,
        schema: pa.Schema,
        batch_size: int = 1000,
        rows_per_file: int = 10000,
        compression: str = "snappy",
        delete_if_exists: bool = True,
    ):
        self._dir = Path(where)
        self._cfg = self._dir / ".cfg"

        self._schema = schema
        self._batch_size = batch_size
        self._rows_per_file = rows_per_file
        self._compression = compression

        if delete_if_exists:
            self.delete_files()
        self._dir.mkdir(parents=True, exist_ok=True)

        # Internal state
        self._writer = None
        self._buffer = []
        self._count = 0

        with open(self._cfg, "w") as f:
            info = dict(
                batch_size=batch_size,
                rows_per_file=rows_per_file,
                compression=compression,
            )
            json.dump(info, f, indent=2)

    @property
    def schema(self):
        return self._schema

    @property
    def dataset(self):
        return ds.dataset(
            self._dir,
            format="parquet",
            schema=self.schema,
        )

    def delete_files(self):
        for file in self.files:
            Path(file).unlink()

    @property
    def next_index(self):
        files = self.files
        if not files:
            return 0
        return max([int(Path(f).stem) for f in files]) + 1

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

    def _next_filename(self) -> Path:
        return self._dir / f"{self.next_index:06d}.parquet"

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
