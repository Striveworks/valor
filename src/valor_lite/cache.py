import glob
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

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


class Cache:

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
        self.schema = schema
        self.batch_size = batch_size
        self.rows_per_file = rows_per_file
        self.compression = compression
        
        if delete_if_exists:
            self.delete_files()
        self._dir.mkdir(parents=True, exist_ok=True)

        # Internal state
        self._writer = None
        self._buffer = None
        self._count = 0

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
        batch = {}
        for k in rows[0].keys():
            batch[k] = [row[k] for row in rows]
        self.write_batch(batch)
    
    def write_batch(
        self,
        batch: dict[str, list],
    ):  
        # append any buffered data
        # if self._buffer:
        #     batch = {
        #         k: v + self._buffer[k]
        #         for k, v in batch.items()
        #     }
        #     self._buffer = None
        
        # check size
        size = max([len(v) for v in batch.values()], default=0)
        # if size < self.batch_size and self._count < self.rows_per_file:
        #     self._buffer = batch
        #     return
        
        # write batch
        writer = self._get_or_create_writer()
        writer.write_batch(
            pa.RecordBatch.from_pydict(batch, schema=self.schema)
        )
        
        # check file size
        self._count += size
        if self._count >= self.rows_per_file:
            self.flush()

    def flush(self):
        if self._buffer is not None:
            writer = self._get_or_create_writer()
            writer.write_batch(
                pa.RecordBatch.from_pydict(self._buffer, schema=self.schema)
            )
        self._buffer = None
        self._count = 0
        self._close_writer()        
    
    def _get_or_create_writer(self) -> pq.ParquetWriter:
        """Open a new parquet file for writing."""
        if self._writer is not None:
            return self._writer
        path = self._dir / f"{self.next_index:06d}.parquet"
        self._writer = pq.ParquetWriter(
            where=path,
            schema=self.schema, 
            compression=self.compression
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

    def sort_by(self, args):
        self.flush()
        dataset = self.dataset
        
        