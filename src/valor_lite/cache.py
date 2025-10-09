import glob
import heapq
import tempfile
import time
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
        batch = pa.RecordBatch.from_pylist(rows, schema=self.schema)
        self.write_batch(batch)

    def write_batch(
        self,
        batch: pa.RecordBatch | dict[str, list | np.ndarray | pa.Array],
    ):
        if isinstance(batch, dict):
            batch = pa.RecordBatch.from_pydict(batch)

        if self._buffer:
            combined_arrays = [
                pa.concat_arrays(
                    [self._buffer.column(name), batch.column(name)]
                )
                for name in self.schema.names
            ]
            batch = pa.RecordBatch.from_arrays(
                combined_arrays, schema=self.schema
            )
            self._buffer = None

        # check size
        size = batch.num_rows
        if size < self.batch_size and self._count < self.rows_per_file:
            self._buffer = batch
            return

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
        if self._buffer is not None:
            writer = self._get_or_create_writer()
            writer.write_batch(
                pa.RecordBatch.from_pydict(self._buffer, schema=self.schema)
            )
        self._buffer = None
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

    def sort_by(
        self,
        destination: str | Path,
        sorting: str | list[tuple[str, str]],
        columns: list[str] | None = None,
    ):

        self.flush()

        if isinstance(sorting, str):
            sorting = [(sorting, "ascending")]

        schema = self.schema
        if columns:
            schema = pa.schema(
                [field for field in schema if field.name in columns]
            )

        # TODO (c.zaloom) - this is slow
        with Cache(
            destination,
            schema=schema,
            batch_size=self.batch_size,
            rows_per_file=self.rows_per_file,
            compression=self.compression,
        ) as cache:

            max_to_table = 0
            max_sort = 0
            max_merge = 0

            if self.num_files == 1:
                pf = pq.ParquetFile(self.files[0])

                start = time.perf_counter()
                tbl = pf.read()
                end = time.perf_counter()
                max_to_table = max([max_to_table, end - start])

                start = time.perf_counter()
                sorted_tbl = tbl.sort_by(sorting)
                end = time.perf_counter()
                max_sort = max([max_sort, end - start])

                start = time.perf_counter()
                cache.write_table(sorted_tbl)
                end = time.perf_counter()
                max_merge = max([max_merge, end - start])

                info = {
                    "number_of_files": 1,
                    "conversion_to_table": max_to_table,
                    "file_sort": max_sort,
                    "file_merge": max_merge,
                }
                import json

                print(json.dumps(info, indent=2))
                return info

            with tempfile.TemporaryDirectory() as tmpdir:

                # sort individual files
                tmpfiles = []
                for idx, fragment in enumerate(self.dataset.get_fragments()):
                    tbl = fragment.to_table(columns=columns)

                    start = time.perf_counter()
                    sorted_tbl = tbl.sort_by(sorting)
                    end = time.perf_counter()
                    max_to_table = max([max_to_table, end - start])

                    path = Path(tmpdir) / f"{idx:06d}.parquet"

                    start = time.perf_counter()
                    pq.write_table(sorted_tbl, path)
                    end = time.perf_counter()
                    max_sort = max([max_sort, end - start])

                    tmpfiles.append(path)

                start = time.perf_counter()

                # merge sorted rows
                # TODO (c.zaloom) - explore more performant ways of doing this...
                heap = []
                batch_iterators = []
                batches = []
                for idx, path in enumerate(tmpfiles):
                    pf = pq.ParquetFile(path)
                    batch_iter = pf.iter_batches(batch_size=self.batch_size)
                    batch_iterators.append(batch_iter)
                    batches.append(next(batch_iterators[idx], None))
                    if batches[idx] is not None and len(batches[idx]) > 0:
                        sort_args = [
                            batches[idx][col][0].as_py()
                            if order == "ascending"
                            else -batches[idx][col][0].as_py()
                            for col, order in sorting
                        ]
                        heapq.heappush(heap, (*sort_args, idx, 0))

                while heap:
                    _, _, idx, row_idx = heapq.heappop(heap)
                    row_table = batches[idx].slice(row_idx, 1)
                    cache.write_batch(row_table)

                    row_idx += 1
                    if row_idx < len(batches[idx]):
                        sort_args = [
                            batches[idx][col][0].as_py()
                            if order == "ascending"
                            else -batches[idx][col][0].as_py()
                            for col, order in sorting
                        ]
                        heapq.heappush(heap, (*sort_args, idx, row_idx))
                    else:
                        batches[idx] = next(batch_iterators[idx], None)
                        if batches[idx] is not None and len(batches[idx]) > 0:
                            sort_args = [
                                batches[idx][col][0].as_py()
                                if order == "ascending"
                                else -batches[idx][col][0].as_py()
                                for col, order in sorting
                            ]
                            heapq.heappush(heap, (*sort_args, idx, 0))

                end = time.perf_counter()
                max_merge = max([max_merge, end - start])

                info = {
                    "number_of_files": len(tmpfiles),
                    "conversion_to_table": max_to_table,
                    "file_sort": max_sort,
                    "file_merge": max_merge,
                }
                import json

                print(json.dumps(info, indent=2))
                return info
