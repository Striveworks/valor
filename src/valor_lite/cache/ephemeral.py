from typing import Any

import numpy as np
import pyarrow as pa


class EphemeralCache:
    def __init__(self, table: pa.Table):
        self._table = table

    def count_rows(self) -> int:
        """Count the number of rows in the cache."""
        return self._table.num_rows


class EphemeralCacheWriter(EphemeralCache):
    def __init__(
        self,
        table: pa.Table,
        batch_size: int,
    ):
        self._table = table
        self._schema = table.schema
        self._batch_size = batch_size

        # internal state
        self._buffer = []

    @classmethod
    def create(
        cls,
        schema: pa.Schema,
        batch_size: int,
    ):
        """
        Create a cache.

        Parameters
        ----------
        schema : pa.Schema
            Cache schema.
        batch_size : int, default=1_000
            Target batch size when writing chunks.
        """
        return cls(
            table=schema.empty_table(),
            batch_size=batch_size,
        )

    def delete(self):
        """
        Delete any existing cache data.
        """
        self._buffer = []
        self._table = self._table.schema.empty_table()

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
        batch = pa.RecordBatch.from_pylist(rows, schema=self._schema)
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
        if size < self._batch_size:
            self._buffer.append(batch)
            return

        if self._buffer:
            self._buffer.append(batch)
            combined_arrays = [
                pa.concat_arrays([b.column(name) for b in self._buffer])
                for name in self._schema.names
            ]
            batch = pa.RecordBatch.from_arrays(
                combined_arrays, schema=self._schema
            )
            self._buffer = []

        # write batch
        self.write_table(pa.Table.from_batches([batch]))

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
        self._table = pa.concat_tables([self._table, table])

    def flush(self):
        """Flush the cache buffer."""
        if self._buffer:
            combined_arrays = [
                pa.concat_arrays([b.column(name) for b in self._buffer])
                for name in self._schema.names
            ]
            batch = pa.RecordBatch.from_arrays(
                combined_arrays, schema=self._schema
            )
            self._table = pa.concat_tables(
                [self._table, pa.Table.from_batches([batch])]
            )
            self._buffer = []

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures data is flushed."""
        self.flush()


class EphemeralCacheReader:
    def __init__(
        self,
        cache: EphemeralCacheWriter,
    ):
        self._cache = cache
        self._schema = self._cache._schema

    @classmethod
    def load(cls, cache: EphemeralCacheWriter):
        """
        Load cache from table.

        Parameters
        ----------
        cache : EphemeralCacheWriter
            A cache writer containing the ephemeral cache.
        """
        return cls(cache=cache)

    def iterate_tables(self):
        """Iterate over tables within the cache."""
        yield self._cache._table

    def count_rows(self) -> int:
        """Count the number of rows in the cache."""
        return self._cache.count_rows()
