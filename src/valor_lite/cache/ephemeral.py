from typing import Any, Generator

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc


class MemoryCache:
    def __init__(
        self,
        table: pa.Table,
        batch_size: int,
    ):
        self._table = table
        self._batch_size = batch_size

    @property
    def schema(self) -> pa.Schema:
        return self._table.schema

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def count_tables(self) -> int:
        """Count the number of tables in the cache."""
        return 1

    def count_rows(self) -> int:
        """Count the number of rows in the cache."""
        return self._table.num_rows


class MemoryCacheReader(MemoryCache):
    def iterate_tables(
        self,
        columns: list[str] | None = None,
        filter: pc.Expression | None = None,
    ) -> Generator[pa.Table, None, None]:
        """
        Iterate over tables within the cache.

        Parameters
        ----------
        columns : list[str], optional
            Optionally select columns to be returned.
        filter : pyarrow.compute.Expression, optional
            Optionally filter table before returning.

        Yields
        ------
        pa.Table
        """
        table = self._table
        if filter is not None:
            table = table.filter(filter)
        if columns is not None:
            table = table.select(columns)
        yield table

    def iterate_pairs(
        self,
        columns: list[str] | None = None,
    ) -> Generator[np.ndarray, None, None]:
        """
        Iterate over chunks within the cache returning arrays.

        Parameters
        ----------
        columns : list[str], optional
            Optionally select columns to be returned.

        Yields
        ------
        np.ndarray
        """
        for tbl in self.iterate_tables(columns=columns):
            yield np.column_stack(
                [tbl.column(i).to_numpy() for i in range(tbl.num_columns)]
            )

    def iterate_pairs_with_table(
        self,
        columns: list[str] | None = None,
    ) -> Generator[tuple[pa.Table, np.ndarray], None, None]:
        """
        Iterate over chunks within the cache returning both tables and arrays.

        Parameters
        ----------
        columns : list[str], optional
            Optionally select columns to be returned.

        Yields
        ------
        tuple[pa.Table, np.ndarray]
        """
        for tbl in self.iterate_tables():
            columns = columns if columns else tbl.columns
            yield tbl, np.column_stack(
                [tbl[col].to_numpy() for col in columns]
            )


class MemoryCacheWriter(MemoryCache):
    def __init__(
        self,
        table: pa.Table,
        batch_size: int,
    ):
        super().__init__(
            table=table,
            batch_size=batch_size,
        )

        # internal state
        self._buffer = []

    @classmethod
    def create(
        cls,
        schema: pa.Schema,
        batch_size: int,
    ):
        """
        Create an in-memory cache.

        Parameters
        ----------
        schema : pa.Schema
            Cache schema.
        batch_size : int
            Target batch size when writing chunks.
        """
        return cls(
            table=schema.empty_table(),
            batch_size=batch_size,
        )

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

    def write_columns(
        self,
        columns: dict[str, list | np.ndarray | pa.Array],
    ):
        """
        Write columnar data to cache.

        Parameters
        ----------
        columns : dict[str, list | np.ndarray | pa.Array]
            A mapping of columnar field names to list of values.
        """
        if not columns:
            return
        batch = pa.RecordBatch.from_pydict(columns)
        self.write_batch(batch)

    def write_batch(
        self,
        batch: pa.RecordBatch,
    ):
        """
        Write a batch to cache.

        Parameters
        ----------
        batch : pa.RecordBatch
            A batch of columnar data.
        """
        size = batch.num_rows
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
                for name in self.schema.names
            ]
            batch = pa.RecordBatch.from_arrays(
                combined_arrays, schema=self.schema
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
                for name in self.schema.names
            ]
            batch = pa.RecordBatch.from_arrays(
                combined_arrays, schema=self.schema
            )
            self._table = pa.concat_tables(
                [self._table, pa.Table.from_batches([batch])]
            )
            self._buffer = []

    def sort_by(
        self,
        sorting: list[tuple[str, str]],
    ):
        """
        Sort cache in-place.

        Parameters
        ----------
        sorting : list[tuple[str, str]]
            Sorting arguments in PyArrow format (e.g. [('a', 'ascending'), ('b', 'descending')]).
        """
        self.flush()
        self._table = self._table.sort_by(sorting)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures data is flushed."""
        self.flush()

    def to_reader(self) -> MemoryCacheReader:
        """Get cache reader."""
        self.flush()
        return MemoryCacheReader(
            table=self._table, batch_size=self._batch_size
        )
