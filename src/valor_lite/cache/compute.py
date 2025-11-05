import heapq
import tempfile
from pathlib import Path
from typing import Callable

import pyarrow as pa

from valor_lite.cache.ephemeral import MemoryCacheReader, MemoryCacheWriter
from valor_lite.cache.persistent import FileCacheReader, FileCacheWriter


def _merge(
    source: MemoryCacheReader | FileCacheReader,
    sink: MemoryCacheWriter | FileCacheWriter,
    intermediate_sink: MemoryCacheWriter | FileCacheWriter,
    batch_size: int,
    sorting: list[tuple[str, str]],
    columns: list[str] | None = None,
    table_sort_override: Callable[[pa.Table], pa.Table] | None = None,
):
    """Merge locally sorted cache fragments."""
    for tbl in source.iterate_tables(columns=columns):
        if table_sort_override is not None:
            sorted_tbl = table_sort_override(tbl)
        else:
            sorted_tbl = tbl.sort_by(sorting)
        intermediate_sink.write_table(sorted_tbl)
    intermediate_source = intermediate_sink.to_reader()

    # define merge key
    def create_sort_key(
        batches: list[pa.RecordBatch],
        batch_idx: int,
        row_idx: int,
    ):
        args = [
            -batches[batch_idx][name][row_idx].as_py()
            if direction == "descending"
            else batches[batch_idx][name][row_idx].as_py()
            for name, direction in sorting
        ]
        return (
            *args,
            batch_idx,
            row_idx,
        )

    # merge sorted rows
    heap = []
    batch_iterators = []
    batches = []
    for batch_idx, batch_iter in enumerate(
        intermediate_source.iterate_fragments(batch_size=batch_size)
    ):
        batch_iterators.append(batch_iter)
        batches.append(next(batch_iterators[batch_idx], None))
        if batches[batch_idx] is not None and len(batches[batch_idx]) > 0:
            heapq.heappush(heap, create_sort_key(batches, batch_idx, 0))

    while heap:
        row = heapq.heappop(heap)
        batch_idx = row[-2]
        row_idx = row[-1]
        row_table = batches[batch_idx].slice(row_idx, 1)
        sink.write_batch(row_table)
        row_idx += 1
        if row_idx < len(batches[batch_idx]):
            heapq.heappush(
                heap,
                create_sort_key(batches, batch_idx, row_idx),
            )
        else:
            batches[batch_idx] = next(batch_iterators[batch_idx], None)
            if batches[batch_idx] is not None and len(batches[batch_idx]) > 0:
                heapq.heappush(
                    heap,
                    create_sort_key(batches, batch_idx, 0),
                )

    sink.flush()


def sort(
    source: MemoryCacheReader | FileCacheReader,
    sink: MemoryCacheWriter | FileCacheWriter,
    batch_size: int,
    sorting: list[tuple[str, str]],
    columns: list[str] | None = None,
    table_sort_override: Callable[[pa.Table], pa.Table] | None = None,
):
    """
    Sort data into new cache.

    Parameters
    ----------
    source : MemoryCacheReader | FileCacheReader
        A read-only cache. If file-based, each file must be locally sorted.
    sink : MemoryCacheWriter | FileCacheWriter
        The cache where sorted data will be written.
    batch_size : int
        Maximum number of rows allowed to be read into memory per cache file.
    sorting : list[tuple[str, str]]
        Sorting arguments in PyArrow format (e.g. [('a', 'ascending'), ('b', 'descending')]).
        Note that only numeric fields are supported.
    columns : list[str], optional
        Option to only read a subset of columns.
    table_sort_override : Callable[[pa.Table], pa.Table], optional
        Option to override sort function for singular cache fragments.
    """
    if source.count_tables() == 1:
        for tbl in source.iterate_tables(columns=columns):
            if table_sort_override is not None:
                sorted_tbl = table_sort_override(tbl)
            else:
                sorted_tbl = tbl.sort_by(sorting)
            sink.write_table(sorted_tbl)
        sink.flush()
        return

    if isinstance(sink, FileCacheWriter):
        with tempfile.TemporaryDirectory() as tmpdir:
            intermediate_sink = FileCacheWriter.create(
                path=Path(tmpdir) / "sorting_intermediate",
                schema=sink.schema,
                batch_size=sink.batch_size,
                rows_per_file=sink.rows_per_file,
                compression=sink.compression,
                delete_if_exists=False,
            )
            _merge(
                source=source,
                sink=sink,
                intermediate_sink=intermediate_sink,
                batch_size=batch_size,
                sorting=sorting,
                columns=columns,
                table_sort_override=table_sort_override,
            )
    else:
        intermediate_sink = MemoryCacheWriter.create(
            schema=sink.schema,
            batch_size=sink.batch_size,
        )
        _merge(
            source=source,
            sink=sink,
            intermediate_sink=intermediate_sink,
            batch_size=batch_size,
            sorting=sorting,
            columns=columns,
            table_sort_override=table_sort_override,
        )
