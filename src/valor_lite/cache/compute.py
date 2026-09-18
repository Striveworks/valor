import heapq
import tempfile
from pathlib import Path
from typing import Callable, Generator

import pyarrow as pa
import pyarrow.compute as pc

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
    memory_fragments = []
    for tbl in source.iterate_tables(columns=columns):
        if table_sort_override is not None:
            sorted_tbl = table_sort_override(tbl)
        else:
            sorted_tbl = tbl.sort_by(sorting)
        if isinstance(intermediate_sink, MemoryCacheWriter):
            # Keep each sorted run separate; concatenation is not a merge.
            memory_fragments.append(sorted_tbl)
        else:
            intermediate_sink.write_table(sorted_tbl)
    if isinstance(intermediate_sink, MemoryCacheWriter):
        fragment_iterators = (
            iter(tbl.to_batches(max_chunksize=batch_size))
            for tbl in memory_fragments
        )
    else:
        intermediate_source = intermediate_sink.to_reader()
        fragment_iterators = (
            intermediate_source.iterate_fragment_batch_iterators(
                batch_size=batch_size
            )
        )

    # Materialize numeric sort keys once per input batch. Python scalars retain
    # integer precision and avoid an Arrow scalar conversion for every row.
    sort_values = []

    def create_sort_key(batch_idx: int, row_idx: int):
        return (
            *(values[row_idx] for values in sort_values[batch_idx]),
            batch_idx,
            row_idx,
        )

    heap = []
    batch_iterators = []
    batches = []
    output_batches = []
    output_offsets = {}
    output_indices = []
    output_size = 0
    output_batch_size = min(sink.batch_size, 65_536)
    if isinstance(sink, FileCacheWriter):
        output_batch_size = min(output_batch_size, sink.rows_per_file)
    output_batch_size = max(1, output_batch_size)

    def load_batch(batch_idx: int):
        batch = next(batch_iterators[batch_idx], None)
        while batch is not None and batch.num_rows == 0:
            batch = next(batch_iterators[batch_idx], None)
        batches[batch_idx] = batch
        output_offsets.pop(batch_idx, None)
        if batch is None:
            sort_values[batch_idx] = []
            return False
        values = [batch[name].to_pylist() for name, _ in sorting]
        sort_values[batch_idx] = [
            [-value for value in column]
            if direction == "descending"
            else column
            for column, (_, direction) in zip(values, sorting)
        ]
        return True

    def flush_output():
        nonlocal output_size
        if not output_indices:
            return
        # Gather in one Arrow operation, retaining all payload columns and
        # exact heap order. Input batches stay alive until their rows are copied.
        table = pa.Table.from_batches(output_batches).take(
            pa.array(output_indices, type=pa.int64())
        )
        for batch in table.to_batches(max_chunksize=output_batch_size):
            sink.write_batch(batch)
        output_batches.clear()
        output_offsets.clear()
        output_indices.clear()
        output_size = 0

    for batch_idx, batch_iter in enumerate(fragment_iterators):
        batch_iterators.append(batch_iter)
        batches.append(None)
        sort_values.append([])
        if load_batch(batch_idx):
            heap.append(create_sort_key(batch_idx, 0))
    heapq.heapify(heap)

    while heap:
        row = heapq.heappop(heap)
        batch_idx = row[-2]
        row_idx = row[-1]
        if batch_idx not in output_offsets:
            output_offsets[batch_idx] = output_size
            output_batches.append(batches[batch_idx])
            output_size += batches[batch_idx].num_rows
        output_indices.append(output_offsets[batch_idx] + row_idx)
        if len(output_indices) >= output_batch_size:
            flush_output()
        row_idx += 1
        if row_idx < len(batches[batch_idx]):
            heapq.heappush(
                heap,
                create_sort_key(batch_idx, row_idx),
            )
        else:
            if load_batch(batch_idx):
                heapq.heappush(
                    heap,
                    create_sort_key(batch_idx, 0),
                )

    flush_output()
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


def paginate_index(
    source: MemoryCacheReader | FileCacheReader,
    column_key: str,
    modifier: pc.Expression | None = None,
    limit: int | None = None,
    offset: int = 0,
) -> Generator[pa.Table, None, None]:
    """
    Iterate through a paginated cache reader.

    Note this function expects unqiue keys to be fragment-aligned.
    """
    total = source.count_rows()
    limit = limit if limit else total

    # pagination broader than data scope
    if offset == 0 and limit >= total:
        for tbl in source.iterate_tables(filter=modifier):
            yield tbl
        return
    elif offset >= total:
        return

    curr_idx = 0
    for tbl in source.iterate_tables(filter=modifier):
        if tbl.num_rows == 0:
            continue

        # sort the unique keys as they may be out of order
        unique_values = pc.unique(tbl[column_key]).sort()  # type: ignore[reportAttributeAccessIssue]
        n_unique = len(unique_values)
        prev_idx = curr_idx
        curr_idx += n_unique

        # check for page overlap
        if curr_idx <= offset:
            continue
        elif prev_idx >= (offset + limit):
            return

        # apply any pagination conditions
        condition = pc.scalar(True)
        if prev_idx < offset and curr_idx > offset:
            condition &= (
                pc.field(column_key) >= unique_values[offset - prev_idx]
            )
        if prev_idx < (offset + limit) and curr_idx > (offset + limit):
            condition &= (
                pc.field(column_key) < unique_values[offset + limit - prev_idx]
            )

        yield tbl.filter(condition)
