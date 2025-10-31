import heapq

import pyarrow as pa

from valor_lite.cache.ephemeral import MemoryCacheReader, MemoryCacheWriter
from valor_lite.cache.persistent import FileCacheReader, FileCacheWriter


def heapsort(
    source: MemoryCacheReader | FileCacheReader,
    sink: MemoryCacheWriter | FileCacheWriter,
    batch_size: int,
    sorting: list[tuple[str, str]],
):
    """
    Perform heapsort on a cache object.

    Parameters
    ----------
    source : MemoryCacheReader | FileCacheReader
        The read-only source cache.
    sink : MemoryCacheWriter | FileCacheWriter
        The cache where sorted data will be written.
    batch_size : int
        Maximum number of rows allowed to be read into memory per cache file.
    sorting : list[tuple[str, str]]
        Sorting arguments in PyArrow format (e.g. [('a', 'ascending'), ('b', 'descending')]).
    """
    if source.count_tables() == 1 or isinstance(source, MemoryCacheReader):
        for tbl in source.iterate_tables():
            sorted_tbl = tbl.sort_by(sorting)
            sink.write_table(sorted_tbl)
    else:

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
        for batch_idx, batch_fragment in enumerate(source.iterate_fragments()):
            batch_iter = batch_fragment.to_batches(batch_size=batch_size)
            batch_iterators.append(batch_iter)
            batches.append(next(batch_iterators[batch_idx], None))
            if batches[batch_idx] is not None and len(batches[batch_idx]) > 0:
                heapq.heappush(heap, create_sort_key(batches, batch_idx, 0))

        while heap:
            _, _, batch_idx, row_idx = heapq.heappop(heap)
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
                if (
                    batches[batch_idx] is not None
                    and len(batches[batch_idx]) > 0
                ):
                    heapq.heappush(
                        heap,
                        create_sort_key(batches, batch_idx, 0),
                    )

    # flush any buffers
    sink.flush()
