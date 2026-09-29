import pyarrow as pa
import pytest

from valor_lite.cache import FileCacheWriter, MemoryCacheWriter


@pytest.mark.parametrize("persistent", [False, True])
@pytest.mark.parametrize("batch_size,rows_per_file", [(1, 1), (4, 7), (10, 4)])
def test_mixed_batch_sizes_and_flushes(
    tmp_path, persistent, batch_size, rows_per_file
):
    schema = pa.schema([("id", pa.int64())])
    if persistent:
        writer = FileCacheWriter.create(
            tmp_path / "cache", schema, batch_size, rows_per_file
        )
    else:
        writer = MemoryCacheWriter.create(schema, batch_size)

    next_id = 0
    for sizes in ([1, 0, 2, 7, 1], [2, 1], [11, 1, 0, 3]):
        for size in sizes:
            writer.write_batch(
                pa.record_batch(
                    [
                        pa.array(
                            range(next_id, next_id + size), type=pa.int64()
                        )
                    ],
                    schema=schema,
                )
            )
            next_id += size
        writer.flush()
        writer.flush()
        actual = pa.concat_tables(list(writer.to_reader().iterate_tables()))
        assert actual["id"].to_pylist() == list(range(next_id))

    # Direct tables and subsequent buffered writes must not carry old sizes.
    writer.write_table(pa.table({"id": [next_id]}, schema=schema))
    writer.write_rows([{"id": next_id + 1}])
    actual = pa.concat_tables(list(writer.to_reader().iterate_tables()))
    assert actual["id"].to_pylist() == list(range(next_id + 2))
