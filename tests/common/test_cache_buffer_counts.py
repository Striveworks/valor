import pyarrow as pa
import pytest

from valor_lite.cache import FileCacheWriter, MemoryCacheWriter


@pytest.mark.parametrize("persistent", [False, True])
def test_buffer_counts_across_batch_boundaries_and_flush(tmp_path, persistent):
    schema = pa.schema([("id", pa.int64())])
    if persistent:
        writer = FileCacheWriter.create(
            path=tmp_path / "cache",
            schema=schema,
            batch_size=5,
            rows_per_file=7,
        )
    else:
        writer = MemoryCacheWriter.create(schema=schema, batch_size=5)

    expected = []
    for index, size in enumerate([1, 2, 2, 1, 1, 6, 0, 3, 2, 1]):
        values = list(range(len(expected), len(expected) + size))
        writer.write_batch(pa.record_batch([values], schema=schema))
        expected.extend(values)
        assert writer._buffer_rows == sum(b.num_rows for b in writer._buffer)
        if index in (3, 7):
            writer.flush()
            assert writer._buffer_rows == 0
    writer.flush()
    writer.flush()
    assert writer._buffer_rows == 0
    actual = [
        value
        for table in writer.to_reader().iterate_tables()
        for value in table["id"].to_pylist()
    ]
    assert actual == expected


def test_file_buffer_count_reset_by_write_table(tmp_path):
    schema = pa.schema([("id", pa.int64())])
    writer = FileCacheWriter.create(
        path=tmp_path / "cache",
        schema=schema,
        batch_size=5,
        rows_per_file=7,
    )
    writer.write_batch(pa.record_batch([[0, 1]], schema=schema))
    writer.write_table(pa.table({"id": [2]}, schema=schema))
    assert writer._buffer_rows == 0
    writer.write_batch(pa.record_batch([[3, 4]], schema=schema))
    assert writer._buffer_rows == 2
    writer.flush()
    assert writer._buffer_rows == 0
    assert writer.to_reader().count_rows() == 5
