import numpy as np
import pyarrow as pa

from valor_lite.common.ephemeral import MemoryCacheWriter


def test_cache_reader():
    batch_size = 10
    schema = pa.schema(
        [
            ("some_int", pa.int64()),
            ("some_float", pa.float64()),
            ("some_str", pa.string()),
        ]
    )
    with MemoryCacheWriter.create(
        schema=schema,
        batch_size=batch_size,
    ) as writer:
        tbl = pa.Table.from_pylist(
            [
                {
                    "some_int": i,
                    "some_float": np.float64(i),
                    "some_str": f"str{i}",
                }
                for i in range(101)
            ]
        )
        writer.write_table(tbl)
        assert writer.count_rows() == 101
        assert writer.count_tables() == 1

    reader = writer.to_reader()
    assert reader.count_rows() == 101
    assert reader.count_tables() == 1
    for tbl in reader.iterate_tables():
        assert tbl["some_int"].to_pylist() == [i for i in range(101)]
        assert tbl["some_float"].to_pylist() == [i for i in range(101)]
        assert tbl["some_str"].to_pylist() == [f"str{i}" for i in range(101)]
    assert reader.count_rows() == 101
    assert reader.count_tables() == 1
    assert reader.schema == schema


def test_cache_write_columns():
    batch_size = 10
    with MemoryCacheWriter.create(
        schema=pa.schema(
            [
                ("some_int", pa.int64()),
                ("some_float", pa.float64()),
                ("some_str", pa.string()),
            ]
        ),
        batch_size=batch_size,
    ) as writer:
        for i in range(1000):
            writer.write_columns(
                {
                    "some_int": [i],
                    "some_float": np.array([i], dtype=np.float64),
                    "some_str": pa.array([f"str{i}"]),
                }
            )
        writer.write_columns({})

    reader = writer.to_reader()
    assert reader.count_rows() == 1000
    assert reader.count_tables() == 1
    for tbl in reader.iterate_tables():
        assert tbl["some_int"].to_pylist() == [i for i in range(1000)]
        assert tbl["some_float"].to_pylist() == [i for i in range(1000)]
        assert tbl["some_str"].to_pylist() == [f"str{i}" for i in range(1000)]


def test_cache_write_rows():
    batch_size = 10
    with MemoryCacheWriter.create(
        schema=pa.schema(
            [
                ("some_int", pa.int64()),
                ("some_float", pa.float64()),
                ("some_str", pa.string()),
            ]
        ),
        batch_size=batch_size,
    ) as writer:
        writer.write_rows([])
        for i in range(1000):
            writer.write_rows(
                [
                    {
                        "some_int": i,
                        "some_float": np.float64(i),
                        "some_str": f"str{i}",
                    }
                ]
            )
            buffer_size = len(writer._buffer)
            writer.write_rows([])
            assert len(writer._buffer) == buffer_size

    reader = writer.to_reader()
    assert reader.count_rows() == 1000
    assert reader.count_tables() == 1
    for tbl in reader.iterate_tables():
        assert tbl["some_int"].to_pylist() == [i for i in range(1000)]
        assert tbl["some_float"].to_pylist() == [i for i in range(1000)]
        assert tbl["some_str"].to_pylist() == [f"str{i}" for i in range(1000)]


def test_cache_write_table():
    batch_size = 10
    with MemoryCacheWriter.create(
        schema=pa.schema(
            [
                ("some_int", pa.int64()),
                ("some_float", pa.float64()),
                ("some_str", pa.string()),
            ]
        ),
        batch_size=batch_size,
    ) as writer:
        tbl = pa.Table.from_pylist(
            [
                {
                    "some_int": i,
                    "some_float": np.float64(i),
                    "some_str": f"str{i}",
                }
                for i in range(101)
            ]
        )
        assert writer.count_rows() == 0
        assert writer.count_tables() == 1

        writer.write_table(tbl)
        assert writer.count_rows() == 101
        assert writer.count_tables() == 1

        writer.write_table(tbl)
        assert writer.count_rows() == 202
        assert writer.count_tables() == 1

        reader = writer.to_reader()

    for tbl in reader.iterate_tables():
        assert tbl["some_int"].to_pylist() == [i for i in range(101)] + [
            i for i in range(101)
        ]
        assert tbl["some_float"].to_pylist() == [i for i in range(101)] + [
            i for i in range(101)
        ]
        assert tbl["some_str"].to_pylist() == [
            f"str{i}" for i in range(101)
        ] + [f"str{i}" for i in range(101)]
