import numpy as np
import pyarrow as pa

from valor_lite.common.ephemeral import MemoryCacheReader, MemoryCacheWriter


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

    reader = MemoryCacheReader.load(writer)
    assert reader.count_rows() == 101
    for tbl in reader.iterate_tables():
        assert tbl["some_int"].to_pylist() == [i for i in range(101)]
        assert tbl["some_float"].to_pylist() == [i for i in range(101)]
        assert tbl["some_str"].to_pylist() == [f"str{i}" for i in range(101)]
    assert reader.count_rows() == 101
    assert reader._schema == schema


def test_cache_write_batch():
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
            writer.write_batch(
                {
                    "some_int": [i],
                    "some_float": np.array([i], dtype=np.float64),
                    "some_str": pa.array([f"str{i}"]),
                }
            )

    reader = MemoryCacheReader(writer)
    assert reader.count_rows() == 1000
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

    reader = MemoryCacheReader.load(writer)
    assert reader.count_rows() == 1000
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

        writer.write_table(tbl)
        assert writer.count_rows() == 101

        writer.write_table(tbl)
        assert writer.count_rows() == 202

        reader = MemoryCacheReader.load(writer)

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


def test_cache_delete():
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

        writer.write_table(tbl)
        assert writer.count_rows() == 101

        writer.write_table(tbl)
        assert writer.count_rows() == 202

        writer.flush()

        reader = MemoryCacheReader.load(writer)
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

        writer.delete()
        assert reader.count_rows() == 0

        # test edge case
        writer.delete()
