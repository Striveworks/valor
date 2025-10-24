from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow as pa

from valor_lite.cache import (
    CacheReader,
    CacheWriter,
    DataType,
    convert_type_mapping_to_schema,
)


def test_datatype_casting_to_arrow():
    assert DataType.FLOAT.to_arrow() == pa.float64()
    assert DataType.INTEGER.to_arrow() == pa.int64()
    assert DataType.STRING.to_arrow() == pa.string()
    assert DataType.TIMESTAMP.to_arrow() == pa.timestamp("us")


def test_datatype_casting_to_python():
    assert DataType.FLOAT.to_py() is float
    assert DataType.INTEGER.to_py() is int
    assert DataType.STRING.to_py() is str
    assert DataType.TIMESTAMP.to_py() is datetime


def test_convert_type_mapping_to_schema():
    x = convert_type_mapping_to_schema(
        {
            "a": DataType.FLOAT,
            "b": DataType.STRING,
        }
    )
    assert x == [
        ("a", pa.float64()),
        ("b", pa.string()),
    ]

    assert convert_type_mapping_to_schema({}) == []
    assert convert_type_mapping_to_schema(None) == []


def test_cache_write_batch(tmp_path: Path):
    batch_size = 10
    rows_per_file = 100
    with CacheWriter.create(
        path=tmp_path / "cache",
        schema=pa.schema(
            [
                ("some_int", pa.int64()),
                ("some_float", pa.float64()),
                ("some_str", pa.string()),
            ]
        ),
        batch_size=batch_size,
        rows_per_file=rows_per_file,
    ) as cache:
        for i in range(1000):
            cache.write_batch(
                {
                    "some_int": [i],
                    "some_float": np.array([i], dtype=np.float64),
                    "some_str": pa.array([f"str{i}"]),
                }
            )
        cache.flush()
        assert cache.num_files == 11
        for idx, fragment in enumerate(cache.dataset.get_fragments()):
            tbl = fragment.to_table()
            assert tbl["some_int"].to_pylist() == [
                i
                for i in range(idx * rows_per_file, (idx + 1) * rows_per_file)
            ]
            assert tbl["some_float"].to_pylist() == [
                i
                for i in range(idx * rows_per_file, (idx + 1) * rows_per_file)
            ]
            assert tbl["some_str"].to_pylist() == [
                f"str{i}"
                for i in range(idx * rows_per_file, (idx + 1) * rows_per_file)
            ]


def test_cache_write_rows(tmp_path: Path):
    batch_size = 10
    rows_per_file = 100
    with CacheWriter.create(
        path=tmp_path / "cache",
        schema=pa.schema(
            [
                ("some_int", pa.int64()),
                ("some_float", pa.float64()),
                ("some_str", pa.string()),
            ]
        ),
        batch_size=batch_size,
        rows_per_file=rows_per_file,
    ) as cache:
        cache.write_rows([])
        for i in range(1000):
            cache.write_rows(
                [
                    {
                        "some_int": i,
                        "some_float": np.float64(i),
                        "some_str": f"str{i}",
                    }
                ]
            )
            buffer_size = len(cache._buffer)
            cache.write_rows([])
            assert len(cache._buffer) == buffer_size
        cache.flush()
        assert cache.num_files == 11
        for idx, fragment in enumerate(cache.dataset.get_fragments()):
            tbl = fragment.to_table()
            assert tbl["some_int"].to_pylist() == [
                i
                for i in range(idx * rows_per_file, (idx + 1) * rows_per_file)
            ]
            assert tbl["some_float"].to_pylist() == [
                i
                for i in range(idx * rows_per_file, (idx + 1) * rows_per_file)
            ]
            assert tbl["some_str"].to_pylist() == [
                f"str{i}"
                for i in range(idx * rows_per_file, (idx + 1) * rows_per_file)
            ]


def test_cache_write_table(tmp_path: Path):
    batch_size = 10
    rows_per_file = 100
    with CacheWriter.create(
        path=tmp_path / "cache",
        schema=pa.schema(
            [
                ("some_int", pa.int64()),
                ("some_float", pa.float64()),
                ("some_str", pa.string()),
            ]
        ),
        batch_size=batch_size,
        rows_per_file=rows_per_file,
    ) as cache:
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
        cache.write_table(tbl)
        assert cache.num_files == 2
        cache.write_table(tbl)
        assert cache.num_files == 3
        cache.flush()
        for _, fragment in enumerate(cache.dataset.get_fragments()):
            tbl = fragment.to_table()
            assert tbl["some_int"].to_pylist() == [i for i in range(101)]
            assert tbl["some_float"].to_pylist() == [i for i in range(101)]
            assert tbl["some_str"].to_pylist() == [
                f"str{i}" for i in range(101)
            ]


def test_cache_reader(tmp_path: Path):
    batch_size = 10
    rows_per_file = 100
    schema = pa.schema(
        [
            ("some_int", pa.int64()),
            ("some_float", pa.float64()),
            ("some_str", pa.string()),
        ]
    )
    with CacheWriter.create(
        path=tmp_path / "cache",
        schema=schema,
        batch_size=batch_size,
        rows_per_file=rows_per_file,
    ) as cache:
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
        cache.write_table(tbl)

        readonly_cache = CacheReader.load(tmp_path / "cache")
        assert readonly_cache.num_files == 2
        assert set(readonly_cache.files) == set(
            [
                tmp_path / "cache" / "000000.parquet",
                tmp_path / "cache" / ".cfg",
            ]
        )
        assert readonly_cache.num_dataset_files == 1
        assert set(readonly_cache.dataset_files) == set(
            [
                tmp_path / "cache" / "000000.parquet",
            ]
        )

        cache.write_table(tbl)
        assert readonly_cache.num_files == 3
        assert readonly_cache.num_dataset_files == 2

        for _, fragment in enumerate(readonly_cache.dataset.get_fragments()):
            tbl = fragment.to_table()
            assert tbl["some_int"].to_pylist() == [i for i in range(101)]
            assert tbl["some_float"].to_pylist() == [i for i in range(101)]
            assert tbl["some_str"].to_pylist() == [
                f"str{i}" for i in range(101)
            ]

        assert readonly_cache.dataset.count_rows() == 202
        assert readonly_cache.schema == schema


def test_cache_delete(tmp_path: Path):
    batch_size = 10
    rows_per_file = 100
    with CacheWriter.create(
        path=tmp_path / "cache",
        schema=pa.schema(
            [
                ("some_int", pa.int64()),
                ("some_float", pa.float64()),
                ("some_str", pa.string()),
            ]
        ),
        batch_size=batch_size,
        rows_per_file=rows_per_file,
    ) as cache:
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
        cache.write_table(tbl)
        assert cache.num_files == 2
        cache.write_table(tbl)
        assert cache.num_files == 3
        cache.flush()
        for _, fragment in enumerate(cache.dataset.get_fragments()):
            tbl = fragment.to_table()
            assert tbl["some_int"].to_pylist() == [i for i in range(101)]
            assert tbl["some_float"].to_pylist() == [i for i in range(101)]
            assert tbl["some_str"].to_pylist() == [
                f"str{i}" for i in range(101)
            ]

        CacheWriter.delete(tmp_path / "cache")
        assert cache.files == []
