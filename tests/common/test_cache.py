import tempfile
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


def test_cache_write_batch():
    batch_size = 10
    rows_per_file = 100
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = CacheWriter(
            where=Path(tmpdir),
            schema=pa.schema(
                [
                    ("some_int", pa.int64()),
                    ("some_float", pa.float64()),
                    ("some_str", pa.string()),
                ]
            ),
            batch_size=batch_size,
            rows_per_file=rows_per_file,
        )
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


def test_cache_write_rows():
    batch_size = 10
    rows_per_file = 100
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = CacheWriter(
            where=Path(tmpdir),
            schema=pa.schema(
                [
                    ("some_int", pa.int64()),
                    ("some_float", pa.float64()),
                    ("some_str", pa.string()),
                ]
            ),
            batch_size=batch_size,
            rows_per_file=rows_per_file,
        )
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


def test_cache_write_table():
    batch_size = 10
    rows_per_file = 100
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = CacheWriter(
            where=Path(tmpdir),
            schema=pa.schema(
                [
                    ("some_int", pa.int64()),
                    ("some_float", pa.float64()),
                    ("some_str", pa.string()),
                ]
            ),
            batch_size=batch_size,
            rows_per_file=rows_per_file,
        )
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


def test_cache_reader():
    batch_size = 10
    rows_per_file = 100
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = CacheWriter(
            where=Path(tmpdir),
            schema=pa.schema(
                [
                    ("some_int", pa.int64()),
                    ("some_float", pa.float64()),
                    ("some_str", pa.string()),
                ]
            ),
            batch_size=batch_size,
            rows_per_file=rows_per_file,
        )
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

        readonly_cache = CacheReader(where=Path(tmpdir))
        assert readonly_cache.num_files == 2
        assert readonly_cache.files == [
            tmpdir + "/000000.parquet",
            tmpdir + "/.cfg",
        ]
        assert readonly_cache.num_dataset_files == 1
        assert readonly_cache.dataset_files == [
            tmpdir + "/000000.parquet",
        ]

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
