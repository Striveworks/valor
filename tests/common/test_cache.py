import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from valor_lite.cache import (
    CacheFiles,
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


def test_cache_files_empty(tmp_path: Path):
    cf = CacheFiles(tmp_path)
    assert cf.path == tmp_path
    assert cf.files == []
    assert cf.num_files == 0
    assert cf.dataset_files == []
    assert cf.num_dataset_files == 0
    assert cf._generate_config_path(tmp_path) == tmp_path / ".cfg"


def test_cache_files_not_a_directory(tmp_path: Path):
    filepath = tmp_path / "not_a_directory"
    with open(filepath, "w") as f:
        f.write("hello world")

    with pytest.raises(NotADirectoryError):
        CacheFiles(filepath)


def test_cache_files_does_not_exist(tmp_path: Path):
    path = tmp_path / "does_not_exist"
    cf = CacheFiles(path)
    assert cf.path == path
    assert cf.files == []
    assert cf.num_files == 0
    assert cf.dataset_files == []
    assert cf.num_dataset_files == 0
    assert cf._generate_config_path(path) == path / ".cfg"


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


def test_cache_reader_does_not_exist(tmp_path: Path):
    path = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        CacheReader.load(path)


def test_cache_reader_not_a_directory(tmp_path: Path):
    filepath = tmp_path / "not_a_directory"
    with open(filepath, "w") as f:
        f.write("hello world")

    with pytest.raises(NotADirectoryError):
        CacheReader.load(filepath)


def test_cache_reader_config_error(tmp_path: Path):
    batch_size = 10
    rows_per_file = 100
    schema = pa.schema(
        [
            ("some_int", pa.int64()),
            ("some_float", pa.float64()),
            ("some_str", pa.string()),
        ]
    )
    path = tmp_path / "cache"
    with CacheWriter.create(
        path=path,
        schema=schema,
        batch_size=batch_size,
        rows_per_file=rows_per_file,
    ) as writer:
        reader = CacheReader.load(writer.path)
        assert reader.path == path
        assert reader.num_files == 1
        assert reader.files == [path / ".cfg"]
        assert reader.num_dataset_files == 0
        assert reader.dataset_files == []

        # overwrite config file with empty JSON
        config_file = reader.files[0]
        with open(config_file, "w") as f:
            json.dump({}, f, indent=2)

        with pytest.raises(KeyError):
            CacheReader.load(writer.path)


def test_cache_writer_does_not_exist(tmp_path: Path):
    path = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        CacheWriter.load(path)


def test_cache_writer_not_a_directory(tmp_path: Path):
    filepath = tmp_path / "not_a_directory"
    with open(filepath, "w") as f:
        f.write("hello world")

    with pytest.raises(NotADirectoryError):
        CacheWriter.load(filepath)


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

        # test edge case
        CacheWriter.delete(tmp_path / "cache")
