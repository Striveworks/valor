import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from valor_lite.common.persistent import (
    FileCache,
    FileCacheReader,
    FileCacheWriter,
)


def test_cache_files_empty(tmp_path: Path):
    cache = FileCache(tmp_path)
    assert cache._path == tmp_path
    assert cache.get_files() == []
    assert cache.get_dataset_files() == []
    assert cache._generate_config_path(tmp_path) == tmp_path / ".cfg"


def test_cache_files_does_not_exist(tmp_path: Path):
    path = tmp_path / "does_not_exist"
    cache = FileCache(path)
    assert cache._path == path
    assert cache.get_files() == []
    assert cache.get_dataset_files() == []
    assert cache._generate_config_path(path) == path / ".cfg"


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
    with FileCacheWriter.create(
        path=tmp_path / "cache",
        schema=schema,
        batch_size=batch_size,
        rows_per_file=rows_per_file,
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
        reader = FileCacheReader.load(tmp_path / "cache")
        assert set(reader.get_files()) == set(
            [
                tmp_path / "cache" / "000000.parquet",
                tmp_path / "cache" / ".cfg",
            ]
        )
        assert set(reader.get_dataset_files()) == set(
            [
                tmp_path / "cache" / "000000.parquet",
            ]
        )

        writer.write_table(tbl)
        assert set(reader.get_files()) == set(
            [
                tmp_path / "cache" / "000000.parquet",
                tmp_path / "cache" / "000001.parquet",
                tmp_path / "cache" / ".cfg",
            ]
        )
        assert set(reader.get_dataset_files()) == set(
            [
                tmp_path / "cache" / "000000.parquet",
                tmp_path / "cache" / "000001.parquet",
            ]
        )

        for tbl in reader.iterate_tables():
            assert tbl["some_int"].to_pylist() == [i for i in range(101)]
            assert tbl["some_float"].to_pylist() == [i for i in range(101)]
            assert tbl["some_str"].to_pylist() == [
                f"str{i}" for i in range(101)
            ]

        assert reader.count_rows() == 202
        assert reader._schema == schema


def test_cache_reader_does_not_exist(tmp_path: Path):
    path = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        FileCacheReader.load(path)


def test_cache_reader_not_a_directory(tmp_path: Path):
    filepath = tmp_path / "not_a_directory"
    with open(filepath, "w") as f:
        f.write("hello world")

    with pytest.raises(NotADirectoryError):
        FileCacheReader.load(filepath)


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
    with FileCacheWriter.create(
        path=path,
        schema=schema,
        batch_size=batch_size,
        rows_per_file=rows_per_file,
    ) as writer:
        reader = FileCacheReader.load(writer.path)
        assert reader.path == path
        assert len(reader.get_files()) == 1
        assert reader.get_files() == [path / ".cfg"]
        assert len(reader.get_dataset_files()) == 0
        assert reader.get_dataset_files() == []

        # overwrite config file with empty JSON
        config_file = reader.get_files()[0]
        with open(config_file, "w") as f:
            json.dump({}, f, indent=2)

        with pytest.raises(KeyError):
            FileCacheReader.load(writer.path)


def test_cache_write_batch(tmp_path: Path):
    batch_size = 10
    rows_per_file = 100
    with FileCacheWriter.create(
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
    ) as writer:
        for i in range(1000):
            writer.write_batch(
                {
                    "some_int": [i],
                    "some_float": np.array([i], dtype=np.float64),
                    "some_str": pa.array([f"str{i}"]),
                }
            )
        writer.flush()
        assert len(writer.get_files()) == 11

        reader = FileCacheReader.load(writer.path)
        assert reader.count_rows() == 1000
        for idx, tbl in enumerate(reader.iterate_tables()):
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
    with FileCacheWriter.create(
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
        writer.flush()
        assert len(writer.get_files()) == 11

        reader = FileCacheReader.load(writer.path)
        assert reader.count_rows() == 1000
        for idx, tbl in enumerate(reader.iterate_tables()):
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
    with FileCacheWriter.create(
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
        assert len(writer.get_files()) == 2
        writer.write_table(tbl)
        assert len(writer.get_files()) == 3
        writer.flush()

        reader = FileCacheReader.load(writer.path)
        assert reader.count_rows() == 202
        for tbl in reader.iterate_tables():
            assert tbl["some_int"].to_pylist() == [i for i in range(101)]
            assert tbl["some_float"].to_pylist() == [i for i in range(101)]
            assert tbl["some_str"].to_pylist() == [
                f"str{i}" for i in range(101)
            ]


def test_cache_delete(tmp_path: Path):
    batch_size = 10
    rows_per_file = 100
    with FileCacheWriter.create(
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
        assert len(writer.get_files()) == 2
        writer.write_table(tbl)
        assert len(writer.get_files()) == 3
        writer.flush()

        reader = FileCacheReader.load(path=writer.path)
        assert reader.count_rows() == 202
        for tbl in reader.iterate_tables():
            assert tbl["some_int"].to_pylist() == [i for i in range(101)]
            assert tbl["some_float"].to_pylist() == [i for i in range(101)]
            assert tbl["some_str"].to_pylist() == [
                f"str{i}" for i in range(101)
            ]

        writer.delete()
        assert reader.get_files() == []

        # test edge case
        writer.delete()
