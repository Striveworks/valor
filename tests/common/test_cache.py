import tempfile
import pyarrow as pa
import numpy as np
from pathlib import Path

from valor_lite.cache import Cache


def test_cache_write_batch():
    batch_size = 10
    rows_per_file = 100
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = Cache(
            where=Path(tmpdir),
            schema=pa.schema([
                ("some_int", pa.int64()),
                ("some_float", pa.float64()),
                ("some_str", pa.string()),
            ]),
            batch_size=batch_size,
            rows_per_file=rows_per_file,
        )
        for i in range(1000):
            cache.write_batch({
                "some_int": [i],
                "some_float": np.array([i], dtype=np.float64),
                "some_str": pa.array([f"str{i}"])
            })
        cache.flush()
        assert cache.num_files == 10
        for idx, fragment in enumerate(cache.dataset.get_fragments()):
            tbl = fragment.to_table()
            assert tbl["some_int"].to_pylist() == [i for i in range(idx*rows_per_file, (idx+1)*rows_per_file)] 
            assert tbl["some_float"].to_pylist() == [i for i in range(idx*rows_per_file, (idx+1)*rows_per_file)] 
            assert tbl["some_str"].to_pylist() == [f"str{i}" for i in range(idx*rows_per_file, (idx+1)*rows_per_file)]


def test_cache_write_rows():
    batch_size = 10
    rows_per_file = 100
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = Cache(
            where=Path(tmpdir),
            schema=pa.schema([
                ("some_int", pa.int64()),
                ("some_float", pa.float64()),
                ("some_str", pa.string()),
            ]),
            batch_size=batch_size,
            rows_per_file=rows_per_file,
        )
        for i in range(1000):
            cache.write_rows([{
                "some_int": i,
                "some_float": np.float64(i),
                "some_str": f"str{i}",
            }])
        cache.flush()
        assert cache.num_files == 10
        for idx, fragment in enumerate(cache.dataset.get_fragments()):
            tbl = fragment.to_table()
            assert tbl["some_int"].to_pylist() == [i for i in range(idx*rows_per_file, (idx+1)*rows_per_file)] 
            assert tbl["some_float"].to_pylist() == [i for i in range(idx*rows_per_file, (idx+1)*rows_per_file)] 
            assert tbl["some_str"].to_pylist() == [f"str{i}" for i in range(idx*rows_per_file, (idx+1)*rows_per_file)]


def test_cache_write_table():
    batch_size = 10
    rows_per_file = 100
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = Cache(
            where=Path(tmpdir),
            schema=pa.schema([
                ("some_int", pa.int64()),
                ("some_float", pa.float64()),
                ("some_str", pa.string()),
            ]),
            batch_size=batch_size,
            rows_per_file=rows_per_file,
        )
        tbl = pa.Table.from_pylist([
            {
                "some_int": i,
                "some_float": np.float64(i),
                "some_str": f"str{i}",
            }
            for i in range(101)
        ])
        cache.write_table(tbl)
        assert cache.num_files == 1
        cache.write_table(tbl)
        assert cache.num_files == 2
        cache.flush()
        for idx, fragment in enumerate(cache.dataset.get_fragments()):
            tbl = fragment.to_table()
            assert tbl["some_int"].to_pylist() == [i for i in range(101)] 
            assert tbl["some_float"].to_pylist() == [i for i in range(101)] 
            assert tbl["some_str"].to_pylist() == [f"str{i}" for i in range(101)]
        

def test_cache_sorting():
    batch_size = 10
    rows_per_file = 100
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = Cache(
            where=Path(tmpdir),
            schema=pa.schema([
                ("some_int", pa.int64()),
                ("some_float", pa.float64()),
                ("some_str", pa.string()),
            ]),
            batch_size=batch_size,
            rows_per_file=rows_per_file,
        )
        for i in range(1000):
            cache.write_rows([{
                "some_int": i,
                "some_float": np.float64(i),
                "some_str": f"str{i}",
            }])
        cache.flush()

        with tempfile.TemporaryDirectory() as sorteddir:
            sorted_cache = cache.sort_by(
                destination=Path(sorteddir),
                sorting=[
                    ("some_int", "descending"),
                ]
            )
            for idx, fragment in enumerate(sorted_cache.dataset.get_fragments()):
                tbl = fragment.to_table()
                assert tbl["some_int"].to_pylist() == [999 - i for i in range(idx*rows_per_file, (idx+1)*rows_per_file)] 
                assert tbl["some_float"].to_pylist() == [999 - i for i in range(idx*rows_per_file, (idx+1)*rows_per_file)] 
                assert tbl["some_str"].to_pylist() == [f"str{999 - i}" for i in range(idx*rows_per_file, (idx+1)*rows_per_file)]

        with tempfile.TemporaryDirectory() as sorteddir:
            sorted_cache = cache.sort_by(
                destination=Path(sorteddir),
                sorting=[
                    ("some_int", "ascending"),
                ]
            )
            for idx, fragment in enumerate(sorted_cache.dataset.get_fragments()):
                tbl = fragment.to_table()
                assert tbl["some_int"].to_pylist() == [i for i in range(idx*rows_per_file, (idx+1)*rows_per_file)] 
                assert tbl["some_float"].to_pylist() == [i for i in range(idx*rows_per_file, (idx+1)*rows_per_file)] 
                assert tbl["some_str"].to_pylist() == [f"str{i}" for i in range(idx*rows_per_file, (idx+1)*rows_per_file)]
