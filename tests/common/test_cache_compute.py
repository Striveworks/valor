import random
from pathlib import Path

import pyarrow as pa
import pytest

from valor_lite.cache import FileCacheWriter, MemoryCacheWriter, sort


@pytest.fixture(
    params=[
        ("file_based", 10, 100),
        ("file_based", 1, 1),
        ("in_memory", 1_000, None),
        ("in_memory", 1, None),
    ],
    ids=[
        "file_based_large_chunks",
        "file_based_small_chunks",
        "in_memory_large_chunks",
        "in_memory_small_chunks",
    ],
)
def writer1(request, tmp_path: Path):
    file_type, batch_size, rows_per_file = request.param
    schema = pa.schema(
        [
            ("col0", pa.float64()),
            ("col1", pa.int64()),
        ]
    )
    match file_type:
        case "in_memory":
            return MemoryCacheWriter.create(
                schema=schema,
                batch_size=batch_size,
            )
        case "file_based":
            return FileCacheWriter.create(
                path=tmp_path / "cache1",
                schema=schema,
                batch_size=batch_size,
                rows_per_file=rows_per_file,
            )


@pytest.fixture(
    params=[
        ("file_based", 10, 100),
        ("file_based", 1, 1),
        ("in_memory", 1_000, None),
        ("in_memory", 1, None),
    ],
    ids=[
        "file_based_large_chunks",
        "file_based_small_chunks",
        "in_memory_large_chunks",
        "in_memory_small_chunks",
    ],
)
def writer2(request, tmp_path: Path):
    file_type, batch_size, rows_per_file = request.param
    schema = pa.schema(
        [
            ("col0", pa.float64()),
            ("col1", pa.int64()),
        ]
    )
    match file_type:
        case "in_memory":
            return MemoryCacheWriter.create(
                schema=schema,
                batch_size=batch_size,
            )
        case "file_based":
            return FileCacheWriter.create(
                path=tmp_path / "cache2",
                schema=schema,
                batch_size=batch_size,
                rows_per_file=rows_per_file,
            )


def test_cache_compute_sort(
    writer1: MemoryCacheWriter | FileCacheWriter,
    writer2: MemoryCacheWriter | FileCacheWriter,
):
    n_samples = 201
    sorting_args = [
        ("col1", "ascending"),
        ("col0", "descending"),
    ]

    # ingest to cache 1
    from tqdm import tqdm

    for i in tqdm(range(n_samples)):
        writer1.write_rows(
            [
                {
                    "col0": float(i),
                    "col1": random.randint(0, 100),
                }
            ]
        )
    writer1.flush()

    # sort cache 1, write into cache 2
    reader1 = writer1.to_reader()
    sort(
        source=reader1,
        sink=writer2,
        batch_size=10,
        sorting=sorting_args,
    )

    # validate sorted cache 2
    reader2 = writer2.to_reader()
    print(reader2.count_rows(), reader2.count_tables())
    prev_pair = None
    for pairs in reader2.iterate_pairs():
        for pair in pairs:
            if prev_pair is not None:
                assert pair[1] >= prev_pair[1]
                if pair[1] == prev_pair[1]:
                    assert pair[0] < prev_pair[0]
            prev_pair = pair
