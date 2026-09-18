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
    for i in range(n_samples):
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
    prev_pair = None
    for pairs in reader2.iterate_arrays():
        for pair in pairs:
            if prev_pair is not None:
                assert pair[1] >= prev_pair[1]
                if pair[1] == prev_pair[1]:
                    assert pair[0] < prev_pair[0]
            prev_pair = pair


def test_cache_sort_by(writer1: MemoryCacheWriter | FileCacheWriter):

    n_samples = 201
    sorting_args = [
        ("col1", "ascending"),
        ("col0", "descending"),
    ]

    # ingest to cache 1
    for i in range(n_samples):
        writer1.write_rows(
            [
                {
                    "col0": float(i),
                    "col1": random.randint(0, 100),
                }
            ]
        )
    writer1.flush()
    writer1.sort_by(sorting_args)

    reader1 = writer1.to_reader()
    for pairs in reader1.iterate_arrays():
        prev_pair = None
        for pair in pairs:
            if prev_pair is not None:
                assert pair[1] >= prev_pair[1]
                if pair[1] == prev_pair[1]:
                    assert pair[0] < prev_pair[0]
            prev_pair = pair


@pytest.mark.parametrize("output_size", [1, 7, 64])
@pytest.mark.parametrize("input_size", [1, 5, 19])
@pytest.mark.parametrize("in_memory", [False, True])
def test_merge_preserves_ties_and_payload(
    tmp_path, output_size, input_size, in_memory
):
    schema = pa.schema(
        [
            ("score", pa.float64()),
            ("id", pa.uint64()),
            ("payload", pa.string()),
            ("metadata", pa.list_(pa.int64())),
        ]
    )
    source = FileCacheWriter.create(tmp_path / "source", schema, 10, 100)
    expected = []
    # Empty fragments and ties at batch boundaries must not drop any rows.
    for fragment, size in enumerate([0, 21, 0, 39, 17, 0]):
        rows = [
            {
                "score": float(index % 3),
                "id": 2**63 + index % 4,
                "payload": f"{fragment}:{index}",
                "metadata": None if index % 2 else [fragment, index],
            }
            for index in range(size)
        ]
        source.write_table(pa.Table.from_pylist(rows, schema=schema))
        expected.extend(rows)
    sorting = [("score", "descending"), ("id", "descending")]
    expected = sorted(expected, key=lambda row: (-row["score"], -row["id"]))
    if in_memory:
        sink = MemoryCacheWriter.create(schema, output_size)
    else:
        sink = FileCacheWriter.create(
            tmp_path / "sink", schema, output_size, 23
        )
    sort(source.to_reader(), sink, input_size, sorting)
    actual = pa.concat_tables(list(sink.to_reader().iterate_tables()))
    assert actual.equals(pa.Table.from_pylist(expected, schema=schema))
