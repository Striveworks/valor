from typing import Callable

import pyarrow as pa
import pyarrow.compute as pc
import pytest

from valor_lite.cache import (
    FileCacheReader,
    FileCacheWriter,
    MemoryCacheReader,
    MemoryCacheWriter,
)
from valor_lite.cache.compute import paginate_index


@pytest.fixture
def reader(
    create_writer: Callable[[pa.Schema], MemoryCacheWriter | FileCacheWriter]
) -> MemoryCacheReader | FileCacheReader:
    schema = pa.schema(
        [
            ("a", pa.int64()),
            ("b", pa.int64()),
        ]
    )
    writer = create_writer(schema)
    writer.write_columns(
        {
            "a": list(range(100)),
            "b": [i % 10 for i in range(100)],
        }
    )
    writer.flush()
    return writer.to_reader()


def test_paginate_index_skipping(reader: MemoryCacheReader | FileCacheReader):
    # offset = 0, limit = None, modifier = None
    actual = []
    for tbl in paginate_index(
        reader,
        column_key="a",
        modifier=None,
        limit=None,
        offset=0,
    ):
        actual += tbl["a"].to_pylist()
    assert actual == list(range(100))

    # offset = 0, limit = 100, modifier = None
    actual = []
    for tbl in paginate_index(
        reader,
        column_key="a",
        modifier=None,
        limit=None,
        offset=0,
    ):
        actual += tbl["a"].to_pylist()
    assert actual == list(range(100))


def test_paginate_index_offset(reader: MemoryCacheReader | FileCacheReader):
    # limit = None, modifier = None
    for offset in range(0, 101):
        actual = []
        for tbl in paginate_index(
            reader,
            column_key="a",
            modifier=None,
            limit=None,
            offset=offset,
        ):
            actual += tbl["a"].to_pylist()
        assert actual == list(range(offset, 100))

    # offset = 1_000, limit=None, modifier = None
    actual = []
    for tbl in paginate_index(
        reader,
        column_key="a",
        modifier=None,
        limit=None,
        offset=1_000,
    ):
        actual += tbl["a"].to_pylist()
    assert actual == []


def test_paginate_index_limit(reader: MemoryCacheReader | FileCacheReader):
    # offset = 0, modifier = None
    for limit in range(1, 101):
        actual = []
        for tbl in paginate_index(
            reader,
            column_key="a",
            modifier=None,
            limit=limit,
            offset=0,
        ):
            actual += tbl["a"].to_pylist()
        assert actual == list(range(0, limit))


def test_pagination_index_modifer(reader: MemoryCacheReader | FileCacheReader):
    # offset = 0, limit = None, modifer = 4
    actual = []
    for tbl in paginate_index(
        reader,
        column_key="a",
        modifier=pc.field("b") == 4,
        limit=None,
        offset=0,
    ):
        actual += tbl["a"].to_pylist()
    assert actual == [4, 14, 24, 34, 44, 54, 64, 74, 84, 94]

    # offset = 5, limit = None, modifer = 4
    actual = []
    for tbl in paginate_index(
        reader,
        column_key="a",
        modifier=pc.field("b") == 4,
        limit=None,
        offset=5,
    ):
        actual += tbl["a"].to_pylist()
    assert actual == [54, 64, 74, 84, 94]

    # offset = 5, limit = 2, modifer = 4
    actual = []
    for tbl in paginate_index(
        reader,
        column_key="a",
        modifier=pc.field("b") == 4,
        limit=2,
        offset=5,
    ):
        actual += tbl["a"].to_pylist()
    assert actual == [54, 64]
