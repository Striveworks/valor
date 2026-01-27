from pathlib import Path
from typing import Callable

import pyarrow as pa
import pytest

from valor_lite.cache.ephemeral import MemoryCacheWriter
from valor_lite.cache.persistent import FileCacheWriter


@pytest.fixture(
    params=[
        ("persistent", 10_000, 100_000),
        ("persistent", 1, 1),
        ("memory", 10_000, 0),
        ("memory", 1, 0),
    ],
    ids=[
        "persistent_large_chunks",
        "persistent_small_chunks",
        "in-memory_large_chunks",
        "in-memory_small_chunks",
    ],
)
def create_writer(
    request, tmp_path: Path
) -> Callable[[pa.Schema], MemoryCacheWriter | FileCacheWriter]:
    file_type, batch_size, rows_per_file = request.param
    match file_type:
        case "memory":
            return lambda schema: MemoryCacheWriter.create(
                schema=schema,
                batch_size=batch_size,
            )
        case "persistent":
            return lambda schema: FileCacheWriter.create(
                path=tmp_path / "cache",
                schema=schema,
                batch_size=batch_size,
                rows_per_file=rows_per_file,
            )
        case unknown:
            raise RuntimeError(unknown)
