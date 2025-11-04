from typing import Callable

import numpy as np
import pyarrow as pa

from valor_lite.cache import (
    FileCacheReader,
    FileCacheWriter,
    MemoryCacheWriter,
)


def test_cache_reader_iterate_tables_with_arrays(
    create_writer: Callable[[pa.Schema], FileCacheWriter | MemoryCacheWriter]
):
    n_rows_written = 100
    schema = pa.schema(
        [
            ("a", "int64"),
            ("b", "double"),
            ("c", "string"),
            ("d", "string"),
        ]
    )
    writer = create_writer(schema)
    for i in range(n_rows_written):
        writer.write_rows(
            [
                {
                    "a": i,
                    "b": float(i) + 0.1,
                    "c": f"string_{i}",
                    "d": f"other_{i}",
                }
            ]
        )
    writer.flush()

    reader = writer.to_reader()
    n_tables = reader.count_tables()
    n_rows = reader.count_rows()

    assert n_rows == n_rows_written
    if isinstance(reader, FileCacheReader):
        expected_n_tables = n_rows // reader.rows_per_file
        if n_rows % reader.rows_per_file != 0:
            expected_n_tables += 1
        assert n_tables == expected_n_tables
    else:
        assert n_tables == 1

    accumulated_rows = 0
    for i, (tbl, array) in enumerate(
        reader.iterate_tables_with_arrays(
            columns=["c"],
            numeric_columns=["a", "b"],
        )
    ):
        # check table
        assert tbl.num_columns == 3
        assert set(tbl.column_names) == set(["a", "b", "c"])
        if isinstance(reader, FileCacheReader):
            if i == n_tables - 1:
                if reader.rows_per_file == 1:
                    assert tbl.num_rows == 1
                else:
                    assert tbl.num_rows == n_rows % reader.rows_per_file
            else:
                assert tbl.num_rows == reader.rows_per_file
        else:
            assert tbl.num_rows == n_rows

        # test array
        assert array.shape
        expected_array = np.concatenate(
            [
                np.arange(0, tbl.num_rows)[:, np.newaxis],
                np.arange(0, tbl.num_rows)[:, np.newaxis] + 0.1,
            ],
            axis=1,
        )
        expected_array += accumulated_rows
        accumulated_rows += tbl.num_rows
