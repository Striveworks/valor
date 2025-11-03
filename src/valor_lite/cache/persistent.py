import base64
import glob
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq


class FileCache:
    def __init__(
        self,
        path: str | Path,
        schema: pa.Schema,
        batch_size: int,
        rows_per_file: int,
        compression: str,
    ):
        self._path = Path(path)
        self._schema = schema
        self._batch_size = batch_size
        self._rows_per_file = rows_per_file
        self._compression = compression

    @property
    def path(self) -> Path:
        return self._path

    @property
    def schema(self) -> pa.Schema:
        return self._schema

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def rows_per_file(self) -> int:
        return self._rows_per_file

    @property
    def compression(self) -> str:
        return self._compression

    @staticmethod
    def _generate_config_path(path: str | Path) -> Path:
        """Generate cache configuration path."""
        return Path(path) / ".cfg"

    @staticmethod
    def _encode_schema(schema: pa.Schema) -> str:
        """Encode schema to b64 string."""
        schema_bytes = schema.serialize()
        return base64.b64encode(schema_bytes).decode("utf-8")

    @staticmethod
    def _decode_schema(encoded_schema: str) -> pa.Schema:
        """Decode schema from b64 string."""
        schema_bytes = base64.b64decode(encoded_schema)
        return pa.ipc.read_schema(pa.BufferReader(schema_bytes))

    def count_rows(self) -> int:
        """Count the number of rows in the cache."""
        dataset = ds.dataset(
            source=self._path,
            format="parquet",
        )
        return dataset.count_rows()

    def count_tables(self) -> int:
        """Count the number of files in the cache."""
        return len(self.get_dataset_files())

    def get_files(self) -> list[Path]:
        """
        Retrieve all files.

        Returns
        -------
        list[Path]
            A list of paths to files in the cache.
        """
        if not self._path.exists():
            return []
        files = []
        for entry in os.listdir(self._path):
            full_path = os.path.join(self._path, entry)
            if os.path.isfile(full_path):
                files.append(Path(full_path))
        return files

    def get_dataset_files(self) -> list[Path]:
        """
        Retrieve all dataset files.

        Returns
        -------
        list[Path]
            A list of paths to dataset files in the cache.
        """
        if not self._path.exists():
            return []
        return [
            Path(filepath) for filepath in glob.glob(f"{self._path}/*.parquet")
        ]


class FileCacheReader(FileCache):
    @classmethod
    def load(cls, path: str | Path):
        """
        Load cache from disk.

        Parameters
        ----------
        path : str | Path
            Where the cache is stored.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Directory does not exist: {path}")
        elif not path.is_dir():
            raise NotADirectoryError(
                f"Path exists but is not a directory: {path}"
            )

        def _retrieve(config: dict, key: str):
            if value := config.get(key, None):
                return value
            raise KeyError(
                f"'{key}' is not defined within {cls._generate_config_path(path)}"
            )

        # read configuration file
        cfg_path = cls._generate_config_path(path)
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
            batch_size = _retrieve(cfg, "batch_size")
            rows_per_file = _retrieve(cfg, "rows_per_file")
            compression = _retrieve(cfg, "compression")
            schema = cls._decode_schema(_retrieve(cfg, "schema"))

        return cls(
            schema=schema,
            path=path,
            batch_size=batch_size,
            rows_per_file=rows_per_file,
            compression=compression,
        )

    def iterate_tables(
        self,
        columns: list[str] | None = None,
        filter: pc.Expression | None = None,
    ) -> Iterator[pa.Table]:
        """
        Iterate over tables within the cache.

        Parameters
        ----------
        columns : list[str], optional
            Optionally select columns to be returned.
        filter : pyarrow.compute.Expression, optional
            Optionally filter table before returning.

        Returns
        -------
        Iterator[pa.Table]
        """
        dataset = ds.dataset(
            source=self._path,
            schema=self._schema,
            format="parquet",
        )
        for fragment in dataset.get_fragments():
            yield fragment.to_table(columns=columns, filter=filter)

    def iterate_arrays(
        self,
        numeric_columns: list[str] | None = None,
        filter: pc.Expression | None = None,
    ) -> Iterator[np.ndarray]:
        """
        Iterate over chunks within the cache returning arrays.

        Parameters
        ----------
        numeric_columns : list[str], optional
            Optionally select numeric columns to be returned within an array.
        filter : pyarrow.compute.Expression, optional
            Optionally filter table before returning.

        Returns
        -------
        Iterator[np.ndarray]
        """
        for tbl in self.iterate_tables(
            columns=numeric_columns,
            filter=filter,
        ):
            yield np.column_stack(
                [tbl.column(i).to_numpy() for i in range(tbl.num_columns)]
            )

    def iterate_tables_with_arrays(
        self,
        columns: list[str] | None = None,
        filter: pc.Expression | None = None,
        numeric_columns: list[str] | None = None,
    ) -> Iterator[tuple[pa.Table, np.ndarray]]:
        """
        Iterate over chunks within the cache returning both tables and arrays.

        Parameters
        ----------
        columns : list[str], optional
            Optionally select columns to be returned.
        filter : pyarrow.compute.Expression, optional
            Optionally filter table before returning.
        numeric_columns : list[str], optional
            Optionally select numeric columns to be returned within an array.

        Returns
        -------
        Iterator[tuple[pa.Table, np.ndarray]]

        """
        for tbl in self.iterate_tables(
            columns=columns,
            filter=filter,
        ):
            columns = numeric_columns if numeric_columns else tbl.columns
            yield tbl, np.column_stack(
                [tbl[col].to_numpy() for col in columns]
            )

    def iterate_fragments(
        self, batch_size: int
    ) -> Iterator[Iterator[pa.RecordBatch]]:
        """
        Iterate over fragment batch iterators within the file-based cache.

        Parameters
        ----------
        batch_size : int
            Maximum number of rows allowed to be read into memory per cache file.

        Returns
        -------
        Iterator[Iterator[pa.RecordBatch]]
        """
        dataset = ds.dataset(
            source=self._path,
            schema=self._schema,
            format="parquet",
        )
        for fragment in dataset.get_fragments():
            yield fragment.to_batches(batch_size=batch_size)


class FileCacheWriter(FileCache):
    def __init__(
        self,
        path: str | Path,
        schema: pa.Schema,
        batch_size: int,
        rows_per_file: int,
        compression: str,
    ):
        super().__init__(
            path=path,
            schema=schema,
            batch_size=batch_size,
            rows_per_file=rows_per_file,
            compression=compression,
        )

        # internal state
        self._writer = None
        self._buffer = []
        self._count = 0

    @classmethod
    def create(
        cls,
        path: str | Path,
        schema: pa.Schema,
        batch_size: int,
        rows_per_file: int,
        compression: str = "snappy",
        delete_if_exists: bool = False,
    ):
        """
        Create an on-disk cache.

        Parameters
        ----------
        path : str | Path
            Where to write the cache.
        schema : pa.Schema
            Cache schema.
        batch_size : int
            Target batch size when writing chunks.
        rows_per_file : int
            Target number of rows to store per file.
        compression : str, default="snappy"
            Compression method to use when storing on disk.
        delete_if_exists : bool, default=False
            Delete the cache if it already exists.
        """
        path = Path(path)
        if delete_if_exists and path.exists():
            cls.delete(path)
        Path(path).mkdir(parents=True, exist_ok=False)

        # write configuration file
        cfg_path = cls._generate_config_path(path)
        with open(cfg_path, "w") as f:
            cfg = dict(
                batch_size=batch_size,
                rows_per_file=rows_per_file,
                compression=compression,
                schema=cls._encode_schema(schema),
            )
            json.dump(cfg, f, indent=2)

        return cls(
            schema=schema,
            path=path,
            batch_size=batch_size,
            rows_per_file=rows_per_file,
            compression=compression,
        )

    @classmethod
    def delete(cls, path: str | Path):
        """
        Delete a cache at path.

        Parameters
        ----------
        path : str | Path
            Where the cache is stored.
        """
        path = Path(path)
        if not path.exists():
            return

        # delete dataset files
        reader = FileCacheReader.load(path)
        for file in reader.get_dataset_files():
            if file.exists() and file.is_file() and file.suffix == ".parquet":
                file.unlink()

        # delete config file
        cfg_path = cls._generate_config_path(path)
        if cfg_path.exists() and cfg_path.is_file():
            cfg_path.unlink()

        # delete empty cache directory
        path.rmdir()

    def write_rows(
        self,
        rows: list[dict[str, Any]],
    ):
        """
        Write rows to cache.

        Parameters
        ----------
        rows : list[dict[str, Any]]
            A list of rows represented by dictionaries mapping fields to values.
        """
        if not rows:
            return
        batch = pa.RecordBatch.from_pylist(rows, schema=self._schema)
        self.write_batch(batch)

    def write_columns(
        self,
        columns: dict[str, list | np.ndarray | pa.Array],
    ):
        """
        Write columnar data to cache.

        Parameters
        ----------
        columns : dict[str, list | np.ndarray | pa.Array]
            A mapping of columnar field names to list of values.
        """
        if not columns:
            return
        batch = pa.RecordBatch.from_pydict(columns)
        self.write_batch(batch)

    def write_batch(
        self,
        batch: pa.RecordBatch,
    ):
        """
        Write a batch to cache.

        Parameters
        ----------
        batch : pa.RecordBatch
            A batch of columnar data.
        """
        size = batch.num_rows
        if self._buffer:
            size += sum([b.num_rows for b in self._buffer])

        # check size
        if size < self.batch_size and self._count < self.rows_per_file:
            self._buffer.append(batch)
            return

        if self._buffer:
            self._buffer.append(batch)
            batch = pa.concat_batches(self._buffer)
            self._buffer = []

        # write batch
        writer = self._get_or_create_writer()
        writer.write_batch(batch)

        # check file size
        self._count += size
        if self._count >= self.rows_per_file:
            self.flush()

    def write_table(
        self,
        table: pa.Table,
    ):
        """
        Write a table directly to cache.

        Parameters
        ----------
        table : pa.Table
            A populated table.
        """
        self.flush()
        pq.write_table(table, where=self._generate_next_filename())

    def flush(self):
        """Flush the cache buffer."""
        if self._buffer:
            combined_arrays = [
                pa.concat_arrays([b.column(name) for b in self._buffer])
                for name in self._schema.names
            ]
            batch = pa.RecordBatch.from_arrays(
                combined_arrays, schema=self._schema
            )
            writer = self._get_or_create_writer()
            writer.write_batch(batch)
        self._buffer = []
        self._count = 0
        self._close_writer()

    def sort_by(
        self,
        sorting: list[tuple[str, str]],
    ):
        """
        Sort cache files locally and in-place.

        Parameters
        ----------
        sorting : list[tuple[str, str]]
            Sorting arguments in PyArrow format (e.g. [('a', 'ascending'), ('b', 'descending')]).
        """
        self.flush()
        for file in self.get_dataset_files():
            pf = pq.ParquetFile(file)
            tbl = pf.read()
            pf.close()
            sorted_tbl = tbl.sort_by(sorting)
            pq.write_table(sorted_tbl, file)

    def _generate_next_filename(self) -> Path:
        """Generates next dataset filepath."""
        files = self.get_dataset_files()
        if not files:
            next_index = 0
        else:
            next_index = max([int(Path(f).stem) for f in files]) + 1
        return self._path / f"{next_index:06d}.parquet"

    def _get_or_create_writer(self) -> pq.ParquetWriter:
        """Open a new parquet file for writing."""
        if self._writer is not None:
            return self._writer
        self._writer = pq.ParquetWriter(
            where=self._generate_next_filename(),
            schema=self._schema,
            compression=self._compression,
        )
        return self._writer

    def _close_writer(self) -> None:
        """Close the current parquet file."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures data is flushed."""
        self.flush()

    def to_reader(self) -> FileCacheReader:
        """Get cache reader."""
        self.flush()
        return FileCacheReader.load(path=self.path)
