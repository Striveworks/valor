from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


class DataType(StrEnum):
    INTEGER = "int"
    FLOAT = "float"
    STRING = "string"
    TIMESTAMP = "timestamp"

    def to_type(self):
        match self:
            case DataType.INTEGER:
                return int
            case DataType.FLOAT:
                return float
            case DataType.STRING:
                return str
            case DataType.TIMESTAMP:
                return datetime

    def to_arrow(self):
        match self:
            case DataType.INTEGER:
                return pa.int64()
            case DataType.FLOAT:
                return pa.float64()
            case DataType.STRING:
                return pa.string()
            case DataType.TIMESTAMP:
                return pa.timestamp()


class Cache:
    """
    A class for sequentially writing data to a PyArrow dataset.
    Handles batching and file management automatically.
    """

    def __init__(
        self,
        output_dir: str | Path,
        schema: pa.Schema,
        batch_size: int = 1000,
        rows_per_file: int = 10000,
        compression: str = "snappy",
    ):
        """
        Initialize the Cache.

        Args:
            output_dir: Directory where parquet files will be written
            schema: PyArrow schema for the data
            batch_size: Number of rows to accumulate before writing a batch
            rows_per_file: Maximum number of rows per parquet file
            compression: Parquet compression to use ('snappy', 'gzip', 'brotli', 'lz4', 'zstd', or None)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.schema = schema
        self.batch_size = batch_size
        self.rows_per_file = rows_per_file
        self.compression = compression

        # Internal state
        self._batch_buffer = {field.name: [] for field in schema}
        self._batch_count = 0
        self._current_writer = None
        self._current_file_path = None
        self._rows_in_current_file = 0
        self._file_index = 0
        self._total_rows_written = 0
        self._files_written = []

    def write(self, row: dict[str, Any]) -> None:
        """
        Write a single row to the cache.

        Args:
            row: Dictionary containing the row data
        """
        # Add row to batch buffer
        for field in self.schema:
            self._batch_buffer[field.name].append(row.get(field.name))
        self._batch_count += 1

        # Write batch if buffer is full
        if self._batch_count >= self.batch_size:
            self._flush_batch()

    def write_many(self, rows: list[dict[str, Any]]) -> None:
        """
        Write multiple rows to the cache.

        Args:
            rows: List of dictionaries containing row data
        """
        # Add rows to batch buffer
        for row in rows:            
            for field in self.schema:
                self._batch_buffer[field.name].append(row.get(field.name))
            self._batch_count += 1

        # Write batch if buffer is full
        if self._batch_count >= self.batch_size:
            self._flush_batch()

    def _flush_batch(self) -> None:
        """Flush the current batch buffer to disk."""
        if self._batch_count == 0:
            return

        # Create new file writer if needed
        if self._current_writer is None:
            self._open_new_file()

        # Create RecordBatch from buffer
        arrays = []
        for field in self.schema:
            # Get the field type from schema
            field_type = field.type

            # Create array with proper type
            try:
                arrays.append(
                    pa.array(self._batch_buffer[field.name], type=field_type)
                )
            except (pa.ArrowTypeError, pa.ArrowInvalid) as e:
                # If automatic conversion fails, try without type hint
                # This allows PyArrow to infer the type
                arrays.append(pa.array(self._batch_buffer[field.name]))

        batch = pa.RecordBatch.from_arrays(arrays, schema=self.schema)

        # Write batch to current file
        self._current_writer.write_batch(batch)

        # Update counters
        self._rows_in_current_file += self._batch_count
        self._total_rows_written += self._batch_count

        # Check if we need to start a new file
        if self._rows_in_current_file >= self.rows_per_file:
            self._close_current_file()

        # Reset batch buffer
        self._batch_buffer = {field.name: [] for field in self.schema}
        self._batch_count = 0

    def _open_new_file(self) -> None:
        """Open a new parquet file for writing."""
        self._current_file_path = (
            self.output_dir / f"{self._file_index:06d}.parquet"
        )
        self._current_writer = pq.ParquetWriter(
            self._current_file_path, self.schema, compression=self.compression
        )
        self._rows_in_current_file = 0
        self._file_index += 1

    def _close_current_file(self) -> None:
        """Close the current parquet file."""
        if self._current_writer is not None:
            self._current_writer.close()
            self._files_written.append(str(self._current_file_path))
            self._current_writer = None
            self._current_file_path = None
            self._rows_in_current_file = 0

    def flush(self) -> None:
        """
        Flush any remaining data in the buffer to disk.
        Call this when done writing to ensure all data is persisted.
        """
        # Flush any remaining data in the batch buffer
        if self._batch_count > 0:
            self._flush_batch()

        # Close current file if open
        if self._current_writer is not None:
            self._close_current_file()

    def clear(self) -> None:
        """
        Clear the cache, removing all written files and resetting state.
        """
        # Flush and close any open files
        self.flush()

        # Remove all written files
        for file_path in self._files_written:
            Path(file_path).unlink(missing_ok=True)

        # Reset state
        self._batch_buffer = {field.name: [] for field in self.schema}
        self._batch_count = 0
        self._current_writer = None
        self._current_file_path = None
        self._rows_in_current_file = 0
        self._file_index = 0
        self._total_rows_written = 0
        self._files_written = []

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures data is flushed."""
        self.flush()

    def __len__(self) -> int:
        """Return the total number of rows written."""
        return self._total_rows_written + self._batch_count

    @property
    def num_files(self) -> int:
        """Return the number of files written."""
        return len(self._files_written)

    @property
    def total_rows(self) -> int:
        """Return the total number of rows (written + buffered)."""
        return self._total_rows_written + self._batch_count
    
    def get_dataset(self) -> ds.Dataset:
        """
        Get a PyArrow Dataset from the cached files.

        Returns:
            PyArrow Dataset object
        """
        return ds.dataset(
            str(self.output_dir), format="parquet", schema=self.schema
        )

    def get_table(self) -> pa.Table:
        """
        Get a PyArrow Table from the cached files.
        Note: This loads all data into memory.

        Returns:
            PyArrow Table object
        """
        dataset = self.get_dataset()
        return dataset.to_table()

    def get_info(self) -> dict[str, Any]:
        """
        Get information about the cache state.

        Returns:
            Dictionary with cache statistics
        """
        return {
            "output_dir": str(self.output_dir),
            "total_rows": self.total_rows,
            "rows_written": self._total_rows_written,
            "rows_buffered": self._batch_count,
            "num_files": self.num_files,
            "files": self._files_written,
            "batch_size": self.batch_size,
            "rows_per_file": self.rows_per_file,
            "compression": self.compression,
        }


# Example usage
if __name__ == "__main__":
    # Define schema with complex types
    annotation_struct = pa.struct(
        [("id", pa.string()), ("name", pa.string()), ("value", pa.float64())]
    )
    annotation_list_type = pa.list_(annotation_struct)

    schema = pa.schema(
        [
            ("id", pa.string()),
            ("name", pa.string()),
            ("age", pa.int32()),
            ("annotations", annotation_list_type),
        ]
    )

    # Create cache instance
    cache = Cache(
        output_dir="my_dataset",
        schema=schema,
        batch_size=100,
        rows_per_file=500,
        compression="snappy",
    )

    # Example 1: Write rows one by one in a loop
    print("Writing rows one by one...")
    for i in range(1000):
        row = {
            "id": f"row_{i}",
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"][i % 5],
            "age": 20 + (i % 50),
            "annotations": [
                {
                    "id": "a1",
                    "name": "height",
                    "value": 1.60 + (i % 30) * 0.01,
                },
                {"id": "a2", "name": "weight", "value": 60.0 + (i % 20)},
            ],
        }
        cache.write(row)

    # Finalize to ensure all data is written
    cache.flush()

    print(f"\nCache info after writing:")
    for key, value in cache.get_info().items():
        if key != "files":
            print(f"  {key}: {value}")

    # Get dataset
    dataset = cache.get_dataset()
    print(f"\nDataset created with {dataset.count_rows()} rows")
    print(f"Dataset files: {dataset.files[:3]}...")  # Show first 3 files

    # Example 2: Using context manager
    print("\n" + "=" * 50)
    print("Using context manager:")

    with Cache(
        "my_dataset2", schema, batch_size=50, rows_per_file=200
    ) as cache2:
        for i in range(500):
            cache2.write(
                {
                    "id": f"ctx_row_{i}",
                    "name": f"Person_{i}",
                    "age": 25 + (i % 40),
                    "annotations": [
                        {"id": f"a_{i}", "name": "score", "value": float(i)}
                    ],
                }
            )

        print(f"Rows written: {len(cache2)}")

    # Data is automatically flushed when exiting context
    dataset2 = ds.dataset("my_dataset2", format="parquet", schema=schema)
    print(f"Dataset 2 has {dataset2.count_rows()} rows")

    # Example 3: Write many rows at once
    print("\n" + "=" * 50)
    print("Writing many rows at once:")

    cache3 = Cache("my_dataset3", schema, batch_size=100)

    # Generate batch of rows
    batch_rows = [
        {
            "id": f"batch_row_{i}",
            "name": f"BatchPerson_{i}",
            "age": 30 + (i % 35),
            "annotations": [],
        }
        for i in range(250)
    ]

    cache3.write_many(batch_rows)
    cache3.flush()

    print(f"Cache 3 info:")
    print(f"  Total rows: {cache3.total_rows}")
    print(f"  Files written: {cache3.num_files}")

    # Read back some data
    table = cache3.get_table()
    print(f"\nFirst 5 rows from cache 3:")
    print(table.slice(0, 5).to_pandas())
