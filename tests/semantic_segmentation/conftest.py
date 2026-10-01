from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from valor_lite.semantic_segmentation import Loader, Segmentation


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
def loader(request, tmp_path: Path):
    file_type, batch_size, rows_per_file = request.param
    match file_type:
        case "memory":
            return Loader.in_memory(
                batch_size=batch_size,
                metadata_fields=[
                    ("gt_xmin", "float64"),
                    ("pd_xmin", pa.float64()),
                ],
            )
        case "persistent":
            return Loader.persistent(
                path=tmp_path / "cache",
                batch_size=batch_size,
                rows_per_file=rows_per_file,
                metadata_fields=[
                    ("gt_xmin", "float64"),
                    ("pd_xmin", pa.float64()),
                ],
            )


@pytest.fixture
def basic_segmentations() -> list[Segmentation]:
    return [
        Segmentation(
            uid="uid0",
            groundtruths=np.array([[1, 0], [2, 1]], dtype=np.uint16),
            predictions=np.array([[1, 2], [2, 0]], dtype=np.uint16),
            labels=["v1", "v2"],
        )
    ]


@pytest.fixture
def basic_segmentations_three_labels() -> list[Segmentation]:
    return [
        Segmentation(
            uid=f"uid{i}",
            groundtruths=np.array([[1, 3], [2, 1]], dtype=np.uint16),
            predictions=np.array([[1, 2], [2, 3]], dtype=np.uint16),
            labels=["v1", "v2", "v3"],
        )
        for i in range(3)
    ]


@pytest.fixture
def segmentations_from_boxes() -> list[Segmentation]:
    def label_map(rect, label):
        array = np.zeros((900, 300), dtype=np.uint16)
        xmin, xmax, ymin, ymax = rect
        array[ymin:ymax, xmin:xmax] = label
        return array

    rectangles = [
        ((0, 100, 0, 100), (50, 150, 0, 100)),
        ((150, 300, 400, 500), (101, 151, 301, 401)),
    ]
    return [
        Segmentation(
            uid=f"uid{i + 1}",
            groundtruths=label_map(gt, i + 1),
            predictions=label_map(pd, i + 1),
            labels=["v1", "v2"],
            groundtruth_metadata={i + 1: {"gt_xmin": gt[0]}},
            prediction_metadata={i + 1: {"pd_xmin": pd[0]}},
        )
        for i, (gt, pd) in enumerate(rectangles)
    ]


@pytest.fixture
def large_random_segmentations() -> list[Segmentation]:
    rng = np.random.default_rng(42)
    return [
        Segmentation(
            uid=f"uid{i}",
            groundtruths=rng.integers(
                1, 3, size=(2000, 2000), dtype=np.uint16
            ),
            predictions=rng.integers(1, 3, size=(2000, 2000), dtype=np.uint16),
            labels=[f"class-{2 * i}", f"class-{2 * i + 1}"],
        )
        for i in range(3)
    ]
