from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from valor_lite.semantic_segmentation import Bitmask, Loader, Segmentation


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


def _generate_boolean_mask(
    mask_shape: tuple[int, int],
    annotation_shape: tuple[int, int, int, int],
    label: str,
) -> Bitmask:
    mask = np.zeros(mask_shape, dtype=np.bool_)
    xmin = annotation_shape[0]
    xmax = annotation_shape[1]
    ymin = annotation_shape[2]
    ymax = annotation_shape[3]
    mask[ymin:ymax, xmin:xmax] = True
    return Bitmask(
        mask=mask,
        label=label,
        metadata={
            "gt_xmin": xmin,
            "pd_xmin": xmin,
        },
    )


def _generate_random_boolean_mask(
    mask_shape: tuple[int, int],
    infill: float,
) -> np.ndarray:
    mask_size = mask_shape[0] * mask_shape[1]
    mask = np.zeros(mask_size, dtype=np.bool_)

    n_positives = int(mask_size * infill)
    mask[:n_positives] = True
    np.random.shuffle(mask)
    mask = mask.reshape(mask_shape)
    return mask


@pytest.fixture
def basic_segmentations() -> list[Segmentation]:

    gmask1 = Bitmask(
        mask=np.array([[True, False], [False, True]]),
        label="v1",
    )
    gmask2 = Bitmask(
        mask=np.array([[False, False], [True, False]]),
        label="v2",
    )

    pmask1 = Bitmask(
        mask=np.array([[True, False], [False, False]]),
        label="v1",
    )
    pmask2 = Bitmask(
        mask=np.array([[False, True], [True, False]]),
        label="v2",
    )

    return [
        Segmentation(
            uid="uid0",
            groundtruths=[gmask1, gmask2],
            predictions=[pmask1, pmask2],
            shape=tuple(pmask1.mask.shape),
        )
    ]


@pytest.fixture
def segmentations_from_boxes() -> list[Segmentation]:

    mask_shape = (900, 300)

    rect1 = (0, 100, 0, 100)
    rect2 = (150, 300, 400, 500)
    rect3 = (50, 150, 0, 100)  # overlaps 50% with rect1
    rect4 = (101, 151, 301, 401)  # overlaps 1 pixel with rect2

    gmask1 = _generate_boolean_mask(mask_shape, rect1, "v1")
    gmask2 = _generate_boolean_mask(mask_shape, rect2, "v2")

    pmask1 = _generate_boolean_mask(mask_shape, rect3, "v1")
    pmask2 = _generate_boolean_mask(mask_shape, rect4, "v2")

    return [
        Segmentation(
            uid="uid1",
            groundtruths=[gmask1],
            predictions=[pmask1],
            shape=tuple(gmask1.mask.shape),
        ),
        Segmentation(
            uid="uid2",
            groundtruths=[gmask2],
            predictions=[pmask2],
            shape=tuple(pmask2.mask.shape),
        ),
    ]


@pytest.fixture
def large_random_segmentations() -> list[Segmentation]:

    mask_shape = (2000, 2000)
    infills_per_seg = [
        (0.9, 0.09, 0.01),
        (0.4, 0.4, 0.1),
        (0.3, 0.3, 0.3),
    ]
    labels_per_seg = [
        ("v1", "v2", "v3"),
        ("v4", "v5", "v6"),
        ("v7", "v8", "v9"),
    ]

    gt_bitmasks_per_datum = []
    pd_bitmasks_per_datum = []
    for infills in infills_per_seg:
        gt_accum = None
        pd_accum = None
        gt_masks = []
        pd_masks = []
        for infill in infills:
            gt = _generate_random_boolean_mask(mask_shape, infill)
            if gt_accum is None:
                gt_accum = gt.copy()
            else:
                gt = gt & ~(gt & gt_accum)
                gt_accum = gt_accum | gt
            gt_masks.append(gt)

            pd = _generate_random_boolean_mask(mask_shape, infill)
            if pd_accum is None:
                pd_accum = pd.copy()
            else:
                pd = pd & ~(pd & pd_accum)
                pd_accum = pd_accum | pd
            pd_masks.append(pd)
        gt_bitmasks_per_datum.append(gt_masks)
        pd_bitmasks_per_datum.append(pd_masks)

    return [
        Segmentation(
            uid=f"uid{idx}",
            groundtruths=[
                Bitmask(
                    mask=mask,
                    label=label,
                )
                for mask, label in zip(gts, labels)
            ],
            predictions=[
                Bitmask(
                    mask=mask,
                    label=label,
                )
                for mask, label in zip(pds, labels)
            ],
            shape=mask_shape,
        )
        for idx, (gts, pds, labels) in enumerate(
            zip(gt_bitmasks_per_datum, pd_bitmasks_per_datum, labels_per_seg)
        )
    ]
