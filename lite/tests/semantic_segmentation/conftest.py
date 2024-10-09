import numpy as np
import pytest
from valor_lite.semantic_segmentation import Bitmask, Segmentation


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
    )


def _generate_random_boolean_mask(
    mask_shape: tuple[int, int],
    infill: float,
    label: str,
) -> Bitmask:
    mask_size = mask_shape[0] * mask_shape[1]
    mask = np.zeros(mask_size, dtype=np.bool_)

    n_positives = int(mask_size * infill)
    mask[:n_positives] = True
    np.random.shuffle(mask)
    mask = mask.reshape(mask_shape)

    return Bitmask(
        mask=mask,
        label=label,
    )


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
        ),
        Segmentation(
            uid="uid2",
            groundtruths=[gmask2],
            predictions=[pmask2],
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

    return [
        Segmentation(
            uid=f"uid{idx}",
            groundtruths=[
                _generate_random_boolean_mask(mask_shape, infill, label)
                for infill, label in zip(infills, labels)
            ],
            predictions=[
                _generate_random_boolean_mask(mask_shape, infill, label)
                for infill, label in zip(infills, labels)
            ],
        )
        for idx, (infills, labels) in enumerate(
            zip(infills_per_seg, labels_per_seg)
        )
    ]
