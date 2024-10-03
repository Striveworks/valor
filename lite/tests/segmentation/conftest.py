import numpy as np
import pytest
from valor_lite.segmentation import Bitmask, Segmentation, WeightedMask


def _convert_bool_to_weighted(bitmask: Bitmask) -> WeightedMask:
    return WeightedMask(
        mask=bitmask.mask.astype(float),
        label=bitmask.label,
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
    )


def _generate_weighted_mask(
    mask_shape: tuple[int, int],
    annotation_shape: tuple[int, int, int, int],
    label: str,
) -> WeightedMask:
    xmin = annotation_shape[0]
    xmax = annotation_shape[1]
    ymin = annotation_shape[2]
    ymax = annotation_shape[3]
    mask = np.zeros(mask_shape, dtype=np.float64)
    mask[ymin:ymax, xmin:xmax] = 1.0
    return WeightedMask(
        mask=mask,
        label=label,
    )


def _generate_random_boolean_mask(
    mask_shape: tuple[int, int],
    infill: float,
    label: str,
) -> Bitmask:
    mask = np.zeros(mask_shape, dtype=np.bool_)
    mask_size = mask_shape[0] * mask_shape[1]
    n_positive_pixels = int(infill * mask_size)
    indices = np.random.choice(
        mask_size,
        size=n_positive_pixels,
        replace=False,
    )
    mask[indices] = True
    mask = mask.reshape(mask_shape)
    return Bitmask(
        mask=mask,
        label=label,
    )


def _generate_random_weighted_mask(
    mask_shape: tuple[int, int],
    infill: float,
    label: str,
) -> WeightedMask:
    boolean_mask = _generate_random_boolean_mask(
        mask_shape=mask_shape,
        infill=infill,
        label=label,
    ).mask
    weighted_mask = np.random.rand(*mask_shape)
    weighted_mask[~boolean_mask] = 0.0
    return WeightedMask(
        mask=weighted_mask,
        label=label,
    )


@pytest.fixture
def basic_segmentations() -> list[Segmentation]:

    bitmask1 = Bitmask(
        mask=np.array([[True, False], [False, True]]),
        label="v1",
    )
    bitmask2 = Bitmask(
        mask=np.array([[False, False], [True, False]]),
        label="v2",
    )

    weighted_mask_1 = WeightedMask(
        mask=np.array([[1.0, 0.0], [0.0, 0.5]]),
        label="v1",
    )
    weighted_mask_2 = WeightedMask(
        mask=np.array([[0.0, 0.5], [1.0, 0.0]]),
        label="v2",
    )

    return [
        Segmentation(
            uid="uid0",
            groundtruths=[bitmask1, bitmask2],
            predictions=[weighted_mask_1, weighted_mask_2],
        )
    ]


@pytest.fixture
def segmentations_from_boxes() -> list[Segmentation]:

    mask_shape = (900, 300)

    rect1 = (0, 100, 0, 100)
    rect2 = (150, 300, 400, 500)
    rect3 = (50, 150, 0, 100)  # overlaps 50% with rect1
    rect4 = (101, 151, 301, 401)  # overlaps 1 pixel with rect2

    bitmask1 = _generate_boolean_mask(mask_shape, rect1, "v1")
    bitmask2 = _generate_boolean_mask(mask_shape, rect2, "v2")

    weighted_mask1 = _generate_weighted_mask(mask_shape, rect3, "v1")
    weighted_mask2 = _generate_weighted_mask(mask_shape, rect4, "v2")

    return [
        Segmentation(
            uid="uid1",
            groundtruths=[
                bitmask1,
            ],
            predictions=[
                weighted_mask1,
            ],
        ),
        Segmentation(
            uid="uid2",
            groundtruths=[bitmask2],
            predictions=[
                weighted_mask2,
            ],
        ),
    ]


@pytest.fixture
def segmentations_from_other_boxes() -> list[Segmentation]:

    mask_shape = (900, 300)

    rect1 = (0, 100, 0, 100)
    rect2 = (150, 300, 400, 500)
    rect3 = (50, 150, 0, 100)  # overlaps 50% with rect1
    rect4 = (101, 151, 301, 401)  # overlaps 1 pixel with rect2

    bitmask1 = _generate_boolean_mask(mask_shape, rect1, "v1")
    bitmask2 = _generate_boolean_mask(mask_shape, rect2, "v2")

    weighted_mask1 = _generate_weighted_mask(mask_shape, rect3, "v1")
    weighted_mask2 = _generate_weighted_mask(mask_shape, rect4, "v2")

    return [
        Segmentation(
            uid="uid1",
            groundtruths=[
                bitmask1,
            ],
            predictions=[
                weighted_mask1,
            ],
        ),
        Segmentation(
            uid="uid2",
            groundtruths=[bitmask2],
            predictions=[
                weighted_mask2,
            ],
        ),
    ]


@pytest.fixture
def large_random_segmenations() -> list[Segmentation]:

    mask_shape = (5000, 5000)

    bitmask1 = _generate_random_boolean_mask(
        mask_shape, infill=0.9, label="v1"
    )
    bitmask2 = _generate_random_boolean_mask(
        mask_shape, infill=0.1, label="v2"
    )
    bitmask3 = _generate_random_boolean_mask(
        mask_shape, infill=0.5, label="v1"
    )
    bitmask4 = _generate_random_boolean_mask(
        mask_shape, infill=0.5, label="v2"
    )

    weighted_mask1 = _generate_random_weighted_mask(
        mask_shape, infill=0.9, label="v1"
    )
    weighted_mask2 = _generate_random_weighted_mask(
        mask_shape, infill=0.1, label="v2"
    )
    weighted_mask3 = _generate_random_weighted_mask(
        mask_shape, infill=0.5, label="v1"
    )
    weighted_mask4 = _generate_random_weighted_mask(
        mask_shape, infill=0.5, label="v2"
    )

    return [
        Segmentation(
            uid="uid1",
            groundtruths=[bitmask1, bitmask2],
            predictions=[weighted_mask1, weighted_mask2],
        ),
        Segmentation(
            uid="uid2",
            groundtruths=[bitmask3, bitmask4],
            predictions=[weighted_mask3, weighted_mask4],
        ),
    ]
