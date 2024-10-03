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
    mask[ymin : ymax + 1, xmin : xmax + 1] = True
    return Bitmask(
        mask=mask,
        label=label,
    )


def _generate_weighted_mask(
    mask_shape: tuple[int, int],
    annotation_shape: tuple[int, int, int, int],
    label: str,
) -> WeightedMask:
    mask = np.random.rand(*mask_shape)
    xmin = annotation_shape[0]
    xmax = annotation_shape[1]
    ymin = annotation_shape[2]
    ymax = annotation_shape[3]
    mask[ymin : ymax + 1, xmin : xmax + 1] = 0.0
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
    weighted_mask[boolean_mask] = 0.0
    return WeightedMask(
        mask=weighted_mask,
        label=label,
    )


@pytest.fixture
def rect1() -> tuple[float, float, float, float]:
    """Box with area = 1500."""
    return (10, 60, 10, 40)


@pytest.fixture
def rect2() -> tuple[float, float, float, float]:
    """Box with area = 1100."""
    return (15, 70, 0, 20)


@pytest.fixture
def rect3() -> tuple[float, float, float, float]:
    """Box with area = 57,510."""
    return (87, 158, 10, 820)


@pytest.fixture
def rect4() -> tuple[float, float, float, float]:
    """Box with area = 90."""
    return (1, 10, 10, 20)


@pytest.fixture
def rect5() -> tuple[float, float, float, float]:
    """Box with partial overlap to rect3."""
    return (87, 158, 10, 400)


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
def segmentations_from_boxes(
    rect1: tuple[int, int, int, int],
    rect2: tuple[int, int, int, int],
    rect3: tuple[int, int, int, int],
    rect5: tuple[int, int, int, int],
) -> list[Segmentation]:
    mask_shape = (900, 300)

    bitmask1 = _generate_boolean_mask(mask_shape, rect1, "v1")
    bitmask2 = _generate_boolean_mask(mask_shape, rect2, "v1")
    bitmask3 = _generate_boolean_mask(mask_shape, rect3, "v2")
    bitmask5 = _generate_boolean_mask(mask_shape, rect5, "v2")

    weighted_mask1 = _convert_bool_to_weighted(bitmask1)
    weighted_mask2 = _convert_bool_to_weighted(bitmask2)
    weighted_mask5 = _convert_bool_to_weighted(bitmask5)

    return [
        Segmentation(
            uid="uid1",
            groundtruths=[
                bitmask1,
                bitmask3,
            ],
            predictions=[
                weighted_mask1,
                weighted_mask5,
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
