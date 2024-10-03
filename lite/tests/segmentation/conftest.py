import numpy as np
import pytest
from numpy.typing import NDArray
from valor_lite.segmentation import Bitmask, Segmentation, WeightedMask


def _generate_boolean_mask(
    mask_height: int,
    mask_width: int,
    infill: float,
) -> NDArray[np.bool_]:

    mask = np.zeros((mask_height * mask_width), dtype=np.bool_)

    n_positive_pixels = int(infill * mask_height * mask_width)
    indices = np.random.choice(
        mask_height * mask_width,
        size=n_positive_pixels,
        replace=False,
    )
    mask[indices] = True

    return mask.reshape((mask_height, mask_width))


def _generate_weighted_mask(
    mask_height: int,
    mask_width: int,
    infill: float,
) -> NDArray[np.floating]:
    boolean_mask = _generate_boolean_mask(
        mask_height=mask_height,
        mask_width=mask_width,
        infill=infill,
    )
    weighted_mask = np.random.rand(mask_height, mask_width)
    weighted_mask[boolean_mask] = 0.0
    return weighted_mask


@pytest.fixture
def bitmask_1() -> Bitmask:
    return Bitmask(mask=np.array([]), label="v1")


@pytest.fixture
def bitmask_2() -> Bitmask:
    return Bitmask(mask=np.array([]), label="v2")


@pytest.fixture
def bitmask_3() -> Bitmask:
    return Bitmask(mask=np.array([]), label="v3")


@pytest.fixture
def weighted_mask_1() -> WeightedMask:
    return WeightedMask(mask=np.array([]), label="v1")


@pytest.fixture
def weighted_mask_2() -> WeightedMask:
    return WeightedMask(mask=np.array([]), label="v2")


@pytest.fixture
def weighted_mask_3() -> WeightedMask:
    return WeightedMask(mask=np.array([]), label="v3")


@pytest.fixture
def basic_segmentations() -> list[Segmentation]:
    return [
        # Segmentation(uid="uid1", groundtruths=[]),
        # Segmentation(uid="uid2"),
    ]
