import numpy as np
import pytest
from valor_lite.segmentation import Bitmask, Segmentation


def test_bitmask():

    Bitmask(
        mask=np.array([True, False]),
        label="label",
    )

    with pytest.raises(ValueError) as e:
        Bitmask(mask=np.array([1, 2]), label="label")
    assert "int64" in str(e)


def test_segmentation():

    s = Segmentation(
        uid="uid",
        groundtruths=[
            Bitmask(
                mask=np.array([True, False]),
                label="label",
            )
        ],
        predictions=[
            Bitmask(
                mask=np.array([True, False]),
                label="label",
            )
        ],
    )
    assert s.shape == (2,)
    assert s.size == 2

    with pytest.raises(ValueError) as e:
        Segmentation(
            uid="uid",
            groundtruths=[
                Bitmask(
                    mask=np.array([True, False, False]),
                    label="label",
                )
            ],
            predictions=[
                Bitmask(
                    mask=np.array([True, False]),
                    label="label",
                )
            ],
        )
    assert "mismatch" in str(e)

    with pytest.raises(ValueError) as e:
        Segmentation(
            uid="uid",
            groundtruths=[],
            predictions=[
                Bitmask(
                    mask=np.array([True, False]),
                    label="label",
                )
            ],
        )
    assert "missing ground truths" in str(e)

    with pytest.raises(ValueError) as e:
        Segmentation(
            uid="uid",
            groundtruths=[
                Bitmask(
                    mask=np.array([True, False]),
                    label="label",
                )
            ],
            predictions=[],
        )
    assert "missing predictions" in str(e)
