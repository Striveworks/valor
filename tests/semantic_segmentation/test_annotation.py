import numpy as np
import pytest

from valor_lite.semantic_segmentation import Bitmask, Segmentation


def test_bitmask():

    Bitmask(
        mask=np.array([[True, False]]),
        label="label",
    )

    with pytest.raises(ValueError) as e:
        Bitmask(mask=np.array([[1, 2]]), label="label")
    assert "int64" in str(e)


def test_segmentation():

    s = Segmentation(
        uid="uid",
        groundtruths=[
            Bitmask(
                mask=np.array([[True, False]]),
                label="label",
            )
        ],
        predictions=[
            Bitmask(
                mask=np.array([[True, False]]),
                label="label",
            )
        ],
        shape=(1, 2),
    )
    assert s.shape == (1, 2)
    assert s.size == 2

    # test shape mismatch within ground truths
    with pytest.raises(ValueError) as e:
        Segmentation(
            uid="uid",
            groundtruths=[
                Bitmask(
                    mask=np.array([[True, False, False]]),
                    label="label1",
                ),
                Bitmask(
                    mask=np.array([[False, False]]),
                    label="label2",
                ),
            ],
            predictions=[
                Bitmask(
                    mask=np.array([[True, False]]),
                    label="label",
                )
            ],
            shape=(1, 3),
        )
    assert "Received mask with shape '(1, 2)'" in str(e)

    # test shape mismatch within predictions
    with pytest.raises(ValueError) as e:
        Segmentation(
            uid="uid",
            groundtruths=[
                Bitmask(
                    mask=np.array([[True, False, False]]),
                    label="label1",
                ),
            ],
            predictions=[
                Bitmask(
                    mask=np.array([[True, False, False]]),
                    label="label",
                ),
                Bitmask(
                    mask=np.array([[False, False]]),
                    label="label2",
                ),
            ],
            shape=(1, 3),
        )
    assert "Received mask with shape '(1, 2)'" in str(e)

    # test shape mismatch between ground truths and predictions
    with pytest.raises(ValueError) as e:
        Segmentation(
            uid="uid",
            groundtruths=[
                Bitmask(
                    mask=np.array([[True, False, False]]),
                    label="label",
                )
            ],
            predictions=[
                Bitmask(
                    mask=np.array([[True, False]]),
                    label="label",
                )
            ],
            shape=(1, 3),
        )
    assert "Received mask with shape '(1, 2)'" in str(e)

    # test ground truths cannot overlap
    with pytest.raises(ValueError) as e:
        Segmentation(
            uid="uid",
            groundtruths=[
                Bitmask(
                    mask=np.array([[True, True, True]]),
                    label="label1",
                ),
                Bitmask(
                    mask=np.array([[False, False, True]]),
                    label="label2",
                ),
            ],
            predictions=[
                Bitmask(
                    mask=np.array([[True, False, False]]),
                    label="label",
                )
            ],
            shape=(1, 3),
        )
    assert "ground truth masks cannot overlap" in str(e)

    # test predictions cannot overlap
    with pytest.raises(ValueError) as e:
        Segmentation(
            uid="uid",
            groundtruths=[
                Bitmask(
                    mask=np.array([[True, True, True]]),
                    label="label1",
                ),
            ],
            predictions=[
                Bitmask(
                    mask=np.array([[True, False, True]]),
                    label="label",
                ),
                Bitmask(
                    mask=np.array([[False, False, True]]),
                    label="label2",
                ),
            ],
            shape=(1, 3),
        )
    assert "prediction masks cannot overlap" in str(e)

    # allow missing ground truths
    Segmentation(
        uid="uid",
        groundtruths=[],
        predictions=[
            Bitmask(
                mask=np.array([[True, False]]),
                label="label",
            )
        ],
        shape=(1, 2),
    )

    # allow missing predictions
    Segmentation(
        uid="uid",
        groundtruths=[
            Bitmask(
                mask=np.array([[True, False]]),
                label="label",
            )
        ],
        predictions=[],
        shape=(1, 2),
    )


def test_segmentation_shape():
    Segmentation(uid="uid", groundtruths=[], predictions=[], shape=(1, 1))
    Segmentation(uid="uid", groundtruths=[], predictions=[], shape=(100, 100))
    with pytest.raises(ValueError):
        Segmentation(uid="uid", groundtruths=[], predictions=[], shape=(1,))
    with pytest.raises(ValueError):
        Segmentation(
            uid="uid", groundtruths=[], predictions=[], shape=(1, 2, 3)
        )
    with pytest.raises(ValueError):
        Segmentation(
            uid="uid", groundtruths=[], predictions=[], shape=(0, 100)
        )
    with pytest.raises(ValueError):
        Segmentation(
            uid="uid", groundtruths=[], predictions=[], shape=(-100, 100)
        )
