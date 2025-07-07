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

    # wrong annotation type
    with pytest.raises(ValueError):
        Segmentation(
            uid="uid",
            groundtruths=[{"a": 1}],  # type: ignore - testing
            predictions=[],
            shape=(1, 2),
        )
    with pytest.raises(ValueError):
        Segmentation(
            uid="uid",
            groundtruths=[],
            predictions=[{"a": 1}],  # type: ignore - testing
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


def _create_overlapped_masks() -> tuple[Bitmask, Bitmask]:
    mask0 = np.zeros((100, 100), dtype=np.bool_)
    mask0[:50, :] = True
    mask1 = np.ones((100, 100), dtype=np.bool_)
    bitmask0 = Bitmask(mask=mask0, label="dog")
    bitmask1 = Bitmask(mask=mask1, label="cat")
    return bitmask0, bitmask1


def test_segmentations_overlap():
    bitmask0, bitmask1 = _create_overlapped_masks()
    Segmentation(
        uid="uid123",
        groundtruths=[bitmask0],
        predictions=[bitmask1],
        shape=(100, 100),
    )
    assert bitmask0.mask.sum() == 5000
    assert bitmask1.mask.sum() == 10000

    bitmask0, bitmask1 = _create_overlapped_masks()
    with pytest.warns(UserWarning) as e:
        Segmentation(
            uid="uid123",
            groundtruths=[bitmask0, bitmask1],
            predictions=[],
            shape=(100, 100),
        )
    assert (
        str(e._list[0].message)
        == "ground truth masks for datum 'uid123' had 5000 / 10000 pixels overlapped."
    )
    assert bitmask0.mask.sum() == 5000
    assert (
        bitmask1.mask.sum() == 5000
    )  # overlapped pixels omitted from second mask

    bitmask0, bitmask1 = _create_overlapped_masks()
    with pytest.warns(UserWarning) as e:
        Segmentation(
            uid="uid123",
            groundtruths=[],
            predictions=[bitmask0, bitmask1],
            shape=(100, 100),
        )
    assert (
        str(e._list[0].message)
        == "prediction masks for datum 'uid123' had 5000 / 10000 pixels overlapped."
    )
    assert bitmask0.mask.sum() == 5000
    assert (
        bitmask1.mask.sum() == 5000
    )  # overlapped pixels omitted from second mask
