import numpy as np
import pytest
from valor_lite.semantic_segmentation import (
    Bitmask,
    Segmentation,
    generate_segmentation,
)


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


def test_generate_segmentation():

    # N labels > 1
    segmentation = generate_segmentation(
        datum_uid="uid1",
        number_of_unique_labels=3,
        mask_height=2,
        mask_width=3,
    )

    assert segmentation.uid == "uid1"
    assert segmentation.shape == (2, 3)
    assert segmentation.size == 6

    assert len(segmentation.groundtruths) == 3
    assert all(gt.mask.dtype == np.bool_ for gt in segmentation.groundtruths)
    assert all(gt.mask.shape == (2, 3) for gt in segmentation.groundtruths)

    assert len(segmentation.predictions) == 3
    assert all(pd.mask.dtype == np.bool_ for pd in segmentation.predictions)
    assert all(pd.mask.shape == (2, 3) for pd in segmentation.predictions)

    # N labels = 1
    segmentation = generate_segmentation(
        datum_uid="uid1",
        number_of_unique_labels=1,
        mask_height=2,
        mask_width=3,
    )

    assert segmentation.uid == "uid1"
    assert segmentation.shape == (2, 3)
    assert segmentation.size == 6

    assert len(segmentation.groundtruths) == 1
    assert all(gt.mask.dtype == np.bool_ for gt in segmentation.groundtruths)
    assert all(gt.mask.shape == (2, 3) for gt in segmentation.groundtruths)

    assert len(segmentation.predictions) == 1
    assert all(pd.mask.dtype == np.bool_ for pd in segmentation.predictions)
    assert all(pd.mask.shape == (2, 3) for pd in segmentation.predictions)

    # N labels = 0
    with pytest.raises(ValueError):
        generate_segmentation(
            datum_uid="uid1",
            number_of_unique_labels=0,
            mask_height=2,
            mask_width=3,
        )
