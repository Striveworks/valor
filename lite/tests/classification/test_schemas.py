import pytest
from valor_lite.classification import Classification


def test_Classification():

    Classification(
        uid="uid",
        groundtruths=[("k1", "v1")],
        predictions=[("k1", "v1"), ("k1", "v1")],
        scores=[0.0, 1.0],
    )

    # test that predictions must be of the same length as scores
    with pytest.raises(ValueError):
        Classification(
            uid="uid",
            groundtruths=[("k1", "v1")],
            predictions=[("k1", "v1"), ("k1", "v1")],
            scores=[0.0, 1.0, 0.0],
        )
