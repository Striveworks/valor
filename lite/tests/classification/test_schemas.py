import pytest
from valor_lite.classification import Classification


def test_Classification():

    Classification(
        uid="uid",
        groundtruth="v1",
        predictions=["v1", "v2"],
        scores=[0.0, 1.0],
    )

    # test that a groundtruth must be defined
    with pytest.raises(ValueError):
        Classification(
            uid="uid",
            groundtruth=[],  # type: ignore - testing
            predictions=["v1", "v2"],
            scores=[0.0, 1.0],
        )

    # test that predictions must be of the same length as scores
    with pytest.raises(ValueError):
        Classification(
            uid="uid",
            groundtruth="v1",
            predictions=["v1", "v2"],
            scores=[0.0, 1.0, 0.0],
        )
