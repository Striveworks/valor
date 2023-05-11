import pytest

from velour_api import schemas


@pytest.fixture
def img() -> schemas.Image:
    return schemas.Image(uid="", height=1098, width=4591)


@pytest.fixture
def cm() -> schemas.ConfusionMatrix:
    return schemas.ConfusionMatrix(
        label_key="class",
        entries=[
            schemas.ConfusionMatrixEntry(
                groundtruth="class0", prediction="class0", count=1
            ),
            schemas.ConfusionMatrixEntry(
                groundtruth="class0", prediction="class1", count=1
            ),
            schemas.ConfusionMatrixEntry(
                groundtruth="class0", prediction="class2", count=1
            ),
            schemas.ConfusionMatrixEntry(
                groundtruth="class1", prediction="class1", count=1
            ),
            schemas.ConfusionMatrixEntry(
                groundtruth="class2", prediction="class1", count=2
            ),
        ],
    )
