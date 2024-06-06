import math

import pytest

from valor_api import schemas


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


@pytest.fixture
def metadata() -> dict[str, dict[str, str | float]]:
    return {
        "m1": {"type": "string", "value": "v1"},
        "m2": {"type": "float", "value": 0.1},
    }


@pytest.fixture
def box_points() -> list[tuple[float, float]]:
    return [
        (-5, -5),
        (5, -5),
        (5, 5),
        (-5, 5),
        (-5, -5),
    ]


@pytest.fixture
def rotated_box_points() -> list[tuple[float, float]]:
    """Same area and sides as box_points, but rotated 45 degrees."""
    d = 5.0 * math.sqrt(2)
    return [
        (0, d),
        (d, 0),
        (0, -d),
        (-d, 0),
        (0, d),
    ]


@pytest.fixture
def skewed_box_points() -> list[tuple[float, float]]:
    """Skewed box_points."""
    return [
        (0, 0),
        (10, 0),
        (15, 10),
        (5, 10),
        (0, 0),
    ]


@pytest.fixture
def bbox(box_points) -> schemas.Box:
    return schemas.Box(value=[box_points])


@pytest.fixture
def polygon(box_points) -> schemas.Polygon:
    return schemas.Polygon(value=[box_points])


@pytest.fixture
def raster() -> schemas.Raster:
    """
    Creates a 2d numpy of bools of shape:
    | T  F |
    | F  T |
    """
    mask = "iVBORw0KGgoAAAANSUhEUgAAABQAAAAUAQAAAACl8iCgAAAAF0lEQVR4nGP4f4CBiYGBIGZgsP9AjDoAuysDE0GVDN8AAAAASUVORK5CYII="
    return schemas.Raster(mask=mask)


@pytest.fixture
def labels() -> list[schemas.Label]:
    return [
        schemas.Label(key="k1", value="v1"),
        schemas.Label(key="k1", value="v2"),
        schemas.Label(key="k2", value="v3"),
        schemas.Label(key="k3", value="v4"),
    ]


@pytest.fixture
def scored_labels(labels) -> list[schemas.Label]:
    ret = [la.model_copy() for la in labels]
    for la, score in zip(ret, [0.1, 0.9, 1.0, 1.0]):
        la.score = score

    return ret


@pytest.fixture
def groundtruth_annotations(labels) -> list[schemas.Annotation]:
    return [
        schemas.Annotation(
            labels=[labels[0]],
        ),
        schemas.Annotation(
            labels=[labels[2]],
        ),
        schemas.Annotation(
            labels=[labels[3]],
        ),
    ]


@pytest.fixture
def predicted_annotations(scored_labels) -> list[schemas.Annotation]:
    return [
        schemas.Annotation(
            labels=[scored_labels[0], scored_labels[1]],
        ),
        schemas.Annotation(
            labels=[scored_labels[2]],
        ),
        schemas.Annotation(
            labels=[scored_labels[3]],
        ),
    ]
