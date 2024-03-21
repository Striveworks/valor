import math

import pytest

from valor_api import enums, schemas


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
def metadata() -> dict[str, str | float]:
    return {
        "m1": "v1",
        "m2": 0.1,
    }


@pytest.fixture
def box_points() -> list[schemas.geometry.Point]:
    return [
        schemas.geometry.Point(x=-5, y=-5),
        schemas.geometry.Point(x=5, y=-5),
        schemas.geometry.Point(x=5, y=5),
        schemas.geometry.Point(x=-5, y=5),
    ]


@pytest.fixture
def rotated_box_points() -> list[schemas.geometry.Point]:
    """Same area and sides as box_points, but rotated 45 degrees."""
    d = 5.0 * math.sqrt(2)
    return [
        schemas.geometry.Point(x=0, y=d),
        schemas.geometry.Point(x=d, y=0),
        schemas.geometry.Point(x=0, y=-d),
        schemas.geometry.Point(x=-d, y=0),
    ]


@pytest.fixture
def skewed_box_points() -> list[schemas.geometry.Point]:
    """Skewed box_points."""
    return [
        schemas.geometry.Point(x=0, y=0),
        schemas.geometry.Point(x=10, y=0),
        schemas.geometry.Point(x=15, y=10),
        schemas.geometry.Point(x=5, y=10),
    ]


@pytest.fixture
def component_polygon_box(box_points) -> schemas.geometry.BasicPolygon:
    return schemas.geometry.BasicPolygon(
        points=box_points,
    )


@pytest.fixture
def component_polygon_rotated_box(
    rotated_box_points,
) -> schemas.geometry.BasicPolygon:
    return schemas.geometry.BasicPolygon(
        points=rotated_box_points,
    )


@pytest.fixture
def component_polygon_skewed_box(
    skewed_box_points,
) -> schemas.geometry.BasicPolygon:
    return schemas.geometry.BasicPolygon(
        points=skewed_box_points,
    )


@pytest.fixture
def bbox(component_polygon_box) -> schemas.BoundingBox:
    return schemas.BoundingBox(
        polygon=component_polygon_box,
    )


@pytest.fixture
def polygon(component_polygon_box) -> schemas.Polygon:
    return schemas.Polygon(boundary=component_polygon_box)


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
            labels=[labels[0]], task_type=enums.TaskType.CLASSIFICATION
        ),
        schemas.Annotation(
            labels=[labels[2]], task_type=enums.TaskType.CLASSIFICATION
        ),
        schemas.Annotation(
            labels=[labels[3]],
            task_type=enums.TaskType.CLASSIFICATION,
        ),
    ]


@pytest.fixture
def predicted_annotations(scored_labels) -> list[schemas.Annotation]:
    return [
        schemas.Annotation(
            labels=[scored_labels[0], scored_labels[1]],
            task_type=enums.TaskType.CLASSIFICATION,
        ),
        schemas.Annotation(
            labels=[scored_labels[2]], task_type=enums.TaskType.CLASSIFICATION
        ),
        schemas.Annotation(
            labels=[scored_labels[3]], task_type=enums.TaskType.CLASSIFICATION
        ),
    ]
