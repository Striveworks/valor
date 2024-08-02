import pandas as pd
import pytest
from valor_core import geometry, schemas


@pytest.fixture
def rect1() -> list[tuple[float, float]]:
    """Box with area = 1500."""
    return [
        (10, 10),
        (60, 10),
        (60, 40),
        (10, 40),
        (10, 10),
    ]


@pytest.fixture
def rect2() -> list[tuple[float, float]]:
    """Box with area = 1100."""
    return [
        (15, 0),
        (70, 0),
        (70, 20),
        (15, 20),
        (15, 0),
    ]


@pytest.fixture
def rect3() -> list[tuple[float, float]]:
    """Box with area = 57,510."""
    return [
        (87, 10),
        (158, 10),
        (158, 820),
        (87, 820),
        (87, 10),
    ]


@pytest.fixture
def rect4() -> list[tuple[float, float]]:
    """Box with area = 90."""
    return [
        (1, 10),
        (10, 10),
        (10, 20),
        (1, 20),
        (1, 10),
    ]


@pytest.fixture
def rect5() -> list[tuple[float, float]]:
    """Box with partial overlap to rect3."""
    return [
        (87, 10),
        (158, 10),
        (158, 400),
        (87, 400),
        (87, 10),
    ]


@pytest.fixture
def evaluate_detection_groundtruths(
    rect1: list[tuple[float, float]],
    rect2: list[tuple[float, float]],
    rect3: list[tuple[float, float]],
    img1: schemas.Datum,
    img2: schemas.Datum,
) -> list[schemas.GroundTruth]:
    return [
        schemas.GroundTruth(
            datum=img1,
            annotations=[
                schemas.Annotation(
                    is_instance=True,
                    labels=[schemas.Label(key="k1", value="v1")],
                    bounding_box=geometry.Box([rect1]),
                ),
                schemas.Annotation(
                    is_instance=True,
                    labels=[schemas.Label(key="k2", value="v2")],
                    bounding_box=geometry.Box([rect3]),
                ),
            ],
        ),
        schemas.GroundTruth(
            datum=img2,
            annotations=[
                schemas.Annotation(
                    is_instance=True,
                    labels=[schemas.Label(key="k1", value="v1")],
                    bounding_box=geometry.Box([rect2]),
                )
            ],
        ),
    ]


@pytest.fixture
def evaluate_detection_predictions(
    rect1: list[tuple[float, float]],
    rect2: list[tuple[float, float]],
    img1: schemas.Datum,
    img2: schemas.Datum,
) -> list[schemas.Prediction]:
    return [
        schemas.Prediction(
            datum=img1,
            annotations=[
                schemas.Annotation(
                    is_instance=True,
                    labels=[schemas.Label(key="k1", value="v1", score=0.3)],
                    bounding_box=geometry.Box([rect1]),
                )
            ],
        ),
        schemas.Prediction(
            datum=img2,
            annotations=[
                schemas.Annotation(
                    is_instance=True,
                    labels=[schemas.Label(key="k2", value="v2", score=0.98)],
                    bounding_box=geometry.Box([rect2]),
                )
            ],
        ),
    ]


@pytest.fixture
def evaluate_detection_groundtruths_with_label_maps(
    rect1: list[tuple[float, float]],
    rect2: list[tuple[float, float]],
    rect3: list[tuple[float, float]],
    img1: schemas.Datum,
    img2: schemas.Datum,
) -> list[schemas.GroundTruth]:
    return [
        schemas.GroundTruth(
            datum=img1,
            annotations=[
                schemas.Annotation(
                    is_instance=True,
                    labels=[
                        schemas.Label(key="class_name", value="maine coon cat")
                    ],
                    bounding_box=geometry.Box([rect1]),
                ),
                schemas.Annotation(
                    is_instance=True,
                    labels=[
                        schemas.Label(key="class", value="british shorthair")
                    ],
                    bounding_box=geometry.Box([rect3]),
                ),
                schemas.Annotation(
                    is_instance=True,
                    labels=[schemas.Label(key="k1", value="v1")],
                    bounding_box=geometry.Box([rect1]),
                ),
                schemas.Annotation(
                    is_instance=True,
                    labels=[schemas.Label(key="k2", value="v2")],
                    bounding_box=geometry.Box([rect3]),
                ),
            ],
        ),
        schemas.GroundTruth(
            datum=img2,
            annotations=[
                schemas.Annotation(
                    is_instance=True,
                    labels=[schemas.Label(key="class", value="siamese cat")],
                    bounding_box=geometry.Box([rect2]),
                ),
                schemas.Annotation(
                    is_instance=True,
                    labels=[schemas.Label(key="k1", value="v1")],
                    bounding_box=geometry.Box([rect2]),
                ),
            ],
        ),
    ]


@pytest.fixture
def evaluate_detection_predictions_with_label_maps(
    rect1: list[tuple[float, float]],
    rect2: list[tuple[float, float]],
    img1: schemas.Datum,
    img2: schemas.Datum,
) -> list[schemas.Prediction]:
    return [
        schemas.Prediction(
            datum=img1,
            annotations=[
                schemas.Annotation(
                    is_instance=True,
                    labels=[
                        schemas.Label(key="class", value="cat", score=0.3)
                    ],
                    bounding_box=geometry.Box([rect1]),
                ),
                schemas.Annotation(
                    is_instance=True,
                    labels=[schemas.Label(key="k1", value="v1", score=0.3)],
                    bounding_box=geometry.Box([rect1]),
                ),
            ],
        ),
        schemas.Prediction(
            datum=img2,
            annotations=[
                schemas.Annotation(
                    is_instance=True,
                    labels=[
                        schemas.Label(
                            key="class_name", value="cat", score=0.98
                        )
                    ],
                    bounding_box=geometry.Box([rect2]),
                ),
                schemas.Annotation(
                    is_instance=True,
                    labels=[schemas.Label(key="k2", value="v2", score=0.98)],
                    bounding_box=geometry.Box([rect2]),
                ),
            ],
        ),
    ]


# TODO separate the classification vs. object-detection data?
@pytest.fixture
def evaluate_tabular_clf_groundtruths():
    return pd.DataFrame(
        [
            {
                "id": 9040,
                "annotation_id": 11373,
                "label_id": 8031,
                "created_at": 1722267392923,
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "1",
                "datum_id": 822,
                "datum_uid": "uid0",
            },
            {
                "id": 9041,
                "annotation_id": 11374,
                "label_id": 8031,
                "created_at": 1722267392967,
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "1",
                "datum_id": 823,
                "datum_uid": "uid1",
            },
            {
                "id": 9042,
                "annotation_id": 11375,
                "label_id": 8033,
                "created_at": 1722267393007,
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "2",
                "datum_id": 824,
                "datum_uid": "uid2",
            },
            {
                "id": 9043,
                "annotation_id": 11376,
                "label_id": 8034,
                "created_at": 1722267393047,
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "0",
                "datum_id": 825,
                "datum_uid": "uid3",
            },
            {
                "id": 9044,
                "annotation_id": 11377,
                "label_id": 8034,
                "created_at": 1722267393088,
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "0",
                "datum_id": 826,
                "datum_uid": "uid4",
            },
            {
                "id": 9045,
                "annotation_id": 11378,
                "label_id": 8034,
                "created_at": 1722267393125,
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "0",
                "datum_id": 827,
                "datum_uid": "uid5",
            },
            {
                "id": 9046,
                "annotation_id": 11379,
                "label_id": 8031,
                "created_at": 1722267393166,
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "1",
                "datum_id": 828,
                "datum_uid": "uid6",
            },
            {
                "id": 9047,
                "annotation_id": 11380,
                "label_id": 8031,
                "created_at": 1722267393215,
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "1",
                "datum_id": 829,
                "datum_uid": "uid7",
            },
            {
                "id": 9048,
                "annotation_id": 11381,
                "label_id": 8031,
                "created_at": 1722267393263,
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "1",
                "datum_id": 830,
                "datum_uid": "uid8",
            },
            {
                "id": 9049,
                "annotation_id": 11382,
                "label_id": 8031,
                "created_at": 1722267393306,
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "1",
                "datum_id": 831,
                "datum_uid": "uid9",
            },
        ]
    )


@pytest.fixture
def evaluate_tabular_clf_predictions():
    return pd.DataFrame(
        [
            {
                "id": 4600,
                "annotation_id": 11385,
                "label_id": 8033,
                "score": 0.09,
                "created_at": pd.Timestamp("2024-07-29 15:36:33.502504"),
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "2",
                "datum_id": 824,
                "datum_uid": "uid2",
            },
            {
                "id": 4599,
                "annotation_id": 11385,
                "label_id": 8031,
                "score": 0.88,
                "created_at": pd.Timestamp("2024-07-29 15:36:33.502504"),
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "1",
                "datum_id": 824,
                "datum_uid": "uid2",
            },
            {
                "id": 4598,
                "annotation_id": 11385,
                "label_id": 8034,
                "score": 0.03,
                "created_at": pd.Timestamp("2024-07-29 15:36:33.502504"),
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "0",
                "datum_id": 824,
                "datum_uid": "uid2",
            },
            {
                "id": 4603,
                "annotation_id": 11386,
                "label_id": 8033,
                "score": 0.0,
                "created_at": pd.Timestamp("2024-07-29 15:36:33.546293"),
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "2",
                "datum_id": 825,
                "datum_uid": "uid3",
            },
            {
                "id": 4602,
                "annotation_id": 11386,
                "label_id": 8031,
                "score": 0.03,
                "created_at": pd.Timestamp("2024-07-29 15:36:33.546293"),
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "1",
                "datum_id": 825,
                "datum_uid": "uid3",
            },
            {
                "id": 4601,
                "annotation_id": 11386,
                "label_id": 8034,
                "score": 0.97,
                "created_at": pd.Timestamp("2024-07-29 15:36:33.546293"),
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "0",
                "datum_id": 825,
                "datum_uid": "uid3",
            },
            {
                "id": 4606,
                "annotation_id": 11387,
                "label_id": 8033,
                "score": 0.0,
                "created_at": pd.Timestamp("2024-07-29 15:36:33.586264"),
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "2",
                "datum_id": 826,
                "datum_uid": "uid4",
            },
            {
                "id": 4605,
                "annotation_id": 11387,
                "label_id": 8031,
                "score": 0.0,
                "created_at": pd.Timestamp("2024-07-29 15:36:33.586264"),
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "1",
                "datum_id": 826,
                "datum_uid": "uid4",
            },
            {
                "id": 4604,
                "annotation_id": 11387,
                "label_id": 8034,
                "score": 1.0,
                "created_at": pd.Timestamp("2024-07-29 15:36:33.586264"),
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "0",
                "datum_id": 826,
                "datum_uid": "uid4",
            },
            {
                "id": 4609,
                "annotation_id": 11388,
                "label_id": 8033,
                "score": 0.0,
                "created_at": pd.Timestamp("2024-07-29 15:36:33.631094"),
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "2",
                "datum_id": 827,
                "datum_uid": "uid5",
            },
            {
                "id": 4608,
                "annotation_id": 11388,
                "label_id": 8031,
                "score": 0.0,
                "created_at": pd.Timestamp("2024-07-29 15:36:33.631094"),
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "1",
                "datum_id": 827,
                "datum_uid": "uid5",
            },
            {
                "id": 4607,
                "annotation_id": 11388,
                "label_id": 8034,
                "score": 1.0,
                "created_at": pd.Timestamp("2024-07-29 15:36:33.631094"),
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "0",
                "datum_id": 827,
                "datum_uid": "uid5",
            },
            {
                "id": 4612,
                "annotation_id": 11389,
                "label_id": 8033,
                "score": 0.03,
                "created_at": pd.Timestamp("2024-07-29 15:36:33.673800"),
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "2",
                "datum_id": 828,
                "datum_uid": "uid6",
            },
            {
                "id": 4611,
                "annotation_id": 11389,
                "label_id": 8031,
                "score": 0.96,
                "created_at": pd.Timestamp("2024-07-29 15:36:33.673800"),
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "1",
                "datum_id": 828,
                "datum_uid": "uid6",
            },
            {
                "id": 4610,
                "annotation_id": 11389,
                "label_id": 8034,
                "score": 0.01,
                "created_at": pd.Timestamp("2024-07-29 15:36:33.673800"),
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "0",
                "datum_id": 828,
                "datum_uid": "uid6",
            },
            {
                "id": 4615,
                "annotation_id": 11390,
                "label_id": 8033,
                "score": 0.7,
                "created_at": pd.Timestamp("2024-07-29 15:36:33.709818"),
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "2",
                "datum_id": 829,
                "datum_uid": "uid7",
            },
            {
                "id": 4614,
                "annotation_id": 11390,
                "label_id": 8031,
                "score": 0.02,
                "created_at": pd.Timestamp("2024-07-29 15:36:33.709818"),
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "1",
                "datum_id": 829,
                "datum_uid": "uid7",
            },
            {
                "id": 4613,
                "annotation_id": 11390,
                "label_id": 8034,
                "score": 0.28,
                "created_at": pd.Timestamp("2024-07-29 15:36:33.709818"),
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "0",
                "datum_id": 829,
                "datum_uid": "uid7",
            },
            {
                "id": 4618,
                "annotation_id": 11391,
                "label_id": 8033,
                "score": 0.01,
                "created_at": pd.Timestamp("2024-07-29 15:36:33.745536"),
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "2",
                "datum_id": 830,
                "datum_uid": "uid8",
            },
            {
                "id": 4617,
                "annotation_id": 11391,
                "label_id": 8031,
                "score": 0.21,
                "created_at": pd.Timestamp("2024-07-29 15:36:33.745536"),
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "1",
                "datum_id": 830,
                "datum_uid": "uid8",
            },
            {
                "id": 4616,
                "annotation_id": 11391,
                "label_id": 8034,
                "score": 0.78,
                "created_at": pd.Timestamp("2024-07-29 15:36:33.745536"),
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "0",
                "datum_id": 830,
                "datum_uid": "uid8",
            },
            {
                "id": 4621,
                "annotation_id": 11392,
                "label_id": 8033,
                "score": 0.44,
                "created_at": pd.Timestamp("2024-07-29 15:36:33.797759"),
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "2",
                "datum_id": 831,
                "datum_uid": "uid9",
            },
            {
                "id": 4620,
                "annotation_id": 11392,
                "label_id": 8031,
                "score": 0.11,
                "created_at": pd.Timestamp("2024-07-29 15:36:33.797759"),
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "1",
                "datum_id": 831,
                "datum_uid": "uid9",
            },
            {
                "id": 4619,
                "annotation_id": 11392,
                "label_id": 8034,
                "score": 0.45,
                "created_at": pd.Timestamp("2024-07-29 15:36:33.797759"),
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "0",
                "datum_id": 831,
                "datum_uid": "uid9",
            },
            {
                "id": 4594,
                "annotation_id": 11383,
                "label_id": 8033,
                "score": 0.28,
                "created_at": pd.Timestamp("2024-07-29 15:36:33.411278"),
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "2",
                "datum_id": 822,
                "datum_uid": "uid0",
            },
            {
                "id": 4593,
                "annotation_id": 11383,
                "label_id": 8031,
                "score": 0.35,
                "created_at": pd.Timestamp("2024-07-29 15:36:33.411278"),
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "1",
                "datum_id": 822,
                "datum_uid": "uid0",
            },
            {
                "id": 4592,
                "annotation_id": 11383,
                "label_id": 8034,
                "score": 0.37,
                "created_at": pd.Timestamp("2024-07-29 15:36:33.411278"),
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "0",
                "datum_id": 822,
                "datum_uid": "uid0",
            },
            {
                "id": 4597,
                "annotation_id": 11384,
                "label_id": 8033,
                "score": 0.15,
                "created_at": pd.Timestamp("2024-07-29 15:36:33.465625"),
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "2",
                "datum_id": 823,
                "datum_uid": "uid1",
            },
            {
                "id": 4596,
                "annotation_id": 11384,
                "label_id": 8031,
                "score": 0.61,
                "created_at": pd.Timestamp("2024-07-29 15:36:33.465625"),
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "1",
                "datum_id": 823,
                "datum_uid": "uid1",
            },
            {
                "id": 4595,
                "annotation_id": 11384,
                "label_id": 8034,
                "score": 0.24,
                "created_at": pd.Timestamp("2024-07-29 15:36:33.465625"),
                "dataset_name": "test_dataset",
                "label_key": "class",
                "label_value": "0",
                "datum_id": 823,
                "datum_uid": "uid1",
            },
        ]
    )


@pytest.fixture
def evaluate_image_clf_groundtruths():
    return [
        schemas.GroundTruth(
            datum=schemas.Datum(
                uid="uid5",
                metadata={
                    "height": 900,
                    "width": 300,
                },
            ),
            annotations=[
                schemas.Annotation(
                    labels=[
                        schemas.Label(key="k4", value="v4"),
                        schemas.Label(key="k5", value="v5"),
                    ],
                ),
            ],
        ),
        schemas.GroundTruth(
            datum=schemas.Datum(
                uid="uid6",
                metadata={
                    "height": 900,
                    "width": 300,
                },
            ),
            annotations=[
                schemas.Annotation(
                    labels=[schemas.Label(key="k4", value="v4")],
                )
            ],
        ),
        schemas.GroundTruth(
            datum=schemas.Datum(
                uid="uid8",
                metadata={
                    "height": 900,
                    "width": 300,
                },
            ),
            annotations=[
                schemas.Annotation(
                    labels=[schemas.Label(key="k3", value="v3")],
                )
            ],
        ),
    ]


@pytest.fixture
def evaluate_image_clf_predictions():
    return [
        schemas.Prediction(
            datum=schemas.Datum(
                uid="uid5",
                metadata={
                    "height": 900,
                    "width": 300,
                },
            ),
            annotations=[
                schemas.Annotation(
                    labels=[
                        schemas.Label(key="k4", value="v1", score=0.47),
                        schemas.Label(key="k4", value="v8", score=0.53),
                        schemas.Label(key="k5", value="v1", score=1.0),
                    ],
                )
            ],
        ),
        schemas.Prediction(
            datum=schemas.Datum(
                uid="uid6",
                metadata={
                    "height": 900,
                    "width": 300,
                },
            ),
            annotations=[
                schemas.Annotation(
                    labels=[
                        schemas.Label(key="k4", value="v4", score=0.71),
                        schemas.Label(key="k4", value="v5", score=0.29),
                    ],
                )
            ],
        ),
        schemas.Prediction(
            datum=schemas.Datum(
                uid="uid8",
                metadata={
                    "height": 900,
                    "width": 300,
                },
            ),
            annotations=[
                schemas.Annotation(
                    labels=[
                        schemas.Label(key="k3", value="v1", score=1.0),
                    ],
                )
            ],
        ),
    ]


@pytest.fixture
def gt_clfs_tabular() -> list[int]:
    """ground truth for a tabular classification task"""
    return [1, 1, 2, 0, 0, 0, 1, 1, 1, 1]


@pytest.fixture
def pred_clfs_tabular() -> list[list[float]]:
    """predictions for a tabular classification task"""
    return [
        [0.37, 0.35, 0.28],
        [0.24, 0.61, 0.15],
        [0.03, 0.88, 0.09],
        [0.97, 0.03, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.01, 0.96, 0.03],
        [0.28, 0.02, 0.7],
        [0.78, 0.21, 0.01],
        [0.45, 0.11, 0.44],
    ]


@pytest.fixture
def image_height():
    return 900


@pytest.fixture
def image_width():
    return 300


@pytest.fixture
def img1(
    image_height: int,
    image_width: int,
) -> schemas.Datum:
    coordinates = [
        [
            (125.2750725, 38.760525),
            (125.3902365, 38.775069),
            (125.5054005, 38.789613),
            (125.5051935, 38.71402425),
            (125.5049865, 38.6384355),
            (125.3902005, 38.6244225),
            (125.2754145, 38.6104095),
            (125.2752435, 38.68546725),
            (125.2750725, 38.760525),
        ]
    ]
    return schemas.Datum(
        uid="uid1",
        metadata={
            "geospatial": geometry.Polygon(coordinates),
            "height": image_height,
            "width": image_width,
        },
    )


@pytest.fixture
def img2(
    image_height: int,
    image_width: int,
) -> schemas.Datum:
    coordinates = (44.1, 22.4)
    return schemas.Datum(
        uid="uid2",
        metadata={
            "geospatial": geometry.Point(coordinates),
            "height": image_height,
            "width": image_width,
        },
    )


@pytest.fixture
def img5(
    image_height: int,
    image_width: int,
) -> schemas.Datum:
    return schemas.Datum(
        uid="uid5",
        metadata={
            "height": image_height,
            "width": image_width,
        },
    )


@pytest.fixture
def img6(
    image_height: int,
    image_width: int,
) -> schemas.Datum:
    return schemas.Datum(
        uid="uid6",
        metadata={
            "height": image_height,
            "width": image_width,
        },
    )


@pytest.fixture
def img8(
    image_height: int,
    image_width: int,
) -> schemas.Datum:
    return schemas.Datum(
        uid="uid8",
        metadata={
            "height": image_height,
            "width": image_width,
        },
    )


@pytest.fixture
def gt_clfs_with_label_maps(
    img5: schemas.Datum,
    img6: schemas.Datum,
    img8: schemas.Datum,
) -> list[schemas.GroundTruth]:
    return [
        schemas.GroundTruth(
            datum=img5,
            annotations=[
                schemas.Annotation(
                    labels=[
                        schemas.Label(key="k4", value="v4"),
                        schemas.Label(key="k5", value="v5"),
                        schemas.Label(key="class", value="siamese cat"),
                    ],
                ),
            ],
        ),
        schemas.GroundTruth(
            datum=img6,
            annotations=[
                schemas.Annotation(
                    labels=[
                        schemas.Label(key="k4", value="v4"),
                        schemas.Label(key="class", value="british shorthair"),
                    ],
                )
            ],
        ),
        schemas.GroundTruth(
            datum=img8,
            annotations=[
                schemas.Annotation(
                    labels=[
                        schemas.Label(key="k3", value="v3"),
                        schemas.Label(key="class", value="tabby cat"),
                    ],
                )
            ],
        ),
    ]


@pytest.fixture
def pred_clfs_with_label_maps(
    img5: schemas.Datum,
    img6: schemas.Datum,
    img8: schemas.Datum,
) -> list[schemas.Prediction]:
    return [
        schemas.Prediction(
            datum=img5,
            annotations=[
                schemas.Annotation(
                    labels=[
                        schemas.Label(key="k4", value="v1", score=0.47),
                        schemas.Label(key="k4", value="v8", score=0.53),
                        schemas.Label(key="k5", value="v1", score=1.0),
                        schemas.Label(key="class", value="cat", score=1.0),
                    ],
                )
            ],
        ),
        schemas.Prediction(
            datum=img6,
            annotations=[
                schemas.Annotation(
                    labels=[
                        schemas.Label(key="k4", value="v4", score=0.71),
                        schemas.Label(key="k4", value="v5", score=0.29),
                        schemas.Label(
                            key="class_name", value="cat", score=1.0
                        ),
                    ],
                )
            ],
        ),
        schemas.Prediction(
            datum=img8,
            annotations=[
                schemas.Annotation(
                    labels=[
                        schemas.Label(key="k3", value="v1", score=1.0),
                        schemas.Label(key="class", value="cat", score=1.0),
                    ],
                )
            ],
        ),
    ]


@pytest.fixture
def gt_clfs_label_key_mismatch(
    img5: schemas.Datum,
    img6: schemas.Datum,
    img8: schemas.Datum,
) -> list[schemas.GroundTruth]:
    return [
        schemas.GroundTruth(
            datum=img5,
            annotations=[
                schemas.Annotation(
                    labels=[
                        schemas.Label(key="k4", value="v4"),
                        schemas.Label(key="k5", value="v5"),
                    ],
                ),
            ],
        ),
        schemas.GroundTruth(
            datum=img6,
            annotations=[
                schemas.Annotation(
                    labels=[schemas.Label(key="k4", value="v4")],
                )
            ],
        ),
        schemas.GroundTruth(
            datum=img8,
            annotations=[
                schemas.Annotation(
                    labels=[schemas.Label(key="k3", value="v3")],
                )
            ],
        ),
    ]


@pytest.fixture
def pred_clfs_label_key_mismatch(
    img5: schemas.Datum, img6: schemas.Datum
) -> list[schemas.Prediction]:
    return [
        schemas.Prediction(
            datum=img5,
            annotations=[
                schemas.Annotation(
                    labels=[
                        schemas.Label(key="k12", value="v12", score=0.47),
                        schemas.Label(key="k12", value="v16", score=0.53),
                        schemas.Label(key="k13", value="v13", score=1.0),
                    ],
                )
            ],
        ),
        schemas.Prediction(
            datum=img6,
            annotations=[
                schemas.Annotation(
                    labels=[
                        schemas.Label(key="k4", value="v4", score=0.71),
                        schemas.Label(key="k4", value="v5", score=0.29),
                    ],
                )
            ],
        ),
    ]


@pytest.fixture
def gt_clfs(
    img5: schemas.Datum,
    img6: schemas.Datum,
    img8: schemas.Datum,
) -> list[schemas.GroundTruth]:
    return [
        schemas.GroundTruth(
            datum=img5,
            annotations=[
                schemas.Annotation(
                    labels=[
                        schemas.Label(key="k4", value="v4"),
                        schemas.Label(key="k5", value="v5"),
                    ],
                ),
            ],
        ),
        schemas.GroundTruth(
            datum=img6,
            annotations=[
                schemas.Annotation(
                    labels=[schemas.Label(key="k4", value="v4")],
                )
            ],
        ),
        schemas.GroundTruth(
            datum=img8,
            annotations=[
                schemas.Annotation(
                    labels=[schemas.Label(key="k3", value="v3")],
                )
            ],
        ),
    ]


# TODO handle warnings?
@pytest.fixture
def classification_functional_test_data():
    animal_gts = ["bird", "dog", "bird", "bird", "cat", "dog"]
    animal_preds = [
        {"bird": 0.6, "dog": 0.2, "cat": 0.2},
        {"cat": 0.9, "dog": 0.1, "bird": 0.0},
        {"cat": 0.8, "dog": 0.05, "bird": 0.15},
        {"dog": 0.75, "cat": 0.1, "bird": 0.15},
        {"cat": 1.0, "dog": 0.0, "bird": 0.0},
        {"cat": 0.4, "dog": 0.4, "bird": 0.2},
    ]

    color_gts = ["white", "white", "red", "blue", "black", "red"]
    color_preds = [
        {"white": 0.65, "red": 0.1, "blue": 0.2, "black": 0.05},
        {"blue": 0.5, "white": 0.3, "red": 0.0, "black": 0.2},
        {"red": 0.4, "white": 0.2, "blue": 0.1, "black": 0.3},
        {"white": 1.0, "red": 0.0, "blue": 0.0, "black": 0.0},
        {"red": 0.8, "white": 0.0, "blue": 0.2, "black": 0.0},
        {"red": 0.9, "white": 0.06, "blue": 0.01, "black": 0.03},
    ]

    imgs = [
        schemas.Datum(
            uid=f"uid{i}",
            metadata={
                "height": 128,
                "width": 256,
            },
        )
        for i in range(6)
    ]

    gts = [
        schemas.GroundTruth(
            datum=imgs[i],
            annotations=[
                schemas.Annotation(
                    labels=[
                        schemas.Label(key="animal", value=animal_gts[i]),
                        schemas.Label(key="color", value=color_gts[i]),
                    ],
                )
            ],
        )
        for i in range(6)
    ]

    preds = [
        schemas.Prediction(
            datum=imgs[i],
            annotations=[
                schemas.Annotation(
                    labels=[
                        schemas.Label(key="animal", value=value, score=score)
                        for value, score in animal_preds[i].items()
                    ]
                    + [
                        schemas.Label(key="color", value=value, score=score)
                        for value, score in color_preds[i].items()
                    ],
                )
            ],
        )
        for i in range(6)
    ]

    return (gts, preds)


@pytest.fixture
def classification_functional_test_groundtruth_df():
    """Used in test_rocauc_with_label_map so that we can test _calculate_rocauc directly, since this original text violated the matching groundtruth/prediction label keys criteria."""
    return pd.DataFrame(
        [
            {
                "dataset_name": "delete later",
                "datum_uid": "uid0",
                "datum_id": -5384017641951508119,
                "datum_metadata": {
                    "height": 128,
                    "width": 256,
                    "md1": "md1-val0",
                    "md2": "md1-val0",
                },
                "annotation_id": -7219056621792402854,
                "annotation_metadata": None,
                "bounding_box": None,
                "raster": None,
                "embedding": None,
                "polygon": None,
                "is_instance": None,
                "label_key": "animal",
                "label_value": "bird",
                "label_id": 6844413835611710259,
                "id": -6147199056584656887,
                "mapped_groundtruth_label_keys": "class",
                "label": ("animal", "bird"),
                "grouper_key": "other_class",
                "grouper_value": "avian",
            },
            {
                "dataset_name": "delete later",
                "datum_uid": "uid0",
                "datum_id": -5384017641951508119,
                "datum_metadata": {
                    "height": 128,
                    "width": 256,
                    "md1": "md1-val0",
                    "md2": "md1-val0",
                },
                "annotation_id": -7219056621792402854,
                "annotation_metadata": None,
                "bounding_box": None,
                "raster": None,
                "embedding": None,
                "polygon": None,
                "is_instance": None,
                "label_key": "color",
                "label_value": "white",
                "label_id": 1137203407882171315,
                "id": 8837325099618861823,
                "mapped_groundtruth_label_keys": "color",
                "label": ("color", "white"),
                "grouper_key": "color",
                "grouper_value": "white",
            },
            {
                "dataset_name": "delete later",
                "datum_uid": "uid1",
                "datum_id": -8510955155591861879,
                "datum_metadata": {
                    "height": 128,
                    "width": 256,
                    "md1": "md1-val0",
                    "md2": "md1-val1",
                },
                "annotation_id": 8790918715870844863,
                "annotation_metadata": None,
                "bounding_box": None,
                "raster": None,
                "embedding": None,
                "polygon": None,
                "is_instance": None,
                "label_key": "animal",
                "label_value": "dog",
                "label_id": 8009222289478380372,
                "id": -1593123359500601416,
                "mapped_groundtruth_label_keys": "class",
                "label": ("animal", "dog"),
                "grouper_key": "class",
                "grouper_value": "mammal",
            },
            {
                "dataset_name": "delete later",
                "datum_uid": "uid1",
                "datum_id": -8510955155591861879,
                "datum_metadata": {
                    "height": 128,
                    "width": 256,
                    "md1": "md1-val0",
                    "md2": "md1-val1",
                },
                "annotation_id": 8790918715870844863,
                "annotation_metadata": None,
                "bounding_box": None,
                "raster": None,
                "embedding": None,
                "polygon": None,
                "is_instance": None,
                "label_key": "color",
                "label_value": "white",
                "label_id": 1137203407882171315,
                "id": 3582630467549642626,
                "mapped_groundtruth_label_keys": "color",
                "label": ("color", "white"),
                "grouper_key": "color",
                "grouper_value": "white",
            },
            {
                "dataset_name": "delete later",
                "datum_uid": "uid2",
                "datum_id": -8411940843701065439,
                "datum_metadata": {
                    "height": 128,
                    "width": 256,
                    "md1": "md1-val0",
                    "md2": "md1-val2",
                },
                "annotation_id": -3239983991430348508,
                "annotation_metadata": None,
                "bounding_box": None,
                "raster": None,
                "embedding": None,
                "polygon": None,
                "is_instance": None,
                "label_key": "animal",
                "label_value": "bird",
                "label_id": 6844413835611710259,
                "id": -6917823642762098726,
                "mapped_groundtruth_label_keys": "class",
                "label": ("animal", "bird"),
                "grouper_key": "other_class",
                "grouper_value": "avian",
            },
            {
                "dataset_name": "delete later",
                "datum_uid": "uid2",
                "datum_id": -8411940843701065439,
                "datum_metadata": {
                    "height": 128,
                    "width": 256,
                    "md1": "md1-val0",
                    "md2": "md1-val2",
                },
                "annotation_id": -3239983991430348508,
                "annotation_metadata": None,
                "bounding_box": None,
                "raster": None,
                "embedding": None,
                "polygon": None,
                "is_instance": None,
                "label_key": "color",
                "label_value": "red",
                "label_id": -3886640484917084310,
                "id": -1339278877785114234,
                "mapped_groundtruth_label_keys": "color",
                "label": ("color", "red"),
                "grouper_key": "color",
                "grouper_value": "red",
            },
            {
                "dataset_name": "delete later",
                "datum_uid": "uid3",
                "datum_id": -2265528102457502931,
                "datum_metadata": {
                    "height": 128,
                    "width": 256,
                    "md1": "md1-val0",
                    "md2": "md1-val0",
                },
                "annotation_id": 4382196578706948542,
                "annotation_metadata": None,
                "bounding_box": None,
                "raster": None,
                "embedding": None,
                "polygon": None,
                "is_instance": None,
                "label_key": "animal",
                "label_value": "bird",
                "label_id": 6844413835611710259,
                "id": 1083297721794099590,
                "mapped_groundtruth_label_keys": "class",
                "label": ("animal", "bird"),
                "grouper_key": "other_class",
                "grouper_value": "avian",
            },
            {
                "dataset_name": "delete later",
                "datum_uid": "uid3",
                "datum_id": -2265528102457502931,
                "datum_metadata": {
                    "height": 128,
                    "width": 256,
                    "md1": "md1-val0",
                    "md2": "md1-val0",
                },
                "annotation_id": 4382196578706948542,
                "annotation_metadata": None,
                "bounding_box": None,
                "raster": None,
                "embedding": None,
                "polygon": None,
                "is_instance": None,
                "label_key": "color",
                "label_value": "blue",
                "label_id": -1372075868144138351,
                "id": -615284425434206300,
                "mapped_groundtruth_label_keys": "color",
                "label": ("color", "blue"),
                "grouper_key": "color",
                "grouper_value": "blue",
            },
            {
                "dataset_name": "delete later",
                "datum_uid": "uid4",
                "datum_id": -4389124420839664731,
                "datum_metadata": {
                    "height": 128,
                    "width": 256,
                    "md1": "md1-val1",
                    "md2": "md1-val1",
                },
                "annotation_id": 4962111685767385274,
                "annotation_metadata": None,
                "bounding_box": None,
                "raster": None,
                "embedding": None,
                "polygon": None,
                "is_instance": None,
                "label_key": "animal",
                "label_value": "cat",
                "label_id": 4524343817500814041,
                "id": -7816578330009256692,
                "mapped_groundtruth_label_keys": "class",
                "label": ("animal", "cat"),
                "grouper_key": "class",
                "grouper_value": "mammal",
            },
            {
                "dataset_name": "delete later",
                "datum_uid": "uid4",
                "datum_id": -4389124420839664731,
                "datum_metadata": {
                    "height": 128,
                    "width": 256,
                    "md1": "md1-val1",
                    "md2": "md1-val1",
                },
                "annotation_id": 4962111685767385274,
                "annotation_metadata": None,
                "bounding_box": None,
                "raster": None,
                "embedding": None,
                "polygon": None,
                "is_instance": None,
                "label_key": "color",
                "label_value": "black",
                "label_id": 1817852877141727993,
                "id": -5129897778521880842,
                "mapped_groundtruth_label_keys": "color",
                "label": ("color", "black"),
                "grouper_key": "color",
                "grouper_value": "black",
            },
            {
                "dataset_name": "delete later",
                "datum_uid": "uid5",
                "datum_id": 5314927723853009775,
                "datum_metadata": {
                    "height": 128,
                    "width": 256,
                    "md1": "md1-val0",
                    "md2": "md1-val2",
                },
                "annotation_id": -746121109706998955,
                "annotation_metadata": None,
                "bounding_box": None,
                "raster": None,
                "embedding": None,
                "polygon": None,
                "is_instance": None,
                "label_key": "animal",
                "label_value": "dog",
                "label_id": 8009222289478380372,
                "id": -6769946184488850844,
                "mapped_groundtruth_label_keys": "class",
                "label": ("animal", "dog"),
                "grouper_key": "class",
                "grouper_value": "mammal",
            },
            {
                "dataset_name": "delete later",
                "datum_uid": "uid5",
                "datum_id": 5314927723853009775,
                "datum_metadata": {
                    "height": 128,
                    "width": 256,
                    "md1": "md1-val0",
                    "md2": "md1-val2",
                },
                "annotation_id": -746121109706998955,
                "annotation_metadata": None,
                "bounding_box": None,
                "raster": None,
                "embedding": None,
                "polygon": None,
                "is_instance": None,
                "label_key": "color",
                "label_value": "red",
                "label_id": -3886640484917084310,
                "id": -503991891998595125,
                "mapped_groundtruth_label_keys": "color",
                "label": ("color", "red"),
                "grouper_key": "color",
                "grouper_value": "red",
            },
        ]
    )


@pytest.fixture
def classification_functional_test_prediction_df():
    """Used in test_rocauc_with_label_map so that we can test _calculate_rocauc directly, since this original text violated the matching groundtruth/prediction label keys criteria."""
    return pd.DataFrame(
        [
            {
                "dataset_name": "delete later",
                "datum_uid": "uid0",
                "datum_id": -5384017641951508119,
                "datum_metadata": {
                    "height": 128,
                    "width": 256,
                    "md1": "md1-val0",
                    "md2": "md1-val0",
                },
                "annotation_id": -6728727181236673047,
                "annotation_metadata": None,
                "bounding_box": None,
                "raster": None,
                "embedding": None,
                "polygon": None,
                "is_instance": None,
                "label_key": "animal",
                "label_value": "bird",
                "score": 0.6,
                "label_id": -5215084239238914495,
                "id": -1240527857667701281,
                "mapped_prediction_label_keys": "class",
                "label": ("animal", "bird"),
                "grouper_key": "other_class",
                "grouper_value": "avian",
            },
            {
                "dataset_name": "delete later",
                "datum_uid": "uid0",
                "datum_id": -5384017641951508119,
                "datum_metadata": {
                    "height": 128,
                    "width": 256,
                    "md1": "md1-val0",
                    "md2": "md1-val0",
                },
                "annotation_id": -6728727181236673047,
                "annotation_metadata": None,
                "bounding_box": None,
                "raster": None,
                "embedding": None,
                "polygon": None,
                "is_instance": None,
                "label_key": "animal",
                "label_value": "dog",
                "score": 0.2,
                "label_id": -6049586979668957678,
                "id": 49317224219915580,
                "mapped_prediction_label_keys": "class",
                "label": ("animal", "dog"),
                "grouper_key": "class",
                "grouper_value": "mammal",
            },
            {
                "dataset_name": "delete later",
                "datum_uid": "uid0",
                "datum_id": -5384017641951508119,
                "datum_metadata": {
                    "height": 128,
                    "width": 256,
                    "md1": "md1-val0",
                    "md2": "md1-val0",
                },
                "annotation_id": -6728727181236673047,
                "annotation_metadata": None,
                "bounding_box": None,
                "raster": None,
                "embedding": None,
                "polygon": None,
                "is_instance": None,
                "label_key": "animal",
                "label_value": "cat",
                "score": 0.2,
                "label_id": 7273800936934963489,
                "id": 233173136032973625,
                "mapped_prediction_label_keys": "class",
                "label": ("animal", "cat"),
                "grouper_key": "class",
                "grouper_value": "mammal",
            },
            {
                "dataset_name": "delete later",
                "datum_uid": "uid0",
                "datum_id": -5384017641951508119,
                "datum_metadata": {
                    "height": 128,
                    "width": 256,
                    "md1": "md1-val0",
                    "md2": "md1-val0",
                },
                "annotation_id": -6728727181236673047,
                "annotation_metadata": None,
                "bounding_box": None,
                "raster": None,
                "embedding": None,
                "polygon": None,
                "is_instance": None,
                "label_key": "color",
                "label_value": "white",
                "score": 0.65,
                "label_id": -4826903763707637373,
                "id": -6184807819874130814,
                "mapped_prediction_label_keys": "color",
                "label": ("color", "white"),
                "grouper_key": "color",
                "grouper_value": "white",
            },
            {
                "dataset_name": "delete later",
                "datum_uid": "uid0",
                "datum_id": -5384017641951508119,
                "datum_metadata": {
                    "height": 128,
                    "width": 256,
                    "md1": "md1-val0",
                    "md2": "md1-val0",
                },
                "annotation_id": -6728727181236673047,
                "annotation_metadata": None,
                "bounding_box": None,
                "raster": None,
                "embedding": None,
                "polygon": None,
                "is_instance": None,
                "label_key": "color",
                "label_value": "red",
                "score": 0.1,
                "label_id": 4216827315928697217,
                "id": 5704534164417962892,
                "mapped_prediction_label_keys": "color",
                "label": ("color", "red"),
                "grouper_key": "color",
                "grouper_value": "red",
            },
            {
                "dataset_name": "delete later",
                "datum_uid": "uid0",
                "datum_id": -5384017641951508119,
                "datum_metadata": {
                    "height": 128,
                    "width": 256,
                    "md1": "md1-val0",
                    "md2": "md1-val0",
                },
                "annotation_id": -6728727181236673047,
                "annotation_metadata": None,
                "bounding_box": None,
                "raster": None,
                "embedding": None,
                "polygon": None,
                "is_instance": None,
                "label_key": "color",
                "label_value": "blue",
                "score": 0.2,
                "label_id": -3960395303314501711,
                "id": 1511896606515226706,
                "mapped_prediction_label_keys": "color",
                "label": ("color", "blue"),
                "grouper_key": "color",
                "grouper_value": "blue",
            },
            {
                "dataset_name": "delete later",
                "datum_uid": "uid0",
                "datum_id": -5384017641951508119,
                "datum_metadata": {
                    "height": 128,
                    "width": 256,
                    "md1": "md1-val0",
                    "md2": "md1-val0",
                },
                "annotation_id": -6728727181236673047,
                "annotation_metadata": None,
                "bounding_box": None,
                "raster": None,
                "embedding": None,
                "polygon": None,
                "is_instance": None,
                "label_key": "color",
                "label_value": "black",
                "score": 0.05,
                "label_id": -8589704813442599109,
                "id": 3647731253780364946,
                "mapped_prediction_label_keys": "color",
                "label": ("color", "black"),
                "grouper_key": "color",
                "grouper_value": "black",
            },
            {
                "dataset_name": "delete later",
                "datum_uid": "uid1",
                "datum_id": -8510955155591861879,
                "datum_metadata": {
                    "height": 128,
                    "width": 256,
                    "md1": "md1-val0",
                    "md2": "md1-val1",
                },
                "annotation_id": 4939978831501967353,
                "annotation_metadata": None,
                "bounding_box": None,
                "raster": None,
                "embedding": None,
                "polygon": None,
                "is_instance": None,
                "label_key": "animal",
                "label_value": "cat",
                "score": 0.9,
                "label_id": 2094222191875474652,
                "id": -4753231139294527417,
                "mapped_prediction_label_keys": "class",
                "label": ("animal", "cat"),
                "grouper_key": "class",
                "grouper_value": "mammal",
            },
            {
                "dataset_name": "delete later",
                "datum_uid": "uid1",
                "datum_id": -8510955155591861879,
                "datum_metadata": {
                    "height": 128,
                    "width": 256,
                    "md1": "md1-val0",
                    "md2": "md1-val1",
                },
                "annotation_id": 4939978831501967353,
                "annotation_metadata": None,
                "bounding_box": None,
                "raster": None,
                "embedding": None,
                "polygon": None,
                "is_instance": None,
                "label_key": "animal",
                "label_value": "dog",
                "score": 0.1,
                "label_id": -4878077841794693757,
                "id": 8538318431236799830,
                "mapped_prediction_label_keys": "class",
                "label": ("animal", "dog"),
                "grouper_key": "class",
                "grouper_value": "mammal",
            },
            {
                "dataset_name": "delete later",
                "datum_uid": "uid1",
                "datum_id": -8510955155591861879,
                "datum_metadata": {
                    "height": 128,
                    "width": 256,
                    "md1": "md1-val0",
                    "md2": "md1-val1",
                },
                "annotation_id": 4939978831501967353,
                "annotation_metadata": None,
                "bounding_box": None,
                "raster": None,
                "embedding": None,
                "polygon": None,
                "is_instance": None,
                "label_key": "animal",
                "label_value": "bird",
                "score": 0.0,
                "label_id": 8183125692418530608,
                "id": 5468044993361705841,
                "mapped_prediction_label_keys": "class",
                "label": ("animal", "bird"),
                "grouper_key": "other_class",
                "grouper_value": "avian",
            },
            {
                "dataset_name": "delete later",
                "datum_uid": "uid1",
                "datum_id": -8510955155591861879,
                "datum_metadata": {
                    "height": 128,
                    "width": 256,
                    "md1": "md1-val0",
                    "md2": "md1-val1",
                },
                "annotation_id": 4939978831501967353,
                "annotation_metadata": None,
                "bounding_box": None,
                "raster": None,
                "embedding": None,
                "polygon": None,
                "is_instance": None,
                "label_key": "color",
                "label_value": "blue",
                "score": 0.5,
                "label_id": 5578669252512141405,
                "id": 5993876661711494245,
                "mapped_prediction_label_keys": "color",
                "label": ("color", "blue"),
                "grouper_key": "color",
                "grouper_value": "blue",
            },
            {
                "dataset_name": "delete later",
                "datum_uid": "uid1",
                "datum_id": -8510955155591861879,
                "datum_metadata": {
                    "height": 128,
                    "width": 256,
                    "md1": "md1-val0",
                    "md2": "md1-val1",
                },
                "annotation_id": 4939978831501967353,
                "annotation_metadata": None,
                "bounding_box": None,
                "raster": None,
                "embedding": None,
                "polygon": None,
                "is_instance": None,
                "label_key": "color",
                "label_value": "white",
                "score": 0.3,
                "label_id": -4200814355896957607,
                "id": -1473852835329269153,
                "mapped_prediction_label_keys": "color",
                "label": ("color", "white"),
                "grouper_key": "color",
                "grouper_value": "white",
            },
            {
                "dataset_name": "delete later",
                "datum_uid": "uid1",
                "datum_id": -8510955155591861879,
                "datum_metadata": {
                    "height": 128,
                    "width": 256,
                    "md1": "md1-val0",
                    "md2": "md1-val1",
                },
                "annotation_id": 4939978831501967353,
                "annotation_metadata": None,
                "bounding_box": None,
                "raster": None,
                "embedding": None,
                "polygon": None,
                "is_instance": None,
                "label_key": "color",
                "label_value": "red",
                "score": 0.0,
                "label_id": -519495577997781294,
                "id": -2806063230919808758,
                "mapped_prediction_label_keys": "color",
                "label": ("color", "red"),
                "grouper_key": "color",
                "grouper_value": "red",
            },
            {
                "dataset_name": "delete later",
                "datum_uid": "uid1",
                "datum_id": -8510955155591861879,
                "datum_metadata": {
                    "height": 128,
                    "width": 256,
                    "md1": "md1-val0",
                    "md2": "md1-val1",
                },
                "annotation_id": 4939978831501967353,
                "annotation_metadata": None,
                "bounding_box": None,
                "raster": None,
                "embedding": None,
                "polygon": None,
                "is_instance": None,
                "label_key": "color",
                "label_value": "black",
                "score": 0.2,
                "label_id": -4372451618257326717,
                "id": -9192777550609387657,
                "mapped_prediction_label_keys": "color",
                "label": ("color", "black"),
                "grouper_key": "color",
                "grouper_value": "black",
            },
            {
                "dataset_name": "delete later",
                "datum_uid": "uid2",
                "datum_id": -8411940843701065439,
                "datum_metadata": {
                    "height": 128,
                    "width": 256,
                    "md1": "md1-val0",
                    "md2": "md1-val2",
                },
                "annotation_id": 7499720668016145718,
                "annotation_metadata": None,
                "bounding_box": None,
                "raster": None,
                "embedding": None,
                "polygon": None,
                "is_instance": None,
                "label_key": "animal",
                "label_value": "cat",
                "score": 0.8,
                "label_id": 3361029567128538938,
                "id": -2495225296460022208,
                "mapped_prediction_label_keys": "class",
                "label": ("animal", "cat"),
                "grouper_key": "class",
                "grouper_value": "mammal",
            },
            {
                "dataset_name": "delete later",
                "datum_uid": "uid2",
                "datum_id": -8411940843701065439,
                "datum_metadata": {
                    "height": 128,
                    "width": 256,
                    "md1": "md1-val0",
                    "md2": "md1-val2",
                },
                "annotation_id": 7499720668016145718,
                "annotation_metadata": None,
                "bounding_box": None,
                "raster": None,
                "embedding": None,
                "polygon": None,
                "is_instance": None,
                "label_key": "animal",
                "label_value": "dog",
                "score": 0.05,
                "label_id": 1495879137950468608,
                "id": 96491879800885197,
                "mapped_prediction_label_keys": "class",
                "label": ("animal", "dog"),
                "grouper_key": "class",
                "grouper_value": "mammal",
            },
            {
                "dataset_name": "delete later",
                "datum_uid": "uid2",
                "datum_id": -8411940843701065439,
                "datum_metadata": {
                    "height": 128,
                    "width": 256,
                    "md1": "md1-val0",
                    "md2": "md1-val2",
                },
                "annotation_id": 7499720668016145718,
                "annotation_metadata": None,
                "bounding_box": None,
                "raster": None,
                "embedding": None,
                "polygon": None,
                "is_instance": None,
                "label_key": "animal",
                "label_value": "bird",
                "score": 0.15,
                "label_id": -3283720280595522641,
                "id": 1354699752396805280,
                "mapped_prediction_label_keys": "class",
                "label": ("animal", "bird"),
                "grouper_key": "other_class",
                "grouper_value": "avian",
            },
            {
                "dataset_name": "delete later",
                "datum_uid": "uid2",
                "datum_id": -8411940843701065439,
                "datum_metadata": {
                    "height": 128,
                    "width": 256,
                    "md1": "md1-val0",
                    "md2": "md1-val2",
                },
                "annotation_id": 7499720668016145718,
                "annotation_metadata": None,
                "bounding_box": None,
                "raster": None,
                "embedding": None,
                "polygon": None,
                "is_instance": None,
                "label_key": "color",
                "label_value": "red",
                "score": 0.4,
                "label_id": -2416149083383886333,
                "id": 268130056698580260,
                "mapped_prediction_label_keys": "color",
                "label": ("color", "red"),
                "grouper_key": "color",
                "grouper_value": "red",
            },
            {
                "dataset_name": "delete later",
                "datum_uid": "uid2",
                "datum_id": -8411940843701065439,
                "datum_metadata": {
                    "height": 128,
                    "width": 256,
                    "md1": "md1-val0",
                    "md2": "md1-val2",
                },
                "annotation_id": 7499720668016145718,
                "annotation_metadata": None,
                "bounding_box": None,
                "raster": None,
                "embedding": None,
                "polygon": None,
                "is_instance": None,
                "label_key": "color",
                "label_value": "white",
                "score": 0.2,
                "label_id": -1998826250032086593,
                "id": -4021126010657534621,
                "mapped_prediction_label_keys": "color",
                "label": ("color", "white"),
                "grouper_key": "color",
                "grouper_value": "white",
            },
            {
                "dataset_name": "delete later",
                "datum_uid": "uid2",
                "datum_id": -8411940843701065439,
                "datum_metadata": {
                    "height": 128,
                    "width": 256,
                    "md1": "md1-val0",
                    "md2": "md1-val2",
                },
                "annotation_id": 7499720668016145718,
                "annotation_metadata": None,
                "bounding_box": None,
                "raster": None,
                "embedding": None,
                "polygon": None,
                "is_instance": None,
                "label_key": "color",
                "label_value": "blue",
                "score": 0.1,
                "label_id": -4127427154085111908,
                "id": 6376790152767730567,
                "mapped_prediction_label_keys": "color",
                "label": ("color", "blue"),
                "grouper_key": "color",
                "grouper_value": "blue",
            },
            {
                "dataset_name": "delete later",
                "datum_uid": "uid2",
                "datum_id": -8411940843701065439,
                "datum_metadata": {
                    "height": 128,
                    "width": 256,
                    "md1": "md1-val0",
                    "md2": "md1-val2",
                },
                "annotation_id": 7499720668016145718,
                "annotation_metadata": None,
                "bounding_box": None,
                "raster": None,
                "embedding": None,
                "polygon": None,
                "is_instance": None,
                "label_key": "color",
                "label_value": "black",
                "score": 0.3,
                "label_id": -5292453587279810103,
                "id": 7023758392816762513,
                "mapped_prediction_label_keys": "color",
                "label": ("color", "black"),
                "grouper_key": "color",
                "grouper_value": "black",
            },
            {
                "dataset_name": "delete later",
                "datum_uid": "uid3",
                "datum_id": -2265528102457502931,
                "datum_metadata": {
                    "height": 128,
                    "width": 256,
                    "md1": "md1-val0",
                    "md2": "md1-val0",
                },
                "annotation_id": 4348440930043552140,
                "annotation_metadata": None,
                "bounding_box": None,
                "raster": None,
                "embedding": None,
                "polygon": None,
                "is_instance": None,
                "label_key": "animal",
                "label_value": "dog",
                "score": 0.75,
                "label_id": -1804361582153801946,
                "id": 2109915554097816409,
                "mapped_prediction_label_keys": "class",
                "label": ("animal", "dog"),
                "grouper_key": "class",
                "grouper_value": "mammal",
            },
            {
                "dataset_name": "delete later",
                "datum_uid": "uid3",
                "datum_id": -2265528102457502931,
                "datum_metadata": {
                    "height": 128,
                    "width": 256,
                    "md1": "md1-val0",
                    "md2": "md1-val0",
                },
                "annotation_id": 4348440930043552140,
                "annotation_metadata": None,
                "bounding_box": None,
                "raster": None,
                "embedding": None,
                "polygon": None,
                "is_instance": None,
                "label_key": "animal",
                "label_value": "cat",
                "score": 0.1,
                "label_id": -4720233526095501343,
                "id": -7234886842398502296,
                "mapped_prediction_label_keys": "class",
                "label": ("animal", "cat"),
                "grouper_key": "class",
                "grouper_value": "mammal",
            },
            {
                "dataset_name": "delete later",
                "datum_uid": "uid3",
                "datum_id": -2265528102457502931,
                "datum_metadata": {
                    "height": 128,
                    "width": 256,
                    "md1": "md1-val0",
                    "md2": "md1-val0",
                },
                "annotation_id": 4348440930043552140,
                "annotation_metadata": None,
                "bounding_box": None,
                "raster": None,
                "embedding": None,
                "polygon": None,
                "is_instance": None,
                "label_key": "animal",
                "label_value": "bird",
                "score": 0.15,
                "label_id": -3283720280595522641,
                "id": 1110595858053279959,
                "mapped_prediction_label_keys": "class",
                "label": ("animal", "bird"),
                "grouper_key": "other_class",
                "grouper_value": "avian",
            },
            {
                "dataset_name": "delete later",
                "datum_uid": "uid3",
                "datum_id": -2265528102457502931,
                "datum_metadata": {
                    "height": 128,
                    "width": 256,
                    "md1": "md1-val0",
                    "md2": "md1-val0",
                },
                "annotation_id": 4348440930043552140,
                "annotation_metadata": None,
                "bounding_box": None,
                "raster": None,
                "embedding": None,
                "polygon": None,
                "is_instance": None,
                "label_key": "color",
                "label_value": "white",
                "score": 1.0,
                "label_id": 5280415162891465313,
                "id": 8226781192373612358,
                "mapped_prediction_label_keys": "color",
                "label": ("color", "white"),
                "grouper_key": "color",
                "grouper_value": "white",
            },
            {
                "dataset_name": "delete later",
                "datum_uid": "uid3",
                "datum_id": -2265528102457502931,
                "datum_metadata": {
                    "height": 128,
                    "width": 256,
                    "md1": "md1-val0",
                    "md2": "md1-val0",
                },
                "annotation_id": 4348440930043552140,
                "annotation_metadata": None,
                "bounding_box": None,
                "raster": None,
                "embedding": None,
                "polygon": None,
                "is_instance": None,
                "label_key": "color",
                "label_value": "red",
                "score": 0.0,
                "label_id": -519495577997781294,
                "id": -1930456292948739198,
                "mapped_prediction_label_keys": "color",
                "label": ("color", "red"),
                "grouper_key": "color",
                "grouper_value": "red",
            },
            {
                "dataset_name": "delete later",
                "datum_uid": "uid3",
                "datum_id": -2265528102457502931,
                "datum_metadata": {
                    "height": 128,
                    "width": 256,
                    "md1": "md1-val0",
                    "md2": "md1-val0",
                },
                "annotation_id": 4348440930043552140,
                "annotation_metadata": None,
                "bounding_box": None,
                "raster": None,
                "embedding": None,
                "polygon": None,
                "is_instance": None,
                "label_key": "color",
                "label_value": "blue",
                "score": 0.0,
                "label_id": 6597917751396615534,
                "id": 5770081132013712295,
                "mapped_prediction_label_keys": "color",
                "label": ("color", "blue"),
                "grouper_key": "color",
                "grouper_value": "blue",
            },
            {
                "dataset_name": "delete later",
                "datum_uid": "uid3",
                "datum_id": -2265528102457502931,
                "datum_metadata": {
                    "height": 128,
                    "width": 256,
                    "md1": "md1-val0",
                    "md2": "md1-val0",
                },
                "annotation_id": 4348440930043552140,
                "annotation_metadata": None,
                "bounding_box": None,
                "raster": None,
                "embedding": None,
                "polygon": None,
                "is_instance": None,
                "label_key": "color",
                "label_value": "black",
                "score": 0.0,
                "label_id": 1350538389931074891,
                "id": 9216624913651577421,
                "mapped_prediction_label_keys": "color",
                "label": ("color", "black"),
                "grouper_key": "color",
                "grouper_value": "black",
            },
            {
                "dataset_name": "delete later",
                "datum_uid": "uid4",
                "datum_id": -4389124420839664731,
                "datum_metadata": {
                    "height": 128,
                    "width": 256,
                    "md1": "md1-val1",
                    "md2": "md1-val1",
                },
                "annotation_id": -3609568981720823102,
                "annotation_metadata": None,
                "bounding_box": None,
                "raster": None,
                "embedding": None,
                "polygon": None,
                "is_instance": None,
                "label_key": "animal",
                "label_value": "cat",
                "score": 1.0,
                "label_id": 7155939162232491288,
                "id": 8865373275147915155,
                "mapped_prediction_label_keys": "class",
                "label": ("animal", "cat"),
                "grouper_key": "class",
                "grouper_value": "mammal",
            },
            {
                "dataset_name": "delete later",
                "datum_uid": "uid4",
                "datum_id": -4389124420839664731,
                "datum_metadata": {
                    "height": 128,
                    "width": 256,
                    "md1": "md1-val1",
                    "md2": "md1-val1",
                },
                "annotation_id": -3609568981720823102,
                "annotation_metadata": None,
                "bounding_box": None,
                "raster": None,
                "embedding": None,
                "polygon": None,
                "is_instance": None,
                "label_key": "animal",
                "label_value": "dog",
                "score": 0.0,
                "label_id": -8923497484890863398,
                "id": 7811596003484809003,
                "mapped_prediction_label_keys": "class",
                "label": ("animal", "dog"),
                "grouper_key": "class",
                "grouper_value": "mammal",
            },
            {
                "dataset_name": "delete later",
                "datum_uid": "uid4",
                "datum_id": -4389124420839664731,
                "datum_metadata": {
                    "height": 128,
                    "width": 256,
                    "md1": "md1-val1",
                    "md2": "md1-val1",
                },
                "annotation_id": -3609568981720823102,
                "annotation_metadata": None,
                "bounding_box": None,
                "raster": None,
                "embedding": None,
                "polygon": None,
                "is_instance": None,
                "label_key": "animal",
                "label_value": "bird",
                "score": 0.0,
                "label_id": 8183125692418530608,
                "id": -603291948951724467,
                "mapped_prediction_label_keys": "class",
                "label": ("animal", "bird"),
                "grouper_key": "other_class",
                "grouper_value": "avian",
            },
            {
                "dataset_name": "delete later",
                "datum_uid": "uid4",
                "datum_id": -4389124420839664731,
                "datum_metadata": {
                    "height": 128,
                    "width": 256,
                    "md1": "md1-val1",
                    "md2": "md1-val1",
                },
                "annotation_id": -3609568981720823102,
                "annotation_metadata": None,
                "bounding_box": None,
                "raster": None,
                "embedding": None,
                "polygon": None,
                "is_instance": None,
                "label_key": "color",
                "label_value": "red",
                "score": 0.8,
                "label_id": 1005923488131372002,
                "id": 2186370402320236011,
                "mapped_prediction_label_keys": "color",
                "label": ("color", "red"),
                "grouper_key": "color",
                "grouper_value": "red",
            },
            {
                "dataset_name": "delete later",
                "datum_uid": "uid4",
                "datum_id": -4389124420839664731,
                "datum_metadata": {
                    "height": 128,
                    "width": 256,
                    "md1": "md1-val1",
                    "md2": "md1-val1",
                },
                "annotation_id": -3609568981720823102,
                "annotation_metadata": None,
                "bounding_box": None,
                "raster": None,
                "embedding": None,
                "polygon": None,
                "is_instance": None,
                "label_key": "color",
                "label_value": "white",
                "score": 0.0,
                "label_id": -6581901677798598125,
                "id": 5980951779669100519,
                "mapped_prediction_label_keys": "color",
                "label": ("color", "white"),
                "grouper_key": "color",
                "grouper_value": "white",
            },
            {
                "dataset_name": "delete later",
                "datum_uid": "uid4",
                "datum_id": -4389124420839664731,
                "datum_metadata": {
                    "height": 128,
                    "width": 256,
                    "md1": "md1-val1",
                    "md2": "md1-val1",
                },
                "annotation_id": -3609568981720823102,
                "annotation_metadata": None,
                "bounding_box": None,
                "raster": None,
                "embedding": None,
                "polygon": None,
                "is_instance": None,
                "label_key": "color",
                "label_value": "blue",
                "score": 0.2,
                "label_id": -3960395303314501711,
                "id": -2623103473497724690,
                "mapped_prediction_label_keys": "color",
                "label": ("color", "blue"),
                "grouper_key": "color",
                "grouper_value": "blue",
            },
            {
                "dataset_name": "delete later",
                "datum_uid": "uid4",
                "datum_id": -4389124420839664731,
                "datum_metadata": {
                    "height": 128,
                    "width": 256,
                    "md1": "md1-val1",
                    "md2": "md1-val1",
                },
                "annotation_id": -3609568981720823102,
                "annotation_metadata": None,
                "bounding_box": None,
                "raster": None,
                "embedding": None,
                "polygon": None,
                "is_instance": None,
                "label_key": "color",
                "label_value": "black",
                "score": 0.0,
                "label_id": 1350538389931074891,
                "id": 1948160906536205683,
                "mapped_prediction_label_keys": "color",
                "label": ("color", "black"),
                "grouper_key": "color",
                "grouper_value": "black",
            },
            {
                "dataset_name": "delete later",
                "datum_uid": "uid5",
                "datum_id": 5314927723853009775,
                "datum_metadata": {
                    "height": 128,
                    "width": 256,
                    "md1": "md1-val0",
                    "md2": "md1-val2",
                },
                "annotation_id": 2454836867465092903,
                "annotation_metadata": None,
                "bounding_box": None,
                "raster": None,
                "embedding": None,
                "polygon": None,
                "is_instance": None,
                "label_key": "animal",
                "label_value": "cat",
                "score": 0.4,
                "label_id": -5278394517120365112,
                "id": 8196690759347808946,
                "mapped_prediction_label_keys": "class",
                "label": ("animal", "cat"),
                "grouper_key": "class",
                "grouper_value": "mammal",
            },
            {
                "dataset_name": "delete later",
                "datum_uid": "uid5",
                "datum_id": 5314927723853009775,
                "datum_metadata": {
                    "height": 128,
                    "width": 256,
                    "md1": "md1-val0",
                    "md2": "md1-val2",
                },
                "annotation_id": 2454836867465092903,
                "annotation_metadata": None,
                "bounding_box": None,
                "raster": None,
                "embedding": None,
                "polygon": None,
                "is_instance": None,
                "label_key": "animal",
                "label_value": "dog",
                "score": 0.4,
                "label_id": -3672411415008402703,
                "id": -1938030899200555758,
                "mapped_prediction_label_keys": "class",
                "label": ("animal", "dog"),
                "grouper_key": "class",
                "grouper_value": "mammal",
            },
            {
                "dataset_name": "delete later",
                "datum_uid": "uid5",
                "datum_id": 5314927723853009775,
                "datum_metadata": {
                    "height": 128,
                    "width": 256,
                    "md1": "md1-val0",
                    "md2": "md1-val2",
                },
                "annotation_id": 2454836867465092903,
                "annotation_metadata": None,
                "bounding_box": None,
                "raster": None,
                "embedding": None,
                "polygon": None,
                "is_instance": None,
                "label_key": "animal",
                "label_value": "bird",
                "score": 0.2,
                "label_id": -4720668901151276709,
                "id": -375807178484672075,
                "mapped_prediction_label_keys": "class",
                "label": ("animal", "bird"),
                "grouper_key": "other_class",
                "grouper_value": "avian",
            },
            {
                "dataset_name": "delete later",
                "datum_uid": "uid5",
                "datum_id": 5314927723853009775,
                "datum_metadata": {
                    "height": 128,
                    "width": 256,
                    "md1": "md1-val0",
                    "md2": "md1-val2",
                },
                "annotation_id": 2454836867465092903,
                "annotation_metadata": None,
                "bounding_box": None,
                "raster": None,
                "embedding": None,
                "polygon": None,
                "is_instance": None,
                "label_key": "color",
                "label_value": "red",
                "score": 0.9,
                "label_id": -2571710428146614475,
                "id": 7302285613830353470,
                "mapped_prediction_label_keys": "color",
                "label": ("color", "red"),
                "grouper_key": "color",
                "grouper_value": "red",
            },
            {
                "dataset_name": "delete later",
                "datum_uid": "uid5",
                "datum_id": 5314927723853009775,
                "datum_metadata": {
                    "height": 128,
                    "width": 256,
                    "md1": "md1-val0",
                    "md2": "md1-val2",
                },
                "annotation_id": 2454836867465092903,
                "annotation_metadata": None,
                "bounding_box": None,
                "raster": None,
                "embedding": None,
                "polygon": None,
                "is_instance": None,
                "label_key": "color",
                "label_value": "white",
                "score": 0.06,
                "label_id": 6423587877188027700,
                "id": -5213005280939427276,
                "mapped_prediction_label_keys": "color",
                "label": ("color", "white"),
                "grouper_key": "color",
                "grouper_value": "white",
            },
            {
                "dataset_name": "delete later",
                "datum_uid": "uid5",
                "datum_id": 5314927723853009775,
                "datum_metadata": {
                    "height": 128,
                    "width": 256,
                    "md1": "md1-val0",
                    "md2": "md1-val2",
                },
                "annotation_id": 2454836867465092903,
                "annotation_metadata": None,
                "bounding_box": None,
                "raster": None,
                "embedding": None,
                "polygon": None,
                "is_instance": None,
                "label_key": "color",
                "label_value": "blue",
                "score": 0.01,
                "label_id": -7515229394567381620,
                "id": 3837015023039237314,
                "mapped_prediction_label_keys": "color",
                "label": ("color", "blue"),
                "grouper_key": "color",
                "grouper_value": "blue",
            },
            {
                "dataset_name": "delete later",
                "datum_uid": "uid5",
                "datum_id": 5314927723853009775,
                "datum_metadata": {
                    "height": 128,
                    "width": 256,
                    "md1": "md1-val0",
                    "md2": "md1-val2",
                },
                "annotation_id": 2454836867465092903,
                "annotation_metadata": None,
                "bounding_box": None,
                "raster": None,
                "embedding": None,
                "polygon": None,
                "is_instance": None,
                "label_key": "color",
                "label_value": "black",
                "score": 0.03,
                "label_id": -824168874021550241,
                "id": 551917309394979383,
                "mapped_prediction_label_keys": "color",
                "label": ("color", "black"),
                "grouper_key": "color",
                "grouper_value": "black",
            },
        ]
    )
