import pandas as pd
import pytest
from valor_core import schemas


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
