import pytest
from sqlalchemy.orm import Session

from velour_api import crud, enums, exceptions, schemas


@pytest.fixture
def groundtruth_detections(img1: schemas.Datum) -> list[schemas.GroundTruth]:
    return [
        schemas.GroundTruth(
            datum=img1,
            annotations=[
                schemas.Annotation(
                    task_type=enums.TaskType.DETECTION,
                    labels=[
                        schemas.Label(key="k1", value="v1"),
                        schemas.Label(key="k2", value="v2"),
                    ],
                    metadata={},
                    bounding_box=schemas.BoundingBox(
                        polygon=schemas.BasicPolygon(
                            points=[
                                schemas.Point(x=10, y=20),
                                schemas.Point(x=10, y=30),
                                schemas.Point(x=20, y=30),
                                schemas.Point(
                                    x=20, y=20
                                ),  # removed repeated first point
                            ]
                        )
                    ),
                ),
                schemas.Annotation(
                    task_type=enums.TaskType.DETECTION,
                    labels=[schemas.Label(key="k2", value="v2")],
                    metadata={},
                    bounding_box=schemas.BoundingBox(
                        polygon=schemas.BasicPolygon(
                            points=[
                                schemas.Point(x=10, y=20),
                                schemas.Point(x=10, y=30),
                                schemas.Point(x=20, y=30),
                                schemas.Point(
                                    x=20, y=20
                                ),  # removed repeated first point
                            ]
                        )
                    ),
                ),
            ],
        )
    ]


@pytest.fixture
def prediction_detections(
    model_name: str, img1: schemas.Datum
) -> list[schemas.Prediction]:
    return [
        schemas.Prediction(
            model=model_name,
            datum=img1,
            annotations=[
                schemas.Annotation(
                    task_type=enums.TaskType.DETECTION,
                    labels=[
                        schemas.Label(key="k1", value="v1", score=0.6),
                        schemas.Label(key="k1", value="v2", score=0.4),
                        schemas.Label(key="k2", value="v1", score=0.8),
                        schemas.Label(key="k2", value="v2", score=0.2),
                    ],
                    bounding_box=schemas.BoundingBox(
                        polygon=schemas.BasicPolygon(
                            points=[
                                schemas.Point(x=107, y=207),
                                schemas.Point(x=107, y=307),
                                schemas.Point(x=207, y=307),
                                schemas.Point(x=207, y=207),
                            ]
                        )
                    ),
                ),
                schemas.Annotation(
                    task_type=enums.TaskType.DETECTION,
                    labels=[
                        schemas.Label(key="k2", value="v1", score=0.1),
                        schemas.Label(key="k2", value="v2", score=0.9),
                    ],
                    bounding_box=schemas.BoundingBox(
                        polygon=schemas.BasicPolygon(
                            points=[
                                schemas.Point(x=107, y=207),
                                schemas.Point(x=107, y=307),
                                schemas.Point(x=207, y=307),
                                schemas.Point(x=207, y=207),
                            ]
                        )
                    ),
                ),
            ],
        )
    ]


@pytest.fixture
def dataset_names():
    return ["dataset1", "dataset2"]


@pytest.fixture
def model_names():
    return ["model1", "model2"]


@pytest.fixture
def dataset_model_create(
    db: Session,
    groundtruth_detections: list[schemas.GroundTruth],
    prediction_detections: list[schemas.Prediction],
    dataset_names: list[str],
    model_names: list[str],
):
    # create dataset1
    crud.create_dataset(
        db=db,
        dataset=schemas.Dataset(name=dataset_names[0]),
    )
    for gt in groundtruth_detections:
        gt.datum.dataset = dataset_names[0]
        crud.create_groundtruth(db=db, groundtruth=gt)
    crud.finalize(db=db, dataset_name=dataset_names[0])

    # Create model1
    crud.create_model(db=db, model=schemas.Model(name=model_names[0]))

    # Link model1 to dataset1
    for pd in prediction_detections:
        pd.model = model_names[0]
        pd.datum.dataset = dataset_names[0]
        crud.create_prediction(db=db, prediction=pd)

    # Finalize model1 over dataset1
    crud.finalize(
        db=db,
        dataset_name=dataset_names[0],
        model_name=model_names[0],
    )

    yield

    # clean up
    crud.delete(db=db, model_name=model_names[0])
    crud.delete(db=db, dataset_name=dataset_names[0])


def test_get_dataset(
    db: Session,
    dataset_name: str,
):
    with pytest.raises(exceptions.DatasetDoesNotExistError) as exc_info:
        crud.get_dataset(db=db, dataset_name=dataset_name)
    assert "does not exist" in str(exc_info)

    crud.create_dataset(db=db, dataset=schemas.Dataset(name=dataset_name))

    dset = crud.get_dataset(db=db, dataset_name=dataset_name)
    assert dset.name == dataset_name


def test_get_model(
    db: Session,
    model_name: str,
):
    with pytest.raises(exceptions.ModelDoesNotExistError) as exc_info:
        crud.get_model(db=db, model_name=model_name)
    assert "does not exist" in str(exc_info)

    crud.create_model(db=db, model=schemas.Model(name=model_name))
    model = crud.get_model(db=db, model_name=model_name)
    assert model.name == model_name


def test_get_all_labels(
    db: Session, dataset_name: str, groundtruth_detections: schemas.GroundTruth
):
    crud.create_dataset(db=db, dataset=schemas.Dataset(name=dataset_name))

    for gt in groundtruth_detections:
        crud.create_groundtruth(db=db, groundtruth=gt)

    labels = crud.get_all_labels(db=db)

    assert len(labels) == 2
    assert set([(label.key, label.value) for label in labels]) == set(
        [("k1", "v1"), ("k2", "v2")]
    )


def test_get_labels_from_dataset(
    db: Session,
    dataset_names: list[str],
    dataset_model_create,
):
    # Test get all from dataset 1
    ds1 = crud.get_dataset_labels(
        db=db,
        filters=schemas.Filter(
            dataset_names=[dataset_names[0]],
        ),
    )
    assert len(ds1) == 2
    assert schemas.Label(key="k1", value="v1") in ds1
    assert schemas.Label(key="k2", value="v2") in ds1

    # NEGATIVE - Test filter by task type
    ds1 = crud.get_dataset_labels(
        db=db,
        filters=schemas.Filter(
            dataset_names=[dataset_names[0]],
            task_types=[
                enums.TaskType.CLASSIFICATION,
                enums.TaskType.SEGMENTATION,
            ],
        ),
    )
    assert ds1 == []

    # POSITIVE - Test filter by task type
    ds1 = crud.get_dataset_labels(
        db=db,
        filters=schemas.Filter(
            dataset_names=[dataset_names[0]],
            task_types=[enums.TaskType.DETECTION],
        ),
    )
    assert len(ds1) == 2
    assert schemas.Label(key="k1", value="v1") in ds1
    assert schemas.Label(key="k2", value="v2") in ds1

    # NEGATIVE - Test filter by annotation type
    ds1 = crud.get_dataset_labels(
        db=db,
        filters=schemas.Filter(
            dataset_names=[dataset_names[0]],
            annotation_types=[
                enums.AnnotationType.POLYGON,
                enums.AnnotationType.MULTIPOLYGON,
                enums.AnnotationType.RASTER,
            ],
        ),
    )
    assert ds1 == []

    # POSITIVE - Test filter by annotation type
    ds1 = crud.get_dataset_labels(
        db=db,
        filters=schemas.Filter(
            dataset_names=[dataset_names[0]],
            annotation_types=[
                enums.AnnotationType.BOX,
            ],
        ),
    )
    assert len(ds1) == 2
    assert schemas.Label(key="k1", value="v1") in ds1
    assert schemas.Label(key="k2", value="v2") in ds1


def test_get_labels_from_model(
    db: Session,
    model_names: list[str],
    dataset_model_create,
):
    # Test get all labels from model 1
    md1 = crud.get_model_labels(
        db=db,
        filters=schemas.Filter(
            models_names=[model_names[0]],
        ),
    )
    assert len(md1) == 4
    assert schemas.Label(key="k1", value="v1") in md1
    assert schemas.Label(key="k1", value="v2") in md1
    assert schemas.Label(key="k2", value="v1") in md1
    assert schemas.Label(key="k2", value="v2") in md1

    # Test get all but polygon labels from model 1
    md1 = crud.get_model_labels(
        db=db,
        filters=schemas.Filter(
            models_names=[model_names[0]],
            task_types=[enums.TaskType.CLASSIFICATION],
        ),
    )
    assert md1 == []

    # Test get only polygon labels from model 1
    md1 = crud.get_model_labels(
        db=db,
        filters=schemas.Filter(
            models_names=[model_names[0]],
            annotation_types=[enums.AnnotationType.BOX],
        ),
    )
    assert len(md1) == 4
    assert schemas.Label(key="k1", value="v1") in md1
    assert schemas.Label(key="k1", value="v2") in md1
    assert schemas.Label(key="k2", value="v1") in md1
    assert schemas.Label(key="k2", value="v2") in md1


def test_get_joint_labels(
    db: Session,
    dataset_names: list[str],
    model_names: list[str],
    dataset_model_create,
):
    # Test get joint labels from dataset 1 and model 1
    assert set(
        crud.get_joint_labels(
            db=db,
            dataset_name=dataset_names[0],
            model_name=model_names[0],
            task_types=[enums.TaskType.DETECTION],
            groundtruth_type=enums.AnnotationType.BOX,
            prediction_type=enums.AnnotationType.BOX,
        )
    ) == set(
        [
            schemas.Label(key="k1", value="v1"),
            schemas.Label(key="k2", value="v2"),
        ]
    )
