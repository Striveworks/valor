import io
import math
from base64 import b64decode, b64encode

import numpy as np
import pytest
from geoalchemy2.functions import ST_AsText, ST_Count, ST_Polygon
from PIL import Image
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from velour_api import crud, enums, exceptions, schemas
from velour_api.backend import models


def _bytes_to_pil(b: bytes) -> Image.Image:
    f = io.BytesIO(b)
    img = Image.open(f)
    return img


def _np_to_bytes(arr: np.ndarray) -> bytes:
    f = io.BytesIO()
    Image.fromarray(arr).save(f, format="PNG")
    f.seek(0)
    return f.read()


def _check_db_empty(db: Session):
    for model_cls in [
        models.Label,
        models.GroundTruth,
        models.Prediction,
        models.Annotation,
        models.Datum,
        models.Model,
        models.Dataset,
        models.Evaluation,
        models.Metric,
        models.ConfusionMatrix,
    ]:
        assert db.scalar(select(func.count(model_cls.id))) == 0


@pytest.fixture
def poly_without_hole() -> schemas.Polygon:
    # should have area 45.5
    return schemas.Polygon(
        boundary=schemas.BasicPolygon(
            points=[
                schemas.Point(x=14, y=10),
                schemas.Point(x=19, y=7),
                schemas.Point(x=21, y=2),
                schemas.Point(x=12, y=2),
            ]
        )
    )


@pytest.fixture
def poly_with_hole() -> schemas.Polygon:
    # should have area 100 - 8 = 92
    return schemas.Polygon(
        boundary=schemas.BasicPolygon(
            points=[
                schemas.Point(x=0, y=10),
                schemas.Point(x=10, y=10),
                schemas.Point(x=10, y=0),
                schemas.Point(x=0, y=0),
            ]
        ),
        holes=[
            schemas.BasicPolygon(
                points=[
                    schemas.Point(x=2, y=4),
                    schemas.Point(x=2, y=8),
                    schemas.Point(x=6, y=4),
                ]
            ),
        ],
    )


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
            model_name=model_name,
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
def groundtruth_instance_segmentations(
    poly_with_hole: schemas.BasicPolygon,
    poly_without_hole: schemas.BasicPolygon,
    img1: schemas.Datum,
    img2: schemas.Datum,
) -> list[schemas.GroundTruth]:
    return [
        schemas.GroundTruth(
            datum=img1,
            annotations=[
                schemas.Annotation(
                    task_type=enums.TaskType.DETECTION,
                    labels=[schemas.Label(key="k1", value="v1")],
                    polygon=poly_with_hole,
                ),
            ],
        ),
        schemas.GroundTruth(
            datum=img2,
            annotations=[
                schemas.Annotation(
                    task_type=enums.TaskType.DETECTION,
                    labels=[schemas.Label(key="k1", value="v1")],
                    polygon=poly_without_hole,
                ),
                schemas.Annotation(
                    task_type=enums.TaskType.DETECTION,
                    labels=[schemas.Label(key="k3", value="v3")],
                    polygon=poly_without_hole,
                ),
                schemas.Annotation(
                    task_type=enums.TaskType.DETECTION,
                    labels=[schemas.Label(key="k1", value="v1")],
                    multipolygon=schemas.MultiPolygon(
                        polygons=[poly_with_hole, poly_without_hole],
                    ),
                ),
            ],
        ),
    ]


@pytest.fixture
def prediction_instance_segmentations(
    model_name: str,
    img1_pred_mask_bytes1: bytes,
    img1: schemas.Datum,
) -> list[schemas.Prediction]:
    b64_mask1 = b64encode(img1_pred_mask_bytes1).decode()

    return [
        schemas.Prediction(
            model_name=model_name,
            datum=img1,
            annotations=[
                schemas.Annotation(
                    task_type=enums.TaskType.DETECTION,
                    labels=[
                        schemas.Label(key="k1", value="v1", score=0.43),
                        schemas.Label(key="k1", value="v2", score=0.57),
                    ],
                    raster=schemas.Raster(
                        mask=b64_mask1,
                    ),
                ),
                schemas.Annotation(
                    task_type=enums.TaskType.DETECTION,
                    labels=[
                        schemas.Label(key="k2", value="v1", score=0.03),
                        schemas.Label(key="k2", value="v2", score=0.97),
                    ],
                    raster=schemas.Raster(
                        mask=b64_mask1,
                    ),
                ),
                schemas.Annotation(
                    task_type=enums.TaskType.DETECTION,
                    labels=[
                        schemas.Label(key="k2", value="v1", score=0.26),
                        schemas.Label(key="k2", value="v2", score=0.74),
                    ],
                    raster=schemas.Raster(
                        mask=b64_mask1,
                    ),
                ),
                schemas.Annotation(
                    task_type=enums.TaskType.DETECTION,
                    labels=[
                        schemas.Label(key="k2", value="v1", score=0.86),
                        schemas.Label(key="k2", value="v2", score=0.14),
                    ],
                    raster=schemas.Raster(
                        mask=b64_mask1,
                    ),
                ),
            ],
        )
    ]


@pytest.fixture
def gt_clfs_create(
    img1: schemas.Datum,
    img2: schemas.Datum,
) -> list[schemas.GroundTruth]:
    return [
        schemas.GroundTruth(
            datum=img1,
            annotations=[
                schemas.Annotation(
                    task_type=enums.TaskType.CLASSIFICATION,
                    labels=[
                        schemas.Label(key="k1", value="v1"),
                        schemas.Label(key="k2", value="v2"),
                    ],
                ),
            ],
        ),
        schemas.GroundTruth(
            datum=img2,
            annotations=[
                schemas.Annotation(
                    task_type=enums.TaskType.CLASSIFICATION,
                    labels=[schemas.Label(key="k2", value="v3")],
                ),
            ],
        ),
    ]


@pytest.fixture
def pred_clfs_create(
    model_name: str,
    img1: schemas.Datum,
    img2: schemas.Datum,
) -> list[schemas.Prediction]:
    return [
        schemas.Prediction(
            model_name=model_name,
            datum=img1,
            annotations=[
                schemas.Annotation(
                    task_type=enums.TaskType.CLASSIFICATION,
                    labels=[
                        schemas.Label(key="k1", value="v1", score=0.2),
                        schemas.Label(key="k1", value="v2", score=0.8),
                        schemas.Label(key="k4", value="v4", score=1.0),
                    ],
                ),
            ],
        ),
        schemas.Prediction(
            model_name=model_name,
            datum=img2,
            annotations=[
                schemas.Annotation(
                    task_type=enums.TaskType.CLASSIFICATION,
                    labels=[
                        schemas.Label(key="k2", value="v2", score=1.0),
                        schemas.Label(key="k3", value="v3", score=0.87),
                        schemas.Label(key="k3", value="v0", score=0.13),
                    ],
                ),
            ],
        ),
    ]


@pytest.fixture
def model_names():
    return ["model1", "model2"]


def test_create_and_get_datasets(
    db: Session,
    dataset_name: str,
    model_name: str,
):
    # Create dataset
    crud.create_dataset(db=db, dataset=schemas.Dataset(name=dataset_name))

    all_datasets = db.scalars(select(models.Dataset)).all()
    assert len(all_datasets) == 1
    assert all_datasets[0].name == dataset_name

    with pytest.raises(exceptions.DatasetAlreadyExistsError) as exc_info:
        crud.create_dataset(
            db=db,
            dataset=schemas.Dataset(name=dataset_name),
        )
    assert "already exists" in str(exc_info)

    crud.create_dataset(
        db=db,
        dataset=schemas.Dataset(name="other_dataset"),
    )
    datasets = crud.get_datasets(db=db)
    assert len(datasets) == 2
    assert set([d.name for d in datasets]) == {dataset_name, "other_dataset"}


def test_create_and_get_models(
    db: Session,
    model_name: str,
):
    crud.create_model(db=db, model=schemas.Model(name=model_name))

    all_models = db.scalars(select(models.Model)).all()
    assert len(all_models) == 1
    assert all_models[0].name == model_name

    with pytest.raises(exceptions.ModelAlreadyExistsError) as exc_info:
        crud.create_model(db=db, model=schemas.Model(name=model_name))
    assert "already exists" in str(exc_info)

    crud.create_model(db=db, model=schemas.Model(name="other_model"))
    db_models = crud.get_models(db=db)
    assert len(db_models) == 2
    assert set([m.name for m in db_models]) == {model_name, "other_model"}


def test_create_detection_ground_truth_and_delete_dataset(
    db: Session,
    dataset_name: str,
    groundtruth_detections: list[schemas.GroundTruth],
):
    # sanity check nothing in db
    _check_db_empty(db=db)

    crud.create_dataset(db=db, dataset=schemas.Dataset(name=dataset_name))

    for gt in groundtruth_detections:
        crud.create_groundtruth(db=db, groundtruth=gt)

    assert db.scalar(func.count(models.Annotation.id)) == 2
    assert db.scalar(func.count(models.Datum.id)) == 1
    assert db.scalar(func.count(models.GroundTruth.id)) == 3
    assert db.scalar(func.count(models.Label.id)) == 2

    # verify we get the same dets back
    for gt in groundtruth_detections:
        new_gt = crud.get_groundtruth(
            db=db, dataset_name=gt.datum.dataset_name, datum_uid=gt.datum.uid
        )
        assert gt.datum.uid == new_gt.datum.uid
        assert gt.datum.dataset_name == new_gt.datum.dataset_name
        for metadatum in gt.datum.metadata:
            assert metadatum in new_gt.datum.metadata

        for gta, new_gta in zip(gt.annotations, new_gt.annotations):
            assert set(gta.labels) == set(new_gta.labels)
            assert gta.bounding_box == new_gta.bounding_box

    # finalize to free job state
    crud.finalize(db=db, dataset_name=dataset_name)

    # delete dataset and check the cascade worked
    crud.delete(db=db, dataset_name=dataset_name)
    for model_cls in [
        models.Dataset,
        models.Datum,
        models.GroundTruth,
        models.Annotation,
    ]:
        assert db.scalar(func.count(model_cls.id)) == 0

    # make sure labels are still there`
    assert db.scalar(func.count(models.Label.id)) == 2


def test_create_detection_prediction_and_delete_model(
    db: Session,
    dataset_name: str,
    model_name: str,
    prediction_detections: list[schemas.Prediction],
    groundtruth_detections: list[schemas.GroundTruth],
):
    # check this gives an error since the model hasn't been added yet
    with pytest.raises(exceptions.DatasetDoesNotExistError) as exc_info:
        for pd in prediction_detections:
            crud.create_prediction(db=db, prediction=pd)
    assert "does not exist" in str(exc_info)

    # create dataset, add images, and add predictions
    crud.create_dataset(db=db, dataset=schemas.Dataset(name=dataset_name))

    for gt in groundtruth_detections:
        crud.create_groundtruth(db=db, groundtruth=gt)

    # check this gives an error since the model hasn't been created yet
    with pytest.raises(exceptions.ModelDoesNotExistError) as exc_info:
        for pd in prediction_detections:
            crud.create_prediction(db=db, prediction=pd)
    assert "does not exist" in str(exc_info)

    # finalize dataset
    crud.finalize(db=db, dataset_name=dataset_name)

    # check this gives an error since the model hasn't been added yet
    with pytest.raises(exceptions.ModelDoesNotExistError) as exc_info:
        for pd in prediction_detections:
            crud.create_prediction(db=db, prediction=pd)
    assert "does not exist" in str(exc_info)

    # create model
    crud.create_model(db=db, model=schemas.Model(name=model_name))
    for pd in prediction_detections:
        crud.create_prediction(db=db, prediction=pd)

    # check db has the added predictions
    assert db.scalar(func.count(models.Annotation.id)) == 4
    assert db.scalar(func.count(models.Datum.id)) == 1
    assert db.scalar(func.count(models.GroundTruth.id)) == 3
    assert db.scalar(func.count(models.Prediction.id)) == 6
    assert db.scalar(func.count(models.Label.id)) == 4

    # finalize
    crud.finalize(db=db, dataset_name=dataset_name, model_name=model_name)

    # delete model and check all detections from it are gone
    crud.delete(db=db, model_name=model_name)
    assert db.scalar(func.count(models.Annotation.id)) == 2
    assert db.scalar(func.count(models.Datum.id)) == 1
    assert db.scalar(func.count(models.GroundTruth.id)) == 3
    assert db.scalar(func.count(models.Prediction.id)) == 0
    assert db.scalar(func.count(models.Label.id)) == 4


def test_create_detections_as_bbox_or_poly(
    db: Session, dataset_name: str, img1: schemas.Datum
):
    xmin, ymin, xmax, ymax = 50, 70, 120, 300

    det1 = schemas.Annotation(
        task_type=enums.TaskType.DETECTION,
        labels=[schemas.Label(key="k", value="v")],
        polygon=schemas.Polygon(
            boundary=schemas.BasicPolygon(
                points=[
                    schemas.Point(x=xmin, y=ymin),
                    schemas.Point(x=xmax, y=ymin),
                    schemas.Point(x=xmax, y=ymax),
                    schemas.Point(x=xmin, y=ymax),
                ]
            )
        ),
    )

    det2 = schemas.Annotation(
        task_type=enums.TaskType.DETECTION,
        labels=[schemas.Label(key="k", value="v")],
        bounding_box=schemas.BoundingBox.from_extrema(
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
        ),
    )

    crud.create_dataset(db=db, dataset=schemas.Dataset(name=dataset_name))

    crud.create_groundtruth(
        db=db,
        groundtruth=schemas.GroundTruth(
            datum=img1,
            annotations=[det1, det2],
        ),
    )

    dets = db.scalars(select(models.GroundTruth)).all()
    assert len(dets) == 2
    assert set([det.annotation.box is not None for det in dets]) == {
        True,
        False,
    }

    # check we get the same polygon
    assert db.scalar(ST_AsText(dets[0].annotation.polygon)) == db.scalar(
        ST_AsText(dets[1].annotation.box)
    )


def test_create_classification_groundtruth_and_delete_dataset(
    db: Session,
    dataset_name: str,
    gt_clfs_create: list[schemas.GroundTruth],
):
    crud.create_dataset(db=db, dataset=schemas.Dataset(name=dataset_name))

    for gt in gt_clfs_create:
        gt.datum.dataset_name = dataset_name
        crud.create_groundtruth(db=db, groundtruth=gt)

    # should have three GroundTruthClassification rows since one image has two
    # labels and the other has one
    assert db.scalar(func.count(models.GroundTruth.id)) == 3
    assert db.scalar(func.count(models.Datum.id)) == 2
    assert db.scalar(func.count(models.Label.id)) == 3

    # finalize to free dataset
    crud.finalize(db=db, dataset_name=dataset_name)

    # delete dataset and check the cascade worked
    crud.delete(db=db, dataset_name=dataset_name)
    for model_cls in [
        models.Dataset,
        models.Datum,
        models.GroundTruth,
    ]:
        assert db.scalar(func.count(model_cls.id)) == 0

    # make sure labels are still there`
    assert db.scalar(func.count(models.Label.id)) == 3


def test_create_predicted_classifications_and_delete_model(
    db: Session,
    dataset_name: str,
    model_name: str,
    pred_clfs_create: list[schemas.Prediction],
    gt_clfs_create: list[schemas.GroundTruth],
):
    # check this gives an error since the dataset hasn't been added yet
    with pytest.raises(exceptions.DatasetDoesNotExistError) as exc_info:
        crud.create_prediction(db=db, prediction=pred_clfs_create[0])
    assert "does not exist" in str(exc_info)

    # create dataset, add images, and add predictions
    crud.create_dataset(db=db, dataset=schemas.Dataset(name=dataset_name))

    for gt in gt_clfs_create:
        gt.datum.dataset_name = dataset_name
        crud.create_groundtruth(db=db, groundtruth=gt)

    # check this gives an error since the model does not exist
    with pytest.raises(exceptions.ModelDoesNotExistError) as exc_info:
        crud.create_prediction(db=db, prediction=pred_clfs_create[0])
    assert "does not exist" in str(exc_info)

    # finalize dataset
    crud.finalize(db=db, dataset_name=dataset_name)

    # check this gives an error since the images haven't been added yet
    with pytest.raises(exceptions.ModelDoesNotExistError) as exc_info:
        crud.create_prediction(db=db, prediction=pred_clfs_create[0])
    assert "does not exist" in str(exc_info)

    # create model
    crud.create_model(db=db, model=schemas.Model(name=model_name))
    for pd in pred_clfs_create:
        pd.model_name = model_name
        crud.create_prediction(db=db, prediction=pd)

    # check db has the added predictions
    assert db.scalar(func.count(models.Prediction.id)) == 6

    # finalize to free model
    crud.finalize(db=db, dataset_name=dataset_name, model_name=model_name)

    # delete model and check all detections from it are gone
    crud.delete(db=db, model_name=model_name)
    assert db.scalar(func.count(models.Model.id)) == 0
    assert db.scalar(func.count(models.Annotation.id)) == 2
    assert db.scalar(func.count(models.Prediction.id)) == 0

    # delete dataset and check
    crud.delete(db=db, dataset_name=dataset_name)
    assert db.scalar(func.count(models.Model.id)) == 0
    assert db.scalar(func.count(models.Annotation.id)) == 0
    assert db.scalar(func.count(models.GroundTruth.id)) == 0


def _test_create_groundtruth_segmentations_and_delete_dataset(
    db: Session,
    dataset_name: str,
    gts: list[schemas.GroundTruth],
    task: enums.TaskType,
    expected_anns: int,
    expected_gts: int,
    expected_datums: int,
    expected_labels: int,
):
    # sanity check nothing in db
    _check_db_empty(db=db)

    crud.create_dataset(db=db, dataset=schemas.Dataset(name=dataset_name))

    for gt in gts:
        gt.datum.dataset_name = dataset_name
        crud.create_groundtruth(db=db, groundtruth=gt)

    assert db.scalar(func.count(models.Annotation.id)) == expected_anns
    assert db.scalar(func.count(models.Datum.id)) == expected_datums
    assert db.scalar(func.count(models.GroundTruth.id)) == expected_gts
    assert db.scalar(func.count(models.Label.id)) == expected_labels

    for a in db.scalars(select(models.Annotation)):
        assert a.task_type == task

    # finalize to free dataset
    crud.finalize(db=db, dataset_name=dataset_name)

    # delete dataset and check the cascade worked
    crud.delete(db=db, dataset_name=dataset_name)
    for model_cls in [
        models.Dataset,
        models.Datum,
        models.Annotation,
        models.GroundTruth,
    ]:
        assert db.scalar(func.count(model_cls.id)) == 0

    # make sure labels are still there`
    assert db.scalar(func.count(models.Label.id)) == expected_labels


def test_create_groundtruth_instance_segmentations_and_delete_dataset(
    db: Session,
    dataset_name: str,
    groundtruth_instance_segmentations: list[schemas.GroundTruth],
):
    _test_create_groundtruth_segmentations_and_delete_dataset(
        db,
        dataset_name=dataset_name,
        gts=groundtruth_instance_segmentations,
        task=enums.TaskType.DETECTION,
        expected_labels=2,
        expected_anns=4,
        expected_gts=4,
        expected_datums=2,
    )


def test_create_groundtruth_semantic_segmentations_and_delete_dataset(
    db: Session,
    dataset_name: str,
    gt_semantic_segs_create: list[schemas.GroundTruth],
):
    _test_create_groundtruth_segmentations_and_delete_dataset(
        db,
        dataset_name=dataset_name,
        gts=gt_semantic_segs_create,
        task=enums.TaskType.SEGMENTATION,
        expected_labels=4,
        expected_anns=4,
        expected_gts=5,
        expected_datums=2,
    )


def test_create_predicted_segmentations_check_area_and_delete_model(
    db: Session,
    dataset_name: str,
    model_name: str,
    prediction_instance_segmentations: list[schemas.Prediction],
    groundtruth_instance_segmentations: list[schemas.GroundTruth],
):
    # create dataset, add images, and add predictions
    crud.create_dataset(db=db, dataset=schemas.Dataset(name=dataset_name))

    # check this gives an error since the images haven't been added yet
    with pytest.raises(exceptions.ModelDoesNotExistError):
        for pd in prediction_instance_segmentations:
            pd.model_name = model_name
            crud.create_prediction(db=db, prediction=pd)

    # create groundtruths
    for gt in groundtruth_instance_segmentations:
        gt.datum.dataset_name = dataset_name
        crud.create_groundtruth(db=db, groundtruth=gt)

    # check this gives an error since the model has not been crated yet
    with pytest.raises(exceptions.ModelDoesNotExistError) as exc_info:
        for pd in prediction_instance_segmentations:
            crud.create_prediction(db=db, prediction=pd)
    assert "does not exist" in str(exc_info)

    # finalize dataset
    crud.finalize(db=db, dataset_name=dataset_name)

    # check this gives an error since the model hasn't been added yet
    with pytest.raises(exceptions.ModelDoesNotExistError) as exc_info:
        for pd in prediction_instance_segmentations:
            crud.create_prediction(db=db, prediction=pd)
    assert "does not exist" in str(exc_info)

    # create model
    crud.create_model(db=db, model=schemas.Model(name=model_name))

    # check this gives an error since the images haven't been added yet
    with pytest.raises(exceptions.DatumDoesNotExistError) as exc_info:
        for i, pd in enumerate(prediction_instance_segmentations):
            temp_pd = pd.__deepcopy__()
            temp_pd.model_name = model_name
            temp_pd.datum.uid = f"random{i}"
            crud.create_prediction(db=db, prediction=temp_pd)
    assert "does not exist" in str(exc_info)

    # create predictions
    for pd in prediction_instance_segmentations:
        pd.model_name = model_name
        crud.create_prediction(db=db, prediction=pd)

    # check db has the added predictions
    assert db.scalar(func.count(models.Annotation.id)) == 8
    assert db.scalar(func.count(models.Prediction.id)) == 8

    # grab the first one and check that the area of the raster
    # matches the area of the image
    img = crud.get_prediction(
        db=db,
        model_name=model_name,
        dataset_name=dataset_name,
        datum_uid="uid1",
    )

    raster_counts = set(
        db.scalars(
            select(ST_Count(models.Annotation.raster)).where(
                models.Annotation.model_id.isnot(None)
            )
        )
    )

    for i in range(len(img.annotations)):
        mask = _bytes_to_pil(
            b64decode(
                prediction_instance_segmentations[0].annotations[i].raster.mask
            )
        )
        assert np.array(mask).sum() in raster_counts

    # finalize to free model
    crud.finalize(db=db, dataset_name=dataset_name, model_name=model_name)

    # delete model and check all detections from it are gone
    crud.delete(db=db, model_name=model_name)
    assert db.scalar(func.count(models.Model.id)) == 0
    assert db.scalar(func.count(models.Annotation.id)) == 4
    assert db.scalar(func.count(models.Prediction.id)) == 0


def test_segmentation_area_no_hole(
    db: Session,
    dataset_name: str,
    poly_without_hole: schemas.Polygon,
    img1: schemas.Datum,
):
    # sanity check nothing in db
    _check_db_empty(db=db)

    crud.create_dataset(db=db, dataset=schemas.Dataset(name=dataset_name))

    crud.create_groundtruth(
        db=db,
        groundtruth=schemas.GroundTruth(
            datum=img1,
            annotations=[
                schemas.Annotation(
                    task_type=enums.TaskType.DETECTION,
                    labels=[schemas.Label(key="k1", value="v1")],
                    multipolygon=schemas.MultiPolygon(
                        polygons=[poly_without_hole],
                    ),
                )
            ],
        ),
    )

    segmentation_count = db.scalar(select(ST_Count(models.Annotation.raster)))

    assert segmentation_count == math.ceil(45.5)  # area of mask will be an int


def test_segmentation_area_with_hole(
    db: Session,
    dataset_name: str,
    poly_with_hole: schemas.Polygon,
    img1: schemas.Datum,
):
    # sanity check nothing in db
    _check_db_empty(db=db)

    crud.create_dataset(db=db, dataset=schemas.Dataset(name=dataset_name))

    crud.create_groundtruth(
        db=db,
        groundtruth=schemas.GroundTruth(
            datum=img1,
            annotations=[
                schemas.Annotation(
                    task_type=enums.TaskType.SEGMENTATION,
                    labels=[schemas.Label(key="k1", value="v1")],
                    multipolygon=schemas.MultiPolygon(
                        polygons=[poly_with_hole],
                    ),
                )
            ],
        ),
    )

    segmentation = db.scalar(select(models.Annotation))

    # give tolerance of 2 pixels because of poly -> mask conversion
    assert (db.scalar(ST_Count(segmentation.raster)) - 92) <= 2


def test_segmentation_area_multi_polygon(
    db: Session,
    dataset_name: str,
    poly_with_hole: schemas.Polygon,
    poly_without_hole: schemas.Polygon,
    img1: schemas.Datum,
):
    # sanity check nothing in db
    _check_db_empty(db=db)

    crud.create_dataset(db=db, dataset=schemas.Dataset(name=dataset_name))

    crud.create_groundtruth(
        db=db,
        groundtruth=schemas.GroundTruth(
            datum=img1,
            annotations=[
                schemas.Annotation(
                    task_type=enums.TaskType.DETECTION,
                    labels=[schemas.Label(key="k1", value="v1")],
                    multipolygon=schemas.MultiPolygon(
                        polygons=[poly_with_hole, poly_without_hole],
                    ),
                )
            ],
        ),
    )

    segmentation = db.scalar(select(models.Annotation))

    # the two shapes don't intersect so area should be sum of the areas
    # give tolerance of 2 pixels because of poly -> mask conversion
    assert (
        abs(db.scalar(ST_Count(segmentation.raster)) - (math.ceil(45.5) + 92))
        <= 2
    )


# @NOTE This should be handle by `velour.schemas.Raster`
# def test__select_statement_from_poly(


def test_gt_seg_as_mask_or_polys(
    db: Session,
    dataset_name: str,
):
    """Check that a groundtruth segmentation can be created as a polygon or mask"""
    xmin, xmax, ymin, ymax = 11, 45, 37, 102
    h, w = 150, 200
    mask = np.zeros((h, w), dtype=bool)
    mask[ymin:ymax, xmin:xmax] = True
    mask_b64 = b64encode(_np_to_bytes(mask)).decode()

    img = schemas.Datum(
        dataset_name=dataset_name,
        uid="uid",
        metadata={
            "height": h,
            "width": w,
        },
    )

    poly = schemas.BasicPolygon(
        points=[
            schemas.Point(x=xmin, y=ymin),
            schemas.Point(x=xmin, y=ymax),
            schemas.Point(x=xmax, y=ymax),
            schemas.Point(x=xmax, y=ymin),
        ]
    )

    gt1 = schemas.Annotation(
        task_type=enums.TaskType.SEGMENTATION,
        labels=[schemas.Label(key="k1", value="v1")],
        raster=schemas.Raster(
            mask=mask_b64,
            height=h,
            width=w,
        ),
    )
    gt2 = schemas.Annotation(
        task_type=enums.TaskType.DETECTION,
        labels=[schemas.Label(key="k1", value="v1")],
        multipolygon=schemas.MultiPolygon(
            polygons=[
                schemas.Polygon(boundary=poly),
            ]
        ),
    )
    gt = schemas.GroundTruth(
        datum=img,
        annotations=[gt1, gt2],
    )

    _check_db_empty(db=db)

    crud.create_dataset(db=db, dataset=schemas.Dataset(name=dataset_name))

    crud.create_groundtruth(db=db, groundtruth=gt)

    shapes = db.scalars(
        select(ST_AsText(ST_Polygon(models.Annotation.raster)))
    ).all()

    assert len(shapes) == 2
    # check that the mask and polygon define the same polygons
    assert shapes[0] == shapes[1]

    # verify we get the same segmentations back
    segs = crud.get_groundtruth(
        db=db,
        dataset_name=dataset_name,
        datum_uid=img.uid,
    )
    assert (
        len(segs.annotations) == 2
    )  # should just be one instance segmentation
    decoded_mask = _bytes_to_pil(b64decode(segs.annotations[0].raster.mask))
    decoded_mask_arr = np.array(decoded_mask)

    np.testing.assert_equal(decoded_mask_arr, mask)
    assert segs.datum.uid == gt.datum.uid
    assert segs.datum.dataset_name == gt.datum.dataset_name
    for metadatum in segs.datum.metadata:
        assert metadatum in gt.datum.metadata
    assert segs.annotations[0].labels == gt.annotations[0].labels


def test_create_detection_metrics(
    db: Session,
    dataset_name: str,
    model_name: str,
    groundtruths,
    predictions,
):
    # the groundtruths and predictions arguments are not used but
    # those fixtures create the necessary dataset, model, groundtruths, and predictions

    def method_to_test(
        label_key: str, min_area: float = None, max_area: float = None
    ):
        geometric_filters = []
        if min_area:
            geometric_filters.append(
                schemas.NumericFilter(
                    value=min_area,
                    operator=">=",
                )
            )
        if max_area:
            geometric_filters.append(
                schemas.NumericFilter(
                    value=max_area,
                    operator="<=",
                )
            )

        job_request = schemas.EvaluationRequest(
            model_filter=schemas.Filter(
                model_names=["test_model"],
                annotation_geometric_area=geometric_filters
                if geometric_filters
                else None,
                label_keys=[label_key],
            ),
            evaluation_filter=schemas.Filter(
                dataset_names=["test_dataset"],
                task_types=[enums.TaskType.DETECTION],
                annotation_types=[enums.AnnotationType.BOX],
                annotation_geometric_area=geometric_filters
                if geometric_filters
                else None,
                label_keys=[label_key],
            ),
            parameters=schemas.EvaluationParameters(
                iou_thresholds_to_compute=[0.2, 0.6],
                iou_thresholds_to_return=[0.2],
            ),
        )

        # create evaluation (return AP Response)
        evaluations = crud.create_or_get_evaluations(
            db=db,
            job_request=job_request,
        )
        assert len(evaluations) == 1
        resp = evaluations[0]
        return (
            resp.id,
            resp.missing_pred_labels,
            resp.ignored_pred_labels,
        )

    # verify we have no evaluations yet
    assert db.scalar(select(func.count()).select_from(models.Evaluation)) == 0

    # run evaluation
    (
        evaluation_id,
        missing_pred_labels,
        ignored_pred_labels,
    ) = method_to_test(label_key="class")

    # check we have one evaluation
    assert len(crud.get_evaluations(db=db, model_names=[model_name])) == 1

    assert missing_pred_labels == []
    assert ignored_pred_labels == [schemas.Label(key="class", value="3")]

    metrics = db.scalar(
        select(models.Evaluation).where(models.Evaluation.id == evaluation_id)
    ).metrics

    metric_ids = [m.id for m in metrics]

    assert set([m.type for m in metrics]) == {
        "AP",
        "APAveragedOverIOUs",
        "mAP",
        "mAPAveragedOverIOUs",
    }

    assert set(
        [m.parameters["iou"] for m in metrics if m.type in {"AP", "mAP"}]
    ) == {0.2}

    # should be five labels (since thats how many are in groundtruth set)
    assert len(set(m.label_id for m in metrics if m.label_id is not None)) == 5

    # test getting metrics from evaluation settings id
    pydantic_metrics = crud.get_evaluations(
        db=db, evaluation_ids=[evaluation_id]
    )
    for m in pydantic_metrics[0].metrics:
        assert isinstance(m, schemas.Metric)
    assert len(pydantic_metrics[0].metrics) == len(metric_ids)

    # run again and make sure no new ids were created
    evaluation_id_again, _, _ = method_to_test(label_key="class")
    assert evaluation_id == evaluation_id_again
    metric_ids_again = [
        m.id
        for m in db.scalar(
            select(models.Evaluation).where(
                models.Evaluation.id == evaluation_id_again
            )
        ).metrics
    ]
    assert sorted(metric_ids) == sorted(metric_ids_again)

    # test crud.get_model_metrics
    metrics_pydantic = crud.get_evaluations(
        db=db,
        model_names=["test_model"],
        evaluation_ids=[evaluation_id],
    )[0].metrics

    assert len(metrics_pydantic) == len(metrics)

    for m in metrics_pydantic:
        assert m.type in {
            "AP",
            "APAveragedOverIOUs",
            "mAP",
            "mAPAveragedOverIOUs",
        }

    # test when min area and max area are specified
    min_area, max_area = 10, 3000
    (
        evaluation_id,
        missing_pred_labels,
        ignored_pred_labels,
    ) = method_to_test(label_key="class", min_area=min_area, max_area=max_area)

    metrics_pydantic = crud.get_evaluations(
        db=db,
        model_names=["test_model"],
        evaluation_ids=[evaluation_id],
    )[0].metrics
    for m in metrics_pydantic:
        assert m.type in {
            "AP",
            "APAveragedOverIOUs",
            "mAP",
            "mAPAveragedOverIOUs",
        }

    # check we have the right evaluations
    model_evals = crud.get_evaluations(db=db, model_names=[model_name])
    assert len(model_evals) == 2
    # Don't examine metrics
    model_evals[0].metrics = []
    model_evals[1].metrics = []
    assert model_evals[0] == schemas.EvaluationResponse(
        model_filter=schemas.Filter(
            dataset_names=[dataset_name],
            model_names=[model_name],
            label_keys=["class"],
        ),
        evaluation_filter=schemas.Filter(
            dataset_names=[dataset_name],
            model_names=[model_name],
            task_types=[enums.TaskType.DETECTION],
            annotation_types=[enums.AnnotationType.BOX],
            label_keys=["class"],
        ),
        parameters=schemas.EvaluationParameters(
            iou_thresholds_to_compute=[0.2, 0.6],
            iou_thresholds_to_return=[0.2],
        ),
        id=1,
        status=enums.EvaluationStatus.DONE,
        metrics=[],
        confusion_matrices=[],
        missing_pred_labels=[],
        ignored_pred_labels=[
            schemas.Label(key="class", value="3", score=None)
        ],
    )
    assert model_evals[1] == schemas.EvaluationResponse(
        model_filter=schemas.Filter(
            dataset_names=[dataset_name],
            model_names=[model_name],
            annotation_geometric_area=[
                schemas.NumericFilter(
                    value=min_area,
                    operator=">=",
                ),
                schemas.NumericFilter(
                    value=max_area,
                    operator="<=",
                ),
            ],
            label_keys=["class"],
        ),
        evaluation_filter=schemas.Filter(
            dataset_names=[dataset_name],
            model_names=[model_name],
            task_types=[enums.TaskType.DETECTION],
            annotation_types=[enums.AnnotationType.BOX],
            annotation_geometric_area=[
                schemas.NumericFilter(
                    value=min_area,
                    operator=">=",
                ),
                schemas.NumericFilter(
                    value=max_area,
                    operator="<=",
                ),
            ],
            label_keys=["class"],
        ),
        parameters=schemas.EvaluationParameters(
            iou_thresholds_to_compute=[0.2, 0.6],
            iou_thresholds_to_return=[0.2],
        ),
        id=2,
        status=enums.EvaluationStatus.DONE,
        metrics=[],
        confusion_matrices=[],
        missing_pred_labels=[],
        ignored_pred_labels=[],
    )


def test_create_clf_metrics(
    db: Session,
    dataset_name: str,
    model_name: str,
    gt_clfs_create: list[schemas.GroundTruth],
    pred_clfs_create: list[schemas.Prediction],
):
    crud.create_dataset(
        db=db,
        dataset=schemas.Dataset(name=dataset_name),
    )
    for gt in gt_clfs_create:
        gt.datum.dataset_name = dataset_name
        crud.create_groundtruth(db=db, groundtruth=gt)
    crud.finalize(db=db, dataset_name=dataset_name)

    crud.create_model(db=db, model=schemas.Model(name=model_name))
    for pd in pred_clfs_create:
        pd.model_name = model_name
        crud.create_prediction(db=db, prediction=pd)
    crud.finalize(db=db, model_name=model_name, dataset_name=dataset_name)

    job_request = schemas.EvaluationRequest(
        model_filter=schemas.Filter(model_names=[model_name]),
        evaluation_filter=schemas.Filter(
            dataset_names=[dataset_name],
            task_types=[enums.TaskType.CLASSIFICATION],
        ),
    )

    # create clf evaluation (returns Clf Response)
    resp = crud.create_or_get_evaluations(
        db=db,
        job_request=job_request,
    )
    assert len(resp) == 1
    resp = resp[0]
    missing_pred_keys = resp.missing_pred_keys
    ignored_pred_keys = resp.ignored_pred_keys
    evaluation_id = resp.id

    assert missing_pred_keys == []
    assert set(ignored_pred_keys) == {"k3", "k4"}

    # check we have one evaluation
    assert db.scalar(select(func.count()).select_from(models.Evaluation)) == 1

    # get all metrics
    metrics = db.scalar(
        select(models.Evaluation).where(models.Evaluation.id == evaluation_id)
    ).metrics

    assert set([metric.type for metric in metrics]) == {
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROCAUC",
    }
    # should have two accuracy metrics and ROC AUC scores (for label keys "k1" and "k2")
    # and four recall, precision, and f1, for the labels ("k1", "v1"), ("k2", "v2"),
    # ("k2", "v3"), ("k1", "v2")
    for t in ["Accuracy", "ROCAUC"]:
        ms = [m for m in metrics if m.type == t]
        assert len(ms) == 2
        assert set([m.parameters["label_key"] for m in ms]) == {"k1", "k2"}

    for t in ["Precision", "Recall", "F1"]:
        ms = [m for m in metrics if m.type == t]
        assert len(ms) == 4
        assert set([(m.label.key, m.label.value) for m in ms]) == {
            ("k1", "v1"),
            ("k1", "v2"),
            ("k2", "v2"),
            ("k2", "v3"),
        }

    confusion_matrices = db.scalars(
        select(models.ConfusionMatrix).where(
            models.ConfusionMatrix.evaluation_id == evaluation_id
        )
    ).all()

    # should have two confusion matrices, one for each key
    assert len(confusion_matrices) == 2

    # test getting metrics from evaluation settings id
    evaluations = crud.get_evaluations(db=db, evaluation_ids=[evaluation_id])
    assert len(evaluations) == 1

    for m in evaluations[0].metrics:
        assert isinstance(m, schemas.Metric)
    assert len(evaluations[0].metrics) == len(metrics)

    # test getting confusion matrices from evaluation settings id
    cms = evaluations[0].confusion_matrices
    cms = sorted(cms, key=lambda cm: cm.label_key)
    assert len(cms) == 2
    assert cms[0].label_key == "k1"
    assert cms[0].entries == [
        schemas.ConfusionMatrixEntry(
            prediction="v2", groundtruth="v1", count=1
        )
    ]
    assert cms[1].label_key == "k2"
    assert cms[1].entries == [
        schemas.ConfusionMatrixEntry(
            prediction="v2", groundtruth="v3", count=1
        )
    ]

    # attempting to run again should just return the existing job id
    resp = crud.create_or_get_evaluations(
        db=db,
        job_request=job_request,
    )
    assert len(resp) == 1
    assert resp[0].status == enums.EvaluationStatus.DONE

    metrics = db.scalar(
        select(models.Evaluation).where(models.Evaluation.id == evaluation_id)
    ).metrics
    assert len(metrics) == 2 + 2 + 4 + 4 + 4
    confusion_matrices = db.scalars(
        select(models.ConfusionMatrix).where(
            models.ConfusionMatrix.evaluation_id == evaluation_id
        )
    ).all()
    assert len(confusion_matrices) == 2
