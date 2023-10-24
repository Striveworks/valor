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

dset_name = "test_dataset"
model_name = "test_model"


def pil_to_bytes(img: Image.Image) -> bytes:
    f = io.BytesIO()
    img.save(f, format="PNG")
    f.seek(0)
    return f.read()


def bytes_to_pil(b: bytes) -> Image.Image:
    f = io.BytesIO(b)
    img = Image.open(f)
    return img


def np_to_bytes(arr: np.ndarray) -> bytes:
    f = io.BytesIO()
    Image.fromarray(arr).save(f, format="PNG")
    f.seek(0)
    return f.read()


def check_db_empty(db: Session):
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
def gt_dets_create(img1: schemas.Datum) -> list[schemas.GroundTruth]:
    return [
        schemas.GroundTruth(
            dataset=dset_name,
            datum=img1.to_datum(),
            annotations=[
                schemas.Annotation(
                    task_type=enums.TaskType.DETECTION,
                    labels=[
                        schemas.Label(key="k1", value="v1"),
                        schemas.Label(key="k2", value="v2"),
                    ],
                    metadata=[],
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
                    boundary=[],
                ),
                schemas.Annotation(
                    task_type=enums.TaskType.DETECTION,
                    labels=[schemas.Label(key="k2", value="v2")],
                    metadata=[],
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
def pred_dets_create(img1: schemas.Datum) -> list[schemas.Prediction]:
    return [
        schemas.Prediction(
            model=model_name,
            datum=img1.to_datum(),
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
def gt_instance_segs_create(
    poly_with_hole: schemas.BasicPolygon,
    poly_without_hole: schemas.BasicPolygon,
    img1: schemas.ImageMetadata,
    img2: schemas.ImageMetadata,
) -> list[schemas.GroundTruth]:
    return [
        schemas.GroundTruth(
            dataset=dset_name,
            datum=img1.to_datum(),
            annotations=[
                schemas.Annotation(
                    task_type=enums.TaskType.DETECTION,
                    labels=[schemas.Label(key="k1", value="v1")],
                    polygon=poly_with_hole,
                ),
            ],
        ),
        schemas.GroundTruth(
            dataset=dset_name,
            datum=img2.to_datum(),
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
def pred_instance_segs_create(
    img1_pred_mask_bytes1: bytes,
    img1: schemas.ImageMetadata,
) -> list[schemas.Prediction]:
    b64_mask1 = b64encode(img1_pred_mask_bytes1).decode()

    return [
        schemas.Prediction(
            model=model_name,
            datum=img1.to_datum(),
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
    img1: schemas.ImageMetadata,
    img2: schemas.ImageMetadata,
) -> list[schemas.GroundTruth]:
    return [
        schemas.GroundTruth(
            dataset=dset_name,
            datum=img1.to_datum(),
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
            dataset=dset_name,
            datum=img2.to_datum(),
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
    img1: schemas.ImageMetadata, img2: schemas.ImageMetadata
) -> list[schemas.Prediction]:
    return [
        schemas.Prediction(
            model=model_name,
            datum=img1.to_datum(),
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
            model=model_name,
            datum=img2.to_datum(),
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
def dataset_names():
    return ["dataset1", "dataset2"]


@pytest.fixture
def model_names():
    return ["model1", "model2"]


@pytest.fixture
def dataset_model_create(
    db: Session,
    gt_dets_create: list[schemas.GroundTruth],
    pred_dets_create: list[schemas.Prediction],
    dataset_names: list[str],
    model_names: list[str],
):
    # create dataset1
    crud.create_dataset(
        db=db,
        dataset=schemas.Dataset(name=dataset_names[0]),
    )
    for gt in gt_dets_create:
        gt.datum.dataset = dataset_names[0]
        crud.create_groundtruth(db=db, groundtruth=gt)
    crud.finalize(db=db, dataset_name=dataset_names[0])

    # Create model1
    crud.create_model(db=db, model=schemas.Model(name=model_names[0]))

    # Link model1 to dataset1
    for pd in pred_dets_create:
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


""" CREATE """


def test_create_and_get_datasets(db: Session):
    # Create dataset
    crud.create_dataset(db=db, dataset=schemas.Dataset(name=dset_name))

    all_datasets = db.scalars(select(models.Dataset)).all()
    assert len(all_datasets) == 1
    assert all_datasets[0].name == dset_name

    with pytest.raises(exceptions.DatasetAlreadyExistsError) as exc_info:
        crud.create_dataset(
            db=db,
            dataset=schemas.Dataset(name=dset_name),
        )
    assert "already exists" in str(exc_info)

    crud.create_dataset(
        db=db,
        dataset=schemas.Dataset(name="other_dataset"),
    )
    datasets = crud.get_datasets(db=db)
    assert len(datasets) == 2
    assert set([d.name for d in datasets]) == {dset_name, "other_dataset"}


def test_create_and_get_models(db: Session):
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
    gt_dets_create: list[schemas.GroundTruth],
):
    # sanity check nothing in db
    check_db_empty(db=db)

    crud.create_dataset(db=db, dataset=schemas.Dataset(name=dset_name))

    for gt in gt_dets_create:
        crud.create_groundtruth(db=db, groundtruth=gt)

    assert db.scalar(func.count(models.Annotation.id)) == 2
    assert db.scalar(func.count(models.Datum.id)) == 1
    assert db.scalar(func.count(models.GroundTruth.id)) == 3
    assert db.scalar(func.count(models.Label.id)) == 2

    # verify we get the same dets back
    for gt in gt_dets_create:
        new_gt = crud.get_groundtruth(
            db=db, dataset_name=gt.datum.dataset, datum_uid=gt.datum.uid
        )
        assert gt.datum.uid == new_gt.datum.uid
        assert gt.datum.dataset == new_gt.datum.dataset
        for metadatum in gt.datum.metadata:
            assert metadatum in new_gt.datum.metadata
        for gta, new_gta in zip(gt.annotations, new_gt.annotations):
            assert set(gta.labels) == set(new_gta.labels)
            assert gta == new_gta

    # delete dataset and check the cascade worked
    crud.delete(db=db, dataset_name=dset_name)
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
    pred_dets_create: list[schemas.Prediction],
    gt_dets_create: list[schemas.GroundTruth],
):
    # check this gives an error since the model hasn't been added yet
    with pytest.raises(exceptions.DatasetDoesNotExistError) as exc_info:
        for pd in pred_dets_create:
            crud.create_prediction(db=db, prediction=pd)
    assert "does not exist" in str(exc_info)

    # create dataset, add images, and add predictions
    crud.create_dataset(db=db, dataset=schemas.Dataset(name=dset_name))

    for gt in gt_dets_create:
        crud.create_groundtruth(db=db, groundtruth=gt)

    # check this gives an error since the model hasn't been created yet
    with pytest.raises(exceptions.ModelDoesNotExistError) as exc_info:
        for pd in pred_dets_create:
            crud.create_prediction(db=db, prediction=pd)
    assert "does not exist" in str(exc_info)

    # finalize dataset
    crud.finalize(db=db, dataset_name=dset_name)

    # check this gives an error since the model hasn't been added yet
    with pytest.raises(exceptions.ModelDoesNotExistError) as exc_info:
        for pd in pred_dets_create:
            crud.create_prediction(db=db, prediction=pd)
    assert "does not exist" in str(exc_info)

    # create model
    crud.create_model(db=db, model=schemas.Model(name=model_name))
    for pd in pred_dets_create:
        crud.create_prediction(db=db, prediction=pd)

    # check db has the added predictions
    assert db.scalar(func.count(models.Annotation.id)) == 4
    assert db.scalar(func.count(models.Datum.id)) == 1
    assert db.scalar(func.count(models.GroundTruth.id)) == 3
    assert db.scalar(func.count(models.Prediction.id)) == 6
    assert db.scalar(func.count(models.Label.id)) == 4

    # delete model and check all detections from it are gone
    crud.delete(db=db, model_name=model_name)
    assert db.scalar(func.count(models.Annotation.id)) == 2
    assert db.scalar(func.count(models.Datum.id)) == 1
    assert db.scalar(func.count(models.GroundTruth.id)) == 3
    assert db.scalar(func.count(models.Prediction.id)) == 0
    assert db.scalar(func.count(models.Label.id)) == 4


def test_create_detections_as_bbox_or_poly(db: Session, img1: schemas.Datum):
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

    crud.create_dataset(db=db, dataset=schemas.Dataset(name=dset_name))

    crud.create_groundtruth(
        db=db,
        groundtruth=schemas.GroundTruth(
            dataset=dset_name,
            datum=img1.to_datum(),
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
    db: Session, gt_clfs_create: list[schemas.GroundTruth]
):
    crud.create_dataset(db=db, dataset=schemas.Dataset(name=dset_name))

    for gt in gt_clfs_create:
        gt.datum.dataset = dset_name
        crud.create_groundtruth(db=db, groundtruth=gt)

    # should have three GroundTruthClassification rows since one image has two
    # labels and the other has one
    assert db.scalar(func.count(models.GroundTruth.id)) == 3
    assert db.scalar(func.count(models.Datum.id)) == 2
    assert db.scalar(func.count(models.Label.id)) == 3

    # delete dataset and check the cascade worked
    crud.delete(db=db, dataset_name=dset_name)
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
    pred_clfs_create: list[schemas.Prediction],
    gt_clfs_create: list[schemas.GroundTruth],
):
    # check this gives an error since the model hasn't been added yet
    with pytest.raises(exceptions.DatasetDoesNotExistError) as exc_info:
        crud.create_prediction(db=db, prediction=pred_clfs_create[0])
    assert "does not exist" in str(exc_info)

    # create dataset, add images, and add predictions
    crud.create_dataset(db=db, dataset=schemas.Dataset(name=dset_name))

    for gt in gt_clfs_create:
        gt.datum.dataset = dset_name
        crud.create_groundtruth(db=db, groundtruth=gt)

    # check this gives an error since the model does not exist
    with pytest.raises(exceptions.ModelDoesNotExistError) as exc_info:
        crud.create_prediction(db=db, prediction=pred_clfs_create[0])
    assert "does not exist" in str(exc_info)

    # finalize dataset
    crud.finalize(db=db, dataset_name=dset_name)

    # check this gives an error since the images haven't been added yet
    with pytest.raises(exceptions.ModelDoesNotExistError) as exc_info:
        crud.create_prediction(db=db, prediction=pred_clfs_create[0])
    assert "does not exist" in str(exc_info)

    # create model
    crud.create_model(db=db, model=schemas.Model(name=model_name))
    for pd in pred_clfs_create:
        pd.model = model_name
        crud.create_prediction(db=db, prediction=pd)

    # check db has the added predictions
    assert db.scalar(func.count(models.Prediction.id)) == 6

    # delete model and check all detections from it are gone
    crud.delete(db=db, model_name=model_name)
    assert db.scalar(func.count(models.Model.id)) == 0
    assert db.scalar(func.count(models.Annotation.id)) == 2
    assert db.scalar(func.count(models.Prediction.id)) == 0

    # delete dataset and check
    crud.delete(db=db, dataset_name=dset_name)
    assert db.scalar(func.count(models.Model.id)) == 0
    assert db.scalar(func.count(models.Annotation.id)) == 0
    assert db.scalar(func.count(models.GroundTruth.id)) == 0


def _test_create_groundtruth_segmentations_and_delete_dataset(
    db: Session,
    gts: list[schemas.GroundTruth],
    task: enums.TaskType,
    expected_anns: int,
    expected_gts: int,
    expected_datums: int,
    expected_labels: int,
):
    # sanity check nothing in db
    check_db_empty(db=db)

    crud.create_dataset(db=db, dataset=schemas.Dataset(name=dset_name))

    for gt in gts:
        gt.datum.dataset = dset_name
        crud.create_groundtruth(db=db, groundtruth=gt)

    assert db.scalar(func.count(models.Annotation.id)) == expected_anns
    assert db.scalar(func.count(models.Datum.id)) == expected_datums
    assert db.scalar(func.count(models.GroundTruth.id)) == expected_gts
    assert db.scalar(func.count(models.Label.id)) == expected_labels

    for a in db.scalars(select(models.Annotation)):
        assert a.task_type == task

    # delete dataset and check the cascade worked
    crud.delete(db=db, dataset_name=dset_name)
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
    db: Session, gt_instance_segs_create: list[schemas.GroundTruth]
):
    _test_create_groundtruth_segmentations_and_delete_dataset(
        db,
        gt_instance_segs_create,
        enums.TaskType.DETECTION,
        expected_labels=2,
        expected_anns=4,
        expected_gts=4,
        expected_datums=2,
    )


def test_create_groundtruth_semantic_segmentations_and_delete_dataset(
    db: Session, gt_semantic_segs_create: list[schemas.GroundTruth]
):
    _test_create_groundtruth_segmentations_and_delete_dataset(
        db,
        gt_semantic_segs_create,
        enums.TaskType.SEGMENTATION,
        expected_labels=4,
        expected_anns=4,
        expected_gts=5,
        expected_datums=2,
    )


def test_create_predicted_segmentations_check_area_and_delete_model(
    db: Session,
    pred_instance_segs_create: list[schemas.Prediction],
    gt_instance_segs_create: list[schemas.GroundTruth],
):
    # create dataset, add images, and add predictions
    crud.create_dataset(db=db, dataset=schemas.Dataset(name=dset_name))

    # check this gives an error since the images haven't been added yet
    with pytest.raises(exceptions.StateflowError) as exc_info:
        for pd in pred_instance_segs_create:
            pd.model = model_name
            crud.create_prediction(db=db, prediction=pd)
    assert "does not support model operations" in str(exc_info)

    # create groundtruths
    for gt in gt_instance_segs_create:
        gt.datum.dataset = dset_name
        crud.create_groundtruth(db=db, groundtruth=gt)

    # check this gives an error since the model has not been crated yet
    with pytest.raises(exceptions.ModelDoesNotExistError) as exc_info:
        for pd in pred_instance_segs_create:
            crud.create_prediction(db=db, prediction=pd)
    assert "does not exist" in str(exc_info)

    # finalize dataset
    crud.finalize(db=db, dataset_name=dset_name)

    # check this gives an error since the model hasn't been added yet
    with pytest.raises(exceptions.ModelDoesNotExistError) as exc_info:
        for pd in pred_instance_segs_create:
            crud.create_prediction(db=db, prediction=pd)
    assert "does not exist" in str(exc_info)

    crud.create_model(db=db, model=schemas.Model(name=model_name))

    # check this gives an error since the images haven't been added yet
    with pytest.raises(exceptions.DatumDoesNotExistError) as exc_info:
        for i, pd in enumerate(pred_instance_segs_create):
            temp_pd = pd.__deepcopy__()
            temp_pd.model = model_name
            temp_pd.datum.uid = f"random{i}"
            crud.create_prediction(db=db, prediction=temp_pd)
    assert "does not exist" in str(exc_info)

    # create predictions
    for pd in pred_instance_segs_create:
        pd.model = model_name
        crud.create_prediction(db=db, prediction=pd)

    # check db has the added predictions
    assert db.scalar(func.count(models.Annotation.id)) == 8
    assert db.scalar(func.count(models.Prediction.id)) == 8

    # grab the first one and check that the area of the raster
    # matches the area of the image
    img = crud.get_prediction(
        db=db, model_name=model_name, dataset_name=dset_name, datum_uid="uid1"
    )

    raster_counts = set(
        db.scalars(
            select(ST_Count(models.Annotation.raster)).where(
                models.Annotation.model_id.isnot(None)
            )
        )
    )

    for i in range(len(img.annotations)):
        mask = bytes_to_pil(
            b64decode(pred_instance_segs_create[0].annotations[i].raster.mask)
        )
        assert np.array(mask).sum() in raster_counts

    # delete model and check all detections from it are gone
    crud.delete(db=db, model_name=model_name)
    assert db.scalar(func.count(models.Model.id)) == 0
    assert db.scalar(func.count(models.Annotation.id)) == 4
    assert db.scalar(func.count(models.Prediction.id)) == 0


def test_segmentation_area_no_hole(
    db: Session,
    poly_without_hole: schemas.Polygon,
    img1: schemas.ImageMetadata,
):
    # sanity check nothing in db
    check_db_empty(db=db)

    crud.create_dataset(db=db, dataset=schemas.Dataset(name=dset_name))

    crud.create_groundtruth(
        db=db,
        groundtruth=schemas.GroundTruth(
            datum=img1.to_datum(),
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
    db: Session, poly_with_hole: schemas.Polygon, img1: schemas.Datum
):
    # sanity check nothing in db
    check_db_empty(db=db)

    crud.create_dataset(db=db, dataset=schemas.Dataset(name=dset_name))

    crud.create_groundtruth(
        db=db,
        groundtruth=schemas.GroundTruth(
            datum=img1.to_datum(),
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
    poly_with_hole: schemas.Polygon,
    poly_without_hole: schemas.Polygon,
    img1: schemas.ImageMetadata,
):
    # sanity check nothing in db
    check_db_empty(db=db)

    crud.create_dataset(db=db, dataset=schemas.Dataset(name=dset_name))

    crud.create_groundtruth(
        db=db,
        groundtruth=schemas.GroundTruth(
            dataset=dset_name,
            datum=img1.to_datum(),
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


def test_gt_seg_as_mask_or_polys(db: Session):
    """Check that a groundtruth segmentation can be created as a polygon or mask"""
    xmin, xmax, ymin, ymax = 11, 45, 37, 102
    h, w = 150, 200
    mask = np.zeros((h, w), dtype=bool)
    mask[ymin:ymax, xmin:xmax] = True
    mask_b64 = b64encode(np_to_bytes(mask)).decode()

    img = schemas.ImageMetadata(
        dataset=dset_name,
        uid="uid",
        height=h,
        width=w,
    ).to_datum()

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

    check_db_empty(db=db)

    crud.create_dataset(db=db, dataset=schemas.Dataset(name=dset_name))

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
        dataset_name=dset_name,
        datum_uid=img.uid,
    )
    assert (
        len(segs.annotations) == 2
    )  # should just be one instance segmentation
    decoded_mask = bytes_to_pil(b64decode(segs.annotations[0].raster.mask))
    decoded_mask_arr = np.array(decoded_mask)

    np.testing.assert_equal(decoded_mask_arr, mask)
    assert segs.datum.uid == gt.datum.uid
    assert segs.datum.dataset == gt.datum.dataset
    for metadatum in segs.datum.metadata:
        assert metadatum in gt.datum.metadata
    assert segs.annotations[0].labels == gt.annotations[0].labels


def test_create_detection_metrics(db: Session, groundtruths, predictions):
    # the groundtruths and predictions arguments are not used but
    # those fixtures create the necessary dataset, model, groundtruths, and predictions

    def method_to_test(
        label_key: str, min_area: float = None, max_area: float = None
    ):
        settings = schemas.EvaluationSettings(
            model="test_model",
            dataset="test_dataset",
            parameters=schemas.DetectionParameters(
                min_area=min_area,
                max_area=max_area,
                annotation_type=enums.AnnotationType.BOX,
                label_key=label_key,
                iou_thresholds_to_compute=[0.2, 0.6],
                iou_thresholds_to_keep=[0.2],
            ),
        )

        # create evaluation (return AP Response)
        resp = crud.create_detection_evaluation(db=db, settings=settings)

        # run computation (returns nothing on completion)
        crud.compute_detection_metrics(
            db=db,
            settings=settings,
            job_id=resp.job_id,
        )

        return (
            resp.job_id,
            resp.missing_pred_labels,
            resp.ignored_pred_labels,
        )

    # verify we have no evaluations yet
    assert (
        len(crud.get_model_evaluation_settings(db=db, model_name=model_name))
        == 0
    )

    # run evaluation
    (
        evaluation_id,
        missing_pred_labels,
        ignored_pred_labels,
    ) = method_to_test(label_key="class")

    # check we have one evaluation
    assert (
        len(crud.get_model_evaluation_settings(db=db, model_name=model_name))
        == 1
    )

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
    pydantic_metrics = crud.get_metrics_from_evaluation_id(
        db=db, evaluation_id=evaluation_id
    )
    for m in pydantic_metrics:
        assert isinstance(m, schemas.Metric)
    assert len(pydantic_metrics) == len(metric_ids)

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
    metrics_pydantic = crud.get_model_metrics(
        db=db,
        model_name="test_model",
        evaluation_id=evaluation_id,
    )

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

    metrics_pydantic = crud.get_model_metrics(
        db=db,
        model_name="test_model",
        evaluation_id=evaluation_id,
    )
    for m in metrics_pydantic:
        assert m.type in {
            "AP",
            "APAveragedOverIOUs",
            "mAP",
            "mAPAveragedOverIOUs",
        }

    # check we have the right evaluations
    model_evals = crud.get_model_evaluation_settings(
        db=db, model_name=model_name
    )
    assert len(model_evals) == 2
    assert model_evals[0] == schemas.EvaluationSettings(
        model=model_name,
        dataset=dset_name,
        parameters=schemas.DetectionParameters(
            annotation_type=enums.AnnotationType.BOX,
            label_key="class",
            iou_thresholds_to_compute=[0.2, 0.6],
            iou_thresholds_to_keep=[0.2],
        ),
        id=1,
    )
    assert model_evals[1] == schemas.EvaluationSettings(
        model=model_name,
        dataset=dset_name,
        parameters=schemas.DetectionParameters(
            annotation_type=enums.AnnotationType.BOX,
            label_key="class",
            min_area=min_area,
            max_area=max_area,
            iou_thresholds_to_compute=[0.2, 0.6],
            iou_thresholds_to_keep=[0.2],
        ),
        id=2,
    )


def test_create_clf_metrics(
    db: Session,
    gt_clfs_create: list[schemas.GroundTruth],
    pred_clfs_create: list[schemas.Prediction],
):
    crud.create_dataset(
        db=db,
        dataset=schemas.Dataset(name=dset_name),
    )
    for gt in gt_clfs_create:
        gt.datum.dataset = dset_name
        crud.create_groundtruth(db=db, groundtruth=gt)
    crud.finalize(db=db, dataset_name=dset_name)

    crud.create_model(db=db, model=schemas.Model(name=model_name))
    for pd in pred_clfs_create:
        pd.model = model_name
        crud.create_prediction(db=db, prediction=pd)
    crud.finalize(db=db, model_name=model_name, dataset_name=dset_name)

    settings = schemas.EvaluationSettings(
        model=model_name,
        dataset=dset_name,
    )

    # create clf evaluation (returns Clf Response)
    resp = crud.create_clf_evaluation(
        db=db,
        settings=settings,
    )
    missing_pred_keys = resp.missing_pred_keys
    ignored_pred_keys = resp.ignored_pred_keys
    evaluation_id = resp.job_id

    assert missing_pred_keys == []
    assert set(ignored_pred_keys) == {"k3", "k4"}

    # compute clf metrics
    crud.compute_clf_metrics(
        db=db,
        settings=settings,
        job_id=evaluation_id,
    )

    # check we have one evaluation
    assert (
        len(crud.get_model_evaluation_settings(db=db, model_name=model_name))
        == 1
    )

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
    pydantic_metrics = crud.get_metrics_from_evaluation_id(
        db=db, evaluation_id=evaluation_id
    )
    for m in pydantic_metrics:
        assert isinstance(m, schemas.Metric)
    assert len(pydantic_metrics) == len(metrics)

    # test getting confusion matrices from evaluation settings id
    cms = crud.get_confusion_matrices_from_evaluation_id(
        db=db, evaluation_id=evaluation_id
    )
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
    crud.compute_clf_metrics(
        db=db,
        settings=settings,
        job_id=evaluation_id,
    )
    assert (
        len(crud.get_model_evaluation_settings(db=db, model_name=model_name))
        == 1
    )

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


# @NOTE: This should now be handled by `velour_api.schemas.Raster`
# def test__raster_to_png_b64(db: Session):


# @NOTE: This is now handled by `velour_api.backend.metrics.detections`
# def test__instance_segmentations_in_dataset_statement(


# @NOTE: Moved to `velour_api.backend.metrics.detections`
# def test___model_instance_segmentation_preds_statement(


# @NOTE: Moved to `velour_api.backend.metrics.detections`
# def test___object_detections_in_dataset_statement(db: Session, groundtruths):


# @NOTE: Moved to `velour_api.backend.metrics.detections`
# def test__model_object_detection_preds_statement(


# @NOTE: Moved to `velour_api.backend.metrics.detections`
# def test__filter_instance_segmentations_by_area(db: Session):


# @NOTE: Moved to `velour_api.backend.metrics.detections`
# def test__filter_object_detections_by_area(db: Session):


# @NOTE: Moved to `velour_api.backend.metrics.detections`
# def test__filter_instance_segmentations_by_area_using_mask(db: Session):


def test_finalize_empty_dataset(db: Session):
    # create dataset
    crud.create_dataset(db=db, dataset=schemas.Dataset(name=dset_name))
    # finalize
    with pytest.raises(exceptions.DatasetIsEmptyError) as e:
        crud.finalize(db=db, dataset_name=dset_name)
    assert "contains no groundtruths" in str(e)


def test_finalize_empty_model(db: Session, groundtruths):
    # create model
    crud.create_model(db=db, model=schemas.Model(name=model_name))
    # finalize
    with pytest.raises(exceptions.ModelInferencesDoNotExist) as e:
        crud.finalize(db=db, dataset_name=dset_name, model_name=model_name)
    assert "do not exist" in str(e)


# @NOTE: `velour_api.backend.io`
# @TODO: Implement a test that checks the existince and linking of metatata


""" READ """


def test_get_dataset(db: Session):
    with pytest.raises(exceptions.DatasetDoesNotExistError) as exc_info:
        crud.get_dataset(db=db, dataset_name=dset_name)
    assert "does not exist" in str(exc_info)

    crud.create_dataset(db=db, dataset=schemas.Dataset(name=dset_name))

    dset = crud.get_dataset(db=db, dataset_name=dset_name)
    assert dset.name == dset_name


def test_get_model(db: Session):
    with pytest.raises(exceptions.ModelDoesNotExistError) as exc_info:
        crud.get_model(db=db, model_name=model_name)
    assert "does not exist" in str(exc_info)

    crud.create_model(db=db, model=schemas.Model(name=model_name))
    model = crud.get_model(db=db, model_name=model_name)
    assert model.name == model_name


# @NOTE: Sub-functionality of `crud.get_info`
# @TODO: Implement `crud.get_info`
# def test_get_dataset_info(

# @NOTE: Sub-functionality of `crud.get_info`
# @TODO: Implement `crud.get_info`
# def test_get_model_info(

# @NOTE: Sub-functionality of `crud.get_info`
# @TODO: Implement `crud.get_info`
# def test__get_associated_models(

# @NOTE: Sub-functionality of `crud.get_info`
# @TODO: Implement `crud.get_info`
# def test__get_associated_datasets(

# @NOTE: Sub-functionality of `crud.get_info`
# @TODO: Implement `crud.get_info`
# def test_get_label_distribution_from_dataset(

# @NOTE: Sub-functionality of `crud.get_info`
# @TODO: Implement `crud.get_info`
# def test_get_label_distribution_from_model(


def test_get_all_labels(db: Session, gt_dets_create: schemas.GroundTruth):
    crud.create_dataset(db=db, dataset=schemas.Dataset(name=dset_name))

    for gt in gt_dets_create:
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
            datasets=schemas.DatasetFilter(names=[dataset_names[0]]),
        ),
    )
    assert len(ds1) == 2
    assert schemas.Label(key="k1", value="v1") in ds1
    assert schemas.Label(key="k2", value="v2") in ds1

    # NEGATIVE - Test filter by task type
    ds1 = crud.get_dataset_labels(
        db=db,
        filters=schemas.Filter(
            datasets=schemas.DatasetFilter(names=[dataset_names[0]]),
            annotations=schemas.AnnotationFilter(
                task_types=[
                    enums.TaskType.CLASSIFICATION,
                    enums.TaskType.SEGMENTATION,
                ]
            ),
        ),
    )
    assert ds1 == []

    # POSITIVE - Test filter by task type
    ds1 = crud.get_dataset_labels(
        db=db,
        filters=schemas.Filter(
            datasets=schemas.DatasetFilter(names=[dataset_names[0]]),
            annotations=schemas.AnnotationFilter(
                task_types=[enums.TaskType.DETECTION]
            ),
        ),
    )
    assert len(ds1) == 2
    assert schemas.Label(key="k1", value="v1") in ds1
    assert schemas.Label(key="k2", value="v2") in ds1

    # NEGATIVE - Test filter by annotation type
    ds1 = crud.get_dataset_labels(
        db=db,
        filters=schemas.Filter(
            datasets=schemas.DatasetFilter(names=[dataset_names[0]]),
            annotations=schemas.AnnotationFilter(
                annotation_types=[
                    enums.AnnotationType.POLYGON,
                    enums.AnnotationType.MULTIPOLYGON,
                    enums.AnnotationType.RASTER,
                ]
            ),
        ),
    )
    assert ds1 == []

    # POSITIVE - Test filter by annotation type
    ds1 = crud.get_dataset_labels(
        db=db,
        filters=schemas.Filter(
            datasets=schemas.DatasetFilter(names=[dataset_names[0]]),
            annotations=schemas.AnnotationFilter(
                annotation_types=[
                    enums.AnnotationType.BOX,
                ]
            ),
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
            models=schemas.ModelFilter(names=[model_names[0]]),
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
            models=schemas.ModelFilter(names=[model_names[0]]),
            annotations=schemas.AnnotationFilter(
                task_types=[enums.TaskType.CLASSIFICATION],
            ),
        ),
    )
    assert md1 == []

    # Test get only polygon labels from model 1
    md1 = crud.get_model_labels(
        db=db,
        filters=schemas.Filter(
            models=schemas.ModelFilter(names=[model_names[0]]),
            annotations=schemas.AnnotationFilter(
                annotation_types=[enums.AnnotationType.BOX],
            ),
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


# @NOTE: `velour_api.backend.io`
# @TODO: Need to implement metadata querys
# def test_get_string_metadata_ids(db: Session):
