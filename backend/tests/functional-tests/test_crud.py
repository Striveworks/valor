import io
import json
import math
from base64 import b64decode, b64encode

import numpy as np
import pytest
from geoalchemy2.functions import (
    ST_Area,
    ST_AsGeoJSON,
    ST_AsText,
    ST_Count,
    ST_Polygon,
)
from PIL import Image, ImageDraw
from sqlalchemy import func, insert, select
from sqlalchemy.orm import Session

from velour_api import crud, enums, exceptions, schemas
from velour_api.crud import _create, _read, _update, _delete
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
        models.MetaDatum,
        models.Annotation,
        models.Datum,
        models.Model,
        models.Dataset,
        models.EvaluationSettings,
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
                schemas.Point(x=0,  y=10),
                schemas.Point(x=10, y=10),
                schemas.Point(x=10, y=0),
                schemas.Point(x=0,  y=0),
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
        ]
    )


@pytest.fixture
def gt_dets_create(img1: schemas.Datum) -> list[schemas.GroundTruth]:
    return [
        schemas.GroundTruth(
            dataset_name=dset_name,
            datum=img1,
            annotations=[
                schemas.GroundTruthAnnotation(
                    labels=[
                        schemas.Label(key="k1", value="v1"),
                        schemas.Label(key="k2", value="v2"),
                    ],
                    annotation=schemas.Annotation(
                        task_type=enums.TaskType.DETECTION,
                        metadata=[],
                        bounding_box=schemas.BoundingBox(
                            polygon=schemas.BasicPolygon(
                                points=[
                                    schemas.Point(x=10, y=20),
                                    schemas.Point(x=10, y=30),
                                    schemas.Point(x=20, y=30),
                                    schemas.Point(x=20, y=20), # removed repeated first point
                                ]
                            )
                        )
                    ),
                    boundary=[],
                ),
                schemas.GroundTruthAnnotation(
                    labels=[schemas.Label(key="k2", value="v2")],
                    annotation=schemas.Annotation(
                        task_type=enums.TaskType.DETECTION,
                        metadata=[],
                        bounding_box=schemas.BoundingBox(
                            polygon=schemas.BasicPolygon(
                                points=[
                                    schemas.Point(x=10, y=20),
                                    schemas.Point(x=10, y=30),
                                    schemas.Point(x=20, y=30),
                                    schemas.Point(x=20, y=20), # removed repeated first point
                                ]
                            )
                        )
                    )
                )
            ]
        )
    ]


@pytest.fixture
def pred_dets_create(img1: schemas.Datum) -> list[schemas.Prediction]:
    return [
        schemas.Prediction(
            model_name=model_name,
            datum=img1,
            annotations=[
                schemas.PredictedAnnotation(
                    scored_labels=[
                        schemas.ScoredLabel(
                            label=schemas.Label(key="k1", value="v1"), score=0.6
                        ),
                        schemas.ScoredLabel(
                            label=schemas.Label(key="k1", value="v2"), score=0.4
                        ),
                        schemas.ScoredLabel(
                            label=schemas.Label(key="k2", value="v1"), score=0.8
                        ),
                        schemas.ScoredLabel(
                            label=schemas.Label(key="k2", value="v2"), score=0.2
                        ),
                    ],
                    annotation=schemas.Annotation(
                        task_type=enums.TaskType.DETECTION,
                        bounding_box=schemas.BoundingBox(
                            polygon=schemas.BasicPolygon(
                                points=[
                                    schemas.Point(x=107, y=207),
                                    schemas.Point(x=107, y=307),
                                    schemas.Point(x=207, y=307),
                                    schemas.Point(x=207, y=207)
                                ]
                            )
                        )
                    )
                ),
                schemas.PredictedAnnotation(
                    scored_labels=[
                        schemas.ScoredLabel(
                            label=schemas.Label(key="k2", value="v1"), score=0.1
                        ),
                        schemas.ScoredLabel(
                            label=schemas.Label(key="k2", value="v2"), score=0.9
                        ),
                    ],
                    annotation=schemas.Annotation(
                        task_type=enums.TaskType.DETECTION,
                        bounding_box=schemas.BoundingBox(
                            polygon=schemas.BasicPolygon(
                                points=[
                                    schemas.Point(x=107, y=207),
                                    schemas.Point(x=107, y=307),
                                    schemas.Point(x=207, y=307),
                                    schemas.Point(x=207, y=207)
                                ]
                            )
                        )
                    )
                ),
            ],
        )
    ]

@pytest.fixture
def gt_segs_create(
    poly_with_hole: schemas.BasicPolygon, 
    poly_without_hole: schemas.BasicPolygon, 
    img1: schemas.Datum, 
    img2: schemas.Datum,
) -> list[schemas.GroundTruth:]:
    return [
        schemas.GroundTruth(
            dataset_name=dset_name,
            datum=img1,
            annotations=[
                schemas.GroundTruthAnnotation(
                    labels=[schemas.Label(key="k1", value="v1")],
                    annotation=schemas.Annotation(
                        task_type=enums.TaskType.INSTANCE_SEGMENTATION,
                        polygon=poly_with_hole,
                    )
                ),
            ]
        ),
        schemas.GroundTruth(
            dataset_name=dset_name,
            datum=img2,
            annotations=[
                schemas.GroundTruthAnnotation(
                    labels=[schemas.Label(key="k1", value="v1")],
                    annotation=schemas.Annotation(
                        task_type=enums.TaskType.SEMANTIC_SEGMENTATION,
                        polygon=poly_without_hole,
                    )
                ),
                schemas.GroundTruthAnnotation(
                    labels=[schemas.Label(key="k3", value="v3")],
                    annotation=schemas.Annotation(
                        task_type=enums.TaskType.INSTANCE_SEGMENTATION,
                        polygon=poly_without_hole,
                    )
                ),
                schemas.GroundTruthAnnotation(
                    labels=[schemas.Label(key="k1", value="v1")],
                    annotation=schemas.Annotation(
                        task_type=enums.TaskType.INSTANCE_SEGMENTATION,
                        multipolygon=schemas.MultiPolygon(
                            polygons=[
                                poly_with_hole, 
                                poly_without_hole
                            ],
                        )
                    )
                ),
            ],
        )
    ]

@pytest.fixture
def pred_segs_create(
    mask_bytes1: tuple[bytes, tuple[float,float]],
    mask_bytes2: tuple[bytes, tuple[float,float]],
    mask_bytes3: tuple[bytes, tuple[float,float]],
    img1: schemas.Datum,
) -> list[schemas.Prediction]:
    
    mask_bytes1, shape1 = mask_bytes1
    mask_bytes2, shape2 = mask_bytes2
    mask_bytes3, shape3 = mask_bytes3

    b64_mask1 = b64encode(mask_bytes1).decode()
    b64_mask2 = b64encode(mask_bytes2).decode()
    b64_mask3 = b64encode(mask_bytes3).decode()
    return [
        schemas.Prediction(
            model_name=model_name,
            datum=img1,
            annotations=[
                schemas.PredictedAnnotation(
                    scored_labels=[
                        schemas.ScoredLabel(
                            label=schemas.Label(key="k1", value="v1"), score=0.43
                        )
                    ],
                    annotation=schemas.Annotation(
                        task_type=enums.TaskType.INSTANCE_SEGMENTATION,
                        raster=schemas.Raster(
                            mask=b64_mask1,
                            shape=shape1,                       
                        )
                    )
                ),
                schemas.PredictedAnnotation(
                    scored_labels=[
                        schemas.ScoredLabel(
                            label=schemas.Label(key="k2", value="v2"), score=0.97
                        )
                    ],
                    annotation=schemas.Annotation(
                        task_type=enums.TaskType.SEMANTIC_SEGMENTATION,
                        raster=schemas.Raster(
                            mask=b64_mask2,
                            shape=shape2,
                        )
                    )
                ),
                schemas.PredictedAnnotation(
                    scored_labels=[
                        schemas.ScoredLabel(
                            label=schemas.Label(key="k2", value="v2"), score=0.74
                        )
                    ],
                    annotation=schemas.Annotation(
                        task_type=enums.TaskType.INSTANCE_SEGMENTATION,
                        raster=schemas.Raster(
                            mask=b64_mask2,
                            shape=shape3,                       
                        )
                    )
                ),
                schemas.PredictedAnnotation(
                    scored_labels=[
                        schemas.ScoredLabel(
                            label=schemas.Label(key="k2", value="v2"), score=0.14
                        )
                    ],
                    annotation=schemas.Annotation(
                        task_type=enums.TaskType.INSTANCE_SEGMENTATION,
                        raster=schemas.Raster(
                            mask=b64_mask3,
                            shape=shape3,
                        )
                    )
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
            dataset_name=dset_name,
            datum=img1,
            annotations=[
                schemas.GroundTruthAnnotation(
                    labels=[
                        schemas.Label(key="k1", value="v1"),
                        schemas.Label(key="k2", value="v2"),
                    ],
                    annotation=schemas.Annotation(
                        task_type=enums.TaskType.CLASSIFICATION
                    )
                ),
            ]
        ),
        schemas.GroundTruth(
            dataset_name=dset_name,
            datum=img2,
            annotations=[
                schemas.GroundTruthAnnotation(
                    labels=[schemas.Label(key="k2", value="v3")],
                    annotation=schemas.Annotation(
                        task_type=enums.TaskType.CLASSIFICATION
                    )
                ),
            ]
        ),
    ]


@pytest.fixture
def pred_clfs_create(
    img1: schemas.Datum, 
    img2: schemas.Datum
) -> list[schemas.Prediction]:
    return [
        schemas.Prediction(
            model_name=model_name,
            datum=img1,
            annotations=[
                schemas.PredictedAnnotation(
                    scored_labels=[
                        schemas.ScoredLabel(
                            label=schemas.Label(key="k1", value="v1"), score=0.2
                        ),
                        schemas.ScoredLabel(
                            label=schemas.Label(key="k1", value="v2"), score=0.8
                        ),
                        schemas.ScoredLabel(
                            label=schemas.Label(key="k4", value="v4"), score=1.0
                        ),
                    ],
                    annotation=schemas.Annotation(
                        task_type=enums.TaskType.CLASSIFICATION
                    )
                ),
            ]
        ),
        schemas.Prediction(
            model_name=model_name,
            datum=img2,
            annotations=[
                schemas.PredictedAnnotation(
                    scored_labels=[
                        schemas.ScoredLabel(
                            label=schemas.Label(key="k2", value="v2"), score=1.0
                        ),
                        schemas.ScoredLabel(
                            label=schemas.Label(key="k3", value="v3"), score=0.87
                        ),
                        schemas.ScoredLabel(
                            label=schemas.Label(key="k3", value="v0"), score=0.13
                        ),
                    ],
                    annotation=schemas.Annotation(
                        task_type=enums.TaskType.CLASSIFICATION
                    )
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
def dataset_model_associations_create(
    db: Session,
    gt_dets_create: list[schemas.GroundTruth],
    pred_dets_create: list[schemas.Prediction],
    dataset_names: list[str],
    model_names: list[str],
):

    datasets = dataset_names
    models = model_names

    for name in datasets:
        crud.create_dataset(
            db,
            schemas.Dataset(name=name),
        )
        for gt in gt_dets_create:
            gt.dataset_name = name
            crud.create_groundtruth(db, gt)
        crud.finalize(db, name)

    # Create model1
    crud.create_model(
        db, model=schemas.Model(name=models[0])
    )

    # Link model1 to dataset1
    for pd in pred_dets_create:
        # pd.dataset_name = datasets[0]
        pd.model_name = models[0]
        crud.create_prediction(db, pd)
    # Finalize model1 over dataset1
    crud.finalize(
        db, 
        dataset_name=datasets[0],
        model_name=models[0], 
    )

    # Link model1 to dataset2
    for pd in pred_dets_create:
        # pds.dataset_name = datasets[1]
        pd.model_name = models[0]
        crud.create_prediction(db, pd)
    # Finalize model1 over dataset2
    crud.finalize(
        db, 
        dataset_name=datasets[1],
        model_name=models[0], 
    )

    # Create model 2
    crud.create_model(
        db, model=schemas.Model(name=models[1])
    )

    # Link model2 to dataset2
    for pd in pred_dets_create:
        # pd.dataset_name = datasets[1]
        pd.model_name = models[1]
        crud.create_prediction(db, pd)
    crud.finalize(
        db, 
        dataset_name=datasets[1],
        model_name=models[1], 
    )

    yield

    # clean up
    crud.delete_model(db, models[0])
    crud.delete_model(db, models[1])
    crud.delete_dataset(db, datasets[0])
    crud.delete_dataset(db, datasets[1])


""" CREATE """


def test_create_and_get_datasets(db: Session):

    # Create dataset
    crud.create_dataset(
        db, 
        schemas.Dataset(name=dset_name)
    )

    all_datasets = db.scalars(select(models.Dataset)).all()
    assert len(all_datasets) == 1
    assert all_datasets[0].name == dset_name

    with pytest.raises(exceptions.DatasetAlreadyExistsError) as exc_info:
        crud.create_dataset(
            db,
            schemas.Dataset(name=dset_name),
        )
    assert "already exists" in str(exc_info)

    crud.create_dataset(
        db,
        schemas.Dataset(name="other_dataset"),
    )
    datasets = crud.get_datasets(db)
    assert len(datasets) == 2
    assert set([d.name for d in datasets]) == {dset_name, "other_dataset"}


def test_create_and_get_models(db: Session):
    crud.create_model(db, schemas.Model(name=model_name))

    all_models = db.scalars(select(models.Model)).all()
    assert len(all_models) == 1
    assert all_models[0].name == model_name

    with pytest.raises(exceptions.ModelAlreadyExistsError) as exc_info:
        crud.create_model(
            db, schemas.Model(name=model_name)
        )
    assert "already exists" in str(exc_info)

    crud.create_model(
        db, schemas.Model(name="other_model")
    )
    db_models = crud.get_models(db)
    assert len(db_models) == 2
    assert set([m.name for m in db_models]) == {model_name, "other_model"}


def test_create_detection_ground_truth_and_delete_dataset(
    db: Session, 
    gt_dets_create: list[schemas.GroundTruth],
):
    # sanity check nothing in db
    check_db_empty(db=db)

    crud.create_dataset(db, schemas.Dataset(name=dset_name))


    for gt in gt_dets_create:
        crud.create_groundtruth(db, gt)

    assert db.scalar(func.count(models.Annotation.id)) == 2
    assert db.scalar(func.count(models.Datum.id)) == 2
    assert db.scalar(func.count(models.GroundTruth.id)) == 2
    assert db.scalar(func.count(models.Label.id)) == 2

    # @TODO delete after pass
    # assert db.scalar(func.count(models.GroundTruthDetection.id)) == 2
    # assert db.scalar(func.count(models.Datum.id)) == 1
    # assert db.scalar(func.count(models.LabeledGroundTruthDetection.id)) == 3
    # assert db.scalar(func.count(models.Label.id)) == 2

    # verify we get the same dets back
    for gt in gt_dets_create:
        assert gt == crud.get_groundtruth(db, gt.dataset_name, gt.datum.uid)

    # delete dataset and check the cascade worked
    crud.delete_dataset(db, dset_name)
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
    with pytest.raises(exceptions.ModelDoesNotExistError) as exc_info:
        for pd in pred_dets_create:
            crud.create_prediction(db, pd)
    assert "does not exist" in str(exc_info)

    # create model
    crud.create_model(db, schemas.Model(name=model_name))

    # check this gives an error since the datums haven't been added yet
    with pytest.raises(exceptions.ImageDoesNotExistError) as exc_info:
        for pd in pred_dets_create:
            pd.model_name=model_name
            crud.create_prediction(db, pd)
    assert "Image with uid" in str(exc_info)

    # create dataset, add images, and add predictions
    crud.create_dataset(db, schemas.Dataset(name=dset_name))

    crud.create_groundtruth(db, gt_dets_create)
    crud.create_prediction(db, pred_dets_create)

    # check db has the added predictions
    assert db.scalar(func.count(models.Annotation.id)) == 2
    assert db.scalar(func.count(models.Prediction.id)) == 3

    # delete model and check all detections from it are gone
    crud.delete_model(db, model_name)
    assert db.scalar(func.count(models.Model.id)) == 0
    assert db.scalar(func.count(models.Annotation.id)) == 0
    assert db.scalar(func.count(models.Prediction.id)) == 0


def test_create_detections_as_bbox_or_poly(db: Session, img1: schemas.Datum):
    xmin, ymin, xmax, ymax = 50, 70, 120, 300
    
    det1 = schemas.GroundTruthAnnotation(
        labels=[schemas.Label(key="k", value="v")],
        annotation=schemas.Annotation(
            task_type=enums.TaskType.DETECTION,
            polygon=schemas.Polygon(
                boundary=schemas.BasicPolygon(
                    points=[
                        schemas.Point(x=xmin, y=ymin),
                        schemas.Point(x=xmax, y=ymin),
                        schemas.Point(x=xmax, y=ymax),
                        schemas.Point(x=xmin, y=ymax),
                    ]
                )
            )
        )
    )

    det2 = schemas.GroundTruthAnnotation(
        labels=[schemas.Label(key="k", value="v")],
        annotation=schemas.Annotation(
            task_type=enums.TaskType.DETECTION,
            bounding_box=schemas.BoundingBox.from_extrema(
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
            )
        )
    )

    crud.create_dataset(db, schemas.Dataset(name=dset_name))

    crud.create_groundtruth(
        db,
        schemas.GroundTruth(
            dataset_name=dset_name, 
            datum=img1,
            annotations=[det1, det2],
        ),
    )

    dets = db.scalars(select(models.GroundTruth)).all()
    assert len(dets) == 2
    assert set([det.annotation.box is not None for det in dets]) == {True, False}

    # check we get the same polygon
    assert db.scalar(ST_AsText(dets[0].annotation.polygon)) == db.scalar(
        ST_AsText(dets[1].annotation.box)
    )


def test_create_classification_groundtruth_and_delete_dataset(
    db: Session, gt_clfs_create: list[schemas.GroundTruth]
):
    crud.create_dataset(db, schemas.Dataset(name=dset_name))


    for gt in gt_clfs_create:
        gt.dataset_name=dset_name
        crud.create_groundtruth(db, gt)

    # should have three GroundTruthClassification rows since one image has two
    # labels and the other has one
    assert db.scalar(func.count(models.GroundTruth.id)) == 3
    assert db.scalar(func.count(models.Datum.id)) == 2
    assert db.scalar(func.count(models.Label.id)) == 3

    # delete dataset and check the cascade worked
    crud.delete_dataset(db, dset_name)
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
    with pytest.raises(exceptions.ModelDoesNotExistError) as exc_info:
        crud.create_prediction(db, pred_clfs_create[0])
    assert "does not exist" in str(exc_info)

    crud.create_model(db, schemas.Model(name=model_name))

    # check this gives an error since the images haven't been added yet
    with pytest.raises(exceptions.DatumDoesNotExistError) as exc_info:
        crud.create_prediction(db, pred_clfs_create[0])
    assert "Datum with uid" in str(exc_info)

    # create dataset, add images, and add predictions
    crud.create_dataset(db, schemas.Dataset(name=dset_name))


    for gt in gt_clfs_create:
        gt.dataset_name=dset_name
        crud.create_groundtruth(db, gt)

    for pd in pred_clfs_create:
        pd.model_name=model_name
        crud.create_prediction(db, pd)

    # check db has the added predictions
    assert db.scalar(func.count(models.Prediction.id)) == 6

    # delete model and check all detections from it are gone
    crud.delete_model(db, model_name)
    assert db.scalar(func.count(models.Model.id)) == 0
    assert db.scalar(func.count(models.Annotation.id)) == 2
    assert db.scalar(func.count(models.Prediction.id)) == 0

    # delete dataset and check
    crud.delete_dataset(db, dset_name)
    assert db.scalar(func.count(models.Model.id)) == 0
    assert db.scalar(func.count(models.Annotation.id)) == 0
    assert db.scalar(func.count(models.GroundTruth.id)) == 0



def test_create_groundtruth_segmentations_and_delete_dataset(
    db: Session, gt_segs_create: list[schemas.GroundTruth]
):
    # sanity check nothing in db
    check_db_empty(db=db)

    crud.create_dataset(db, schemas.Dataset(name=dset_name))


    for gt in gt_segs_create:
        gt.dataset_name=dset_name
        crud.create_groundtruth(db, data=gt)

    assert db.scalar(func.count(models.Annotation.id)) == 4
    assert db.scalar(func.count(models.Datum.id)) == 2
    assert db.scalar(func.count(models.GroundTruth.id)) == 4
    assert db.scalar(func.count(models.Label.id)) == 2

    # delete dataset and check the cascade worked
    crud.delete_dataset(db, dataset_name=dset_name)
    for model_cls in [
        models.Dataset,
        models.Datum,
        models.Annotation,
        models.GroundTruth,
    ]:
        assert db.scalar(func.count(model_cls.id)) == 0

    # make sure labels are still there`
    assert db.scalar(func.count(models.Label.id)) == 2


def test_create_predicted_segmentations_check_area_and_delete_model(
    db: Session,
    pred_segs_create: list[schemas.Prediction],
    gt_segs_create: list[schemas.GroundTruth],
):
    # check this gives an error since the model hasn't been added yet
    with pytest.raises(exceptions.ModelDoesNotExistError) as exc_info:
        for pd in pred_segs_create:
            crud.create_prediction(db, pd)
    assert "does not exist" in str(exc_info)

    crud.create_model(db, schemas.Model(name=model_name))

    # check this gives an error since the images haven't been added yet
    with pytest.raises(exceptions.ImageDoesNotExistError) as exc_info:
        for pd in pred_segs_create:
            pd.model_name=model_name
            crud.create_prediction(db, pd)
    assert "Image with uid" in str(exc_info)

    # create dataset, add images, and add predictions
    crud.create_dataset(db, schemas.Dataset(name=dset_name))

    for gt in gt_segs_create:
        gt.dataset_name=dset_name
        crud.create_groundtruth(db, gt)\
        
    for pd in pred_segs_create:
        pd.model_name=model_name
        crud.create_prediction(db, pd)

    # check db has the added predictions
    assert db.scalar(func.count(models.Annotation.id)) == 4
    assert db.scalar(func.count(models.Prediction.id)) == 4

    # grab the first one and check that the area of the raster
    # matches the area of the image
    img = crud.get_groundtruth(db, datum_uid="uid1", dataset_name=dset_name)
    seg = img.annotations[0].annotation.raster
    mask = bytes_to_pil(
        b64decode(pred_segs_create[0].annotations[0].annotation.raster)
    )
    assert db.scalar(ST_Count(seg)) == np.array(mask).sum()

    # delete model and check all detections from it are gone
    crud.delete_model(db, model_name)
    assert db.scalar(func.count(models.Model.id)) == 0
    assert db.scalar(func.count(models.Annotation.id)) == 0
    assert db.scalar(func.count(models.Prediction.id)) == 0


def test_segmentation_area_no_hole(
    db: Session,
    poly_without_hole: schemas.Polygon,
    img1: schemas.Datum,
):
    # sanity check nothing in db
    check_db_empty(db=db)

    crud.create_dataset(db, schemas.Dataset(name=dset_name))

    crud.create_groundtruth(
        db,
        schemas.GroundTruth(
            dataset_name=dset_name,
            datum=img1,
            annotations=[
                schemas.Annotation(
                    labels=[schemas.Label(key="k1", value="v1")],
                    annotation=schemas.Annotation(
                        task_type=enums.TaskType.INSTANCE_SEGMENTATION,
                        multipolygon=schemas.MultiPolygon(
                            polygons=[poly_without_hole]
                        )                        
                    ),
                )
            ],
        ),
    )

    segmentation = db.scalar(select(models.Annotation))

    assert db.scalar(ST_Count(segmentation.raster)) == math.ceil(
        45.5
    )  # area of mask will be an int


def test_segmentation_area_with_hole(
    db: Session, poly_with_hole: schemas.Polygon, img1: schemas.Datum
):
    # sanity check nothing in db
    check_db_empty(db=db)

    crud.create_dataset(db, schemas.Dataset(name=dset_name))

    crud.create_groundtruth(
        db,
        schemas.GroundTruth(
            dataset_name=dset_name,
            datum=img1,
            annotations=[
                schemas.GroundTruthAnnotation(
                    labels=[schemas.Label(key="k1", value="v1")],
                    annotation=schemas.Annotation(
                        task_type=enums.TaskType.SEMANTIC_SEGMENTATION,
                        multipolygon=schemas.MultiPolygon(
                            polygons=[poly_with_hole],
                        )
                    )
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
    img1: schemas.Datum,
):
    # sanity check nothing in db
    check_db_empty(db=db)

    crud.create_dataset(db, schemas.Dataset(name=dset_name))

    crud.create_groundtruth(
        db,
        schemas.GroundTruth(
            dataset_name=dset_name,
            datum=img1,
            annotations=[
                schemas.GroundTruthAnnotation(
                    labels=[schemas.Label(key="k1", value="v1")],                    
                    annotation=schemas.Annotation(
                        task_type=enums.TaskType.INSTANCE_SEGMENTATION,
                        multipolygon=schemas.MultiPolygon(
                            polygons=[poly_with_hole, poly_without_hole],
                        )
                    )
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
#     db: Session, poly_with_hole: schemas.Polygon, img: models.Datum
# ):
#     gt_seg = db.scalar(
#         insert(models.Annotation)
#         .values(
#             [
#                 {
#                     "raster": crud._create._select_statement_from_poly(
#                         [poly_with_hole]
#                     ),
#                     "datum_id": img.id,
#                     "is_instance": True,
#                 }
#             ]
#         )
#         .returning(models.Annotation)
#     )
#     db.add(gt_seg)
#     db.commit()

#     wkt = db.scalar(ST_AsText(ST_Polygon(gt_seg.shape)))

#     # note the hole, which is a triangle, is jagged due to aliasing
#     assert (
#         wkt
#         == "MULTIPOLYGON(((0 0,0 10,10 10,10 0,0 0),(2 4,2 8,3 8,3 7,4 7,4 6,5 6,5 5,6 5,6 4,2 4)))"
#     )


def test_gt_seg_as_mask_or_polys(db: Session):
    """Check that a groundtruth segmentation can be created as a polygon or mask"""
    xmin, xmax, ymin, ymax = 11, 45, 37, 102
    h, w = 150, 200
    mask = np.zeros((h, w), dtype=bool)
    mask[ymin:ymax, xmin:xmax] = True
    mask_b64 = b64encode(np_to_bytes(mask)).decode()

    img = schemas.Image(
        uid="uid", 
        height=h, 
        width=w,
        frame=0,
    ).to_datum()

    poly = schemas.BasicPolygon(
        points=[
            schemas.Point(x=xmin, y=ymin), 
            schemas.Point(x=xmin, y=ymax), 
            schemas.Point(x=xmax, y=ymax), 
            schemas.Point(x=xmax, y=ymin),
        ]
    )

    gt1 = schemas.GroundTruthAnnotation(
        labels=[schemas.Label(key="k1", value="v1")],
        annotation=schemas.Annotation(
            task_type=enums.TaskType.SEMANTIC_SEGMENTATION,
            raster=schemas.Raster(
                mask=mask_b64,
                height=h,
                width=w,
            )
        )
    )
    gt2 = schemas.GroundTruthAnnotation(
        labels=[schemas.Label(key="k1", value="v1")],
        annotation=schemas.Annotation(
            task_type=enums.TaskType.INSTANCE_SEGMENTATION,
            multipolygon=schemas.MultiPolygon(
                polygons=[
                    schemas.Polygon(boundary=poly)
                ]
            )
        )
    )
    gt = schemas.GroundTruth(
        dataset_name=dset_name, 
        datum=img,
        annotations=[gt1, gt2],
    )

    check_db_empty(db=db)

    crud.create_dataset(db, schemas.Dataset(name=dset_name))

    crud.create_groundtruth(db, gt)

    shapes = db.scalars(
        select(ST_AsText(ST_Polygon(models.Annotation.raster)))
    ).all()

    assert len(shapes) == 2
    # check that the mask and polygon define the same polygons
    assert shapes[0] == shapes[1]

    # verify we get the same segmentations back
    segs = crud.get_groundtruth(
        db, 
        datum_uid=img.uid, 
        dataset_name=dset_name,
        filter_by_task_type=[enums.TaskType.INSTANCE_SEGMENTATION],
    )
    assert len(segs.annotations) == 1  # should just be one instance segmentation
    decoded_mask = bytes_to_pil(b64decode(segs.annotations[0].annotation.raster.mask))
    decoded_mask_arr = np.array(decoded_mask)

    np.testing.assert_equal(decoded_mask_arr, mask)
    assert segs.datum == gt.datum
    assert segs.annotations[0].labels == gt.annotations[0].labels

# @NOTE: This funcionality now belongs to the backend
# @TODO: This is replaced by `crud.get_disjoint_labels`
# def test_get_filtered_preds_statement_and_missing_labels(
#     db: Session,
#     gt_segs_create: list[schemas.GroundTruth],
#     pred_segs_create: list[schemas.Prediction],
# ):
#     crud.create_dataset(db, schemas.Dataset(name=dset_name))

#     crud.create_model(db, schemas.Model(name=model_name))

#     # add three total ground truth segmentations, two of which are instance segmentations with
#     # the same label.
#     for gt in gt_segs_create:
#         crud.create_groundtruth(db, data=gt)
#     # add three total predicted segmentations, two of which are instance segmentations
#     for pd in pred_segs_create:
#         crud.create_prediction(db, pd)

#     gts_statement = crud._create._instance_segmentations_in_dataset_statement(
#         dset_name
#     )
#     preds_statement = crud._read._model_instance_segmentation_preds_statement(
#         model_name=model_name, dataset_name=dset_name
#     )

#     gts = db.scalars(gts_statement).all()
#     preds = db.scalars(preds_statement).all()

#     assert len(gts) == 3
#     assert len(preds) == 3

#     labels = crud._create._labels_in_query(db, gts_statement)
#     assert len(labels) == 2

#     # check get everything if the requested labels argument is empty
#     (
#         new_preds_statement,
#         missing_pred_labels,
#         ignored_pred_labels,
#     ) = crud.get_filtered_preds_statement_and_missing_labels(
#         db=db, gts_statement=gts_statement, preds_statement=preds_statement
#     )

#     gts = db.scalars(gts_statement).all()
#     preds = db.scalars(new_preds_statement).all()

#     assert len(gts) == 3
#     # should not get the pred with label "k2", "v2" since its not
#     # present in the groundtruths
#     assert len(preds) == 1
#     assert missing_pred_labels == [schemas.Label(key="k3", value="v3")]
#     assert ignored_pred_labels == [schemas.Label(key="k2", value="v2")]


def test_create_ap_metrics(db: Session, groundtruths, predictions):
    # the groundtruths and predictions arguments are not used but
    # those fixtures create the necessary dataset, model, groundtruths, and predictions

    def method_to_test(
        label_key: str, min_area: float = None, max_area: float = None
    ):
        request_info = schemas.APRequest(
            settings=schemas.EvaluationSettings(
                model_name="test model",
                dataset_name="test dataset",
                min_area=min_area,
                max_area=max_area,
                task_type=enums.TaskType.DETECTION,
                gt_type=enums.AnnotationType.BOX,
                pd_type=enums.AnnotationType.BOX,
                label_key=label_key,
            ),
            iou_thresholds=[0.2, 0.6],
            ious_to_keep=[0.2],
        )

        disjoint_labels = crud.get_disjoint_labels(
            db, 
            dataset_name=request_info.settings.dataset_name,
            model_name=request_info.settings.model_name,
        )
        missing_pred_labels = disjoint_labels["dataset"]
        ignored_pred_labels = disjoint_labels["models"]

        return (
            crud.create_ap_metrics(
                db,
                request_info=request_info,
            ),
            missing_pred_labels,
            ignored_pred_labels,
        )

    # check we get an error since the dataset is still a draft
    with pytest.raises(exceptions.DatasetIsNotFinalizedError):
        method_to_test(label_key="class")

    # finalize dataset
    crud.finalize(db, dataset_name="test dataset")

    # now if we try again we should get an error that inferences aren't finalized
    with pytest.raises(exceptions.InferencesAreNotFinalizedError):
        method_to_test(label_key="class")

    # verify we have no evaluations yet
    assert len(crud.get_model_evaluation_settings(db, model_name)) == 0

    # finalize inferences and try again
    crud.finalize(db, model_name=model_name, dataset_name=dset_name)

    (
        evaluation_settings_id,
        missing_pred_labels,
        ignored_pred_labels,
    ) = method_to_test(label_key="class")

    # check we have one evaluation
    assert len(crud.get_model_evaluation_settings(db, model_name)) == 1

    assert missing_pred_labels == []
    assert ignored_pred_labels == [schemas.Label(key="class", value="3")]

    metrics = db.scalar(
        select(models.EvaluationSettings).where(
            models.EvaluationSettings.id == evaluation_settings_id
        )
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
    pydantic_metrics = crud.get_metrics_from_evaluation_settings_id(
        db, evaluation_settings_id
    )
    for m in pydantic_metrics:
        assert isinstance(m, schemas.Metric)
    assert len(pydantic_metrics) == len(metric_ids)

    # run again and make sure no new ids were created
    evaluation_settings_id_again, _, _ = method_to_test(label_key="class")
    assert evaluation_settings_id == evaluation_settings_id_again
    metric_ids_again = [
        m.id
        for m in db.scalar(
            select(models.EvaluationSettings).where(
                models.EvaluationSettings.id == evaluation_settings_id_again
            )
        ).metrics
    ]
    assert sorted(metric_ids) == sorted(metric_ids_again)

    # test crud.get_model_metrics
    metrics_pydantic = crud.get_model_metrics(
        db, "test model", evaluation_settings_id
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
        evaluation_settings_id,
        missing_pred_labels,
        ignored_pred_labels,
    ) = method_to_test(label_key="class", min_area=min_area, max_area=max_area)

    metrics_pydantic = crud.get_model_metrics(
        db, "test model", evaluation_settings_id
    )
    for m in metrics_pydantic:
        assert m.type in {
            "AP",
            "APAveragedOverIOUs",
            "mAP",
            "mAPAveragedOverIOUs",
        }

    # check we have the right evaluations
    model_evals = crud.get_model_evaluation_settings(db, model_name)
    assert len(model_evals) == 2
    assert model_evals[0] == schemas.EvaluationSettings(
        model_name=model_name,
        dataset_name=dset_name,
        task_type=enums.TaskType.DETECTION,
        gt_type=enums.AnnotationType.BOX,
        pd_type=enums.AnnotationType.BOX,
        label_key="class",
        id=1,
    )
    assert model_evals[1] == schemas.EvaluationSettings(
        model_name=model_name,
        dataset_name=dset_name,
        task_type=enums.TaskType.DETECTION,
        gt_type=enums.AnnotationType.BOX,
        pd_type=enums.AnnotationType.BOX,
        label_key="class",
        min_area=min_area,
        max_area=max_area,
        id=2,
    )


def test_create_clf_metrics(
    db: Session, 
    gt_clfs_create: list[schemas.GroundTruth], 
    pred_clfs_create: list[schemas.Prediction],
):
    crud.create_dataset(
        db,
        dataset=schemas.Dataset(name=dset_name),
    )
    for gt in gt_clfs_create:
        gt.dataset_name = dset_name
        crud.create_groundtruth(db, gt)
    crud.finalize(db, dset_name)

    crud.create_model(db, schemas.Model(name=model_name))
    for pd in pred_clfs_create:
        pd.model_name = model_name
        crud.create_prediction(db, pd)
    crud.finalize(db, model_name, dset_name)

    request_info = schemas.ClfMetricsRequest(
        settings=schemas.EvaluationSettings(
            model_name=model_name, 
            dataset_name=dset_name
        )
    )

    disjoint_labels = crud.get_disjoint_labels(
        db, 
        dataset_name=request_info.settings.dataset_name,
        model_name=request_info.settings.model_name,
    )
    missing_pred_keys = disjoint_labels["dataset"]
    ignored_pred_keys = disjoint_labels["model"]

    assert missing_pred_keys == []
    assert set(ignored_pred_keys) == {"k3", "k4"}

    evaluation_settings_id = crud.create_clf_metrics(db, request_info)

    # check we have one evaluation
    assert len(crud.get_model_evaluation_settings(db, model_name)) == 1

    # get all metrics
    metrics = db.scalar(
        select(models.EvaluationSettings).where(
            models.EvaluationSettings.id == evaluation_settings_id
        )
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
            ("k2", "v2"),
            ("k2", "v3"),
            ("k1", "v2"),
        }

    confusion_matrices = db.scalars(
        select(models.ConfusionMatrix).where(
            models.ConfusionMatrix.evaluation_settings_id
            == evaluation_settings_id
        )
    ).all()

    # should have two confusion matrices, one for each key
    assert len(confusion_matrices) == 2

    # test getting metrics from evaluation settings id
    pydantic_metrics = crud.get_metrics_from_evaluation_settings_id(
        db, evaluation_settings_id
    )
    for m in pydantic_metrics:
        assert isinstance(m, schemas.Metric)
    assert len(pydantic_metrics) == len(metrics)

    # test getting confusion matrices from evaluation settings id
    cms = crud.get_confusion_matrices_from_evaluation_settings_id(
        db, evaluation_settings_id
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

    # run again and check we still have one evaluation and the same number of metrics
    # and confusion matrices
    crud.create_clf_metrics(db, request_info)
    assert len(crud.get_model_evaluation_settings(db, model_name)) == 1
    metrics = db.scalar(
        select(models.EvaluationSettings).where(
            models.EvaluationSettings.id == evaluation_settings_id
        )
    ).metrics
    assert len(metrics) == 2 + 2 + 4 + 4 + 4
    confusion_matrices = db.scalars(
        select(models.ConfusionMatrix).where(
            models.ConfusionMatrix.evaluation_settings_id
            == evaluation_settings_id
        )
    ).all()
    assert len(confusion_matrices) == 2


# @NOTE: This should now be handled by `velour_api.schemas.Raster`
# def test__raster_to_png_b64(db: Session):
#     # create a mask consisting of an ellipse with a whole in it
#     w, h = 50, 100
#     img = Image.new("1", size=(w, h))
#     draw = ImageDraw.Draw(img)
#     draw.ellipse((15, 40, 30, 70), fill=True)
#     draw.ellipse((20, 50, 25, 60), fill=False)

#     f = io.BytesIO()
#     img.save(f, format="PNG")
#     f.seek(0)
#     b64_mask = b64encode(f.read()).decode()

#     image = schemas.Datum(uid="uid", height=h, width=w)
#     crud.create_dataset(db, schemas.Dataset(name=dset_name))

#     crud.create_groundtruth(
#         db,
#         data=list[schemas.GroundTruth](
#             dataset_name=dset_name,
#             segmentations=[
#                 schemas.Annotation(
#                     shape=b64_mask,
#                     image=image,
#                     labels=[schemas.Label(key="k", value="v")],
#                     is_instance=True,
#                 )
#             ],
#         ),
#     )

#     seg = db.scalar(select(models.Annotation))

#     assert b64_mask == _raster_to_png_b64(db, seg.shape, image)


# @NOTE: This is now handled by `velour_api.backend.metrics.detections`
# def test__instance_segmentations_in_dataset_statement(
#     db: Session, gt_segs_create: list[schemas.GroundTruth]
# ):
#     crud.create_dataset(db, schemas.Dataset(name=dset_name))

#     for gt in gt_segs_create:
#         crud.create_groundtruth(db, data=gt)

#     areas = db.scalars(
#         select(ST_Count(models.Annotation.shape)).where(
#             models.Annotation.is_instance
#         )
#     ).all()

#     assert sorted(areas) == [46, 90, 136]

#     # sanity check no min_area and max_area arguments
#     stmt = _instance_segmentations_in_dataset_statement(dataset_name=dset_name)
#     assert len(db.scalars(stmt).all()) == 3

#     # check min_area arg
#     stmt = _instance_segmentations_in_dataset_statement(
#         dataset_name=dset_name, min_area=45
#     )
#     assert len(db.scalars(stmt).all()) == 3
#     stmt = _instance_segmentations_in_dataset_statement(
#         dataset_name=dset_name, min_area=137
#     )
#     assert len(db.scalars(stmt).all()) == 0

#     # check max_area argument
#     stmt = _instance_segmentations_in_dataset_statement(
#         dataset_name=dset_name, max_area=45
#     )
#     assert len(db.scalars(stmt).all()) == 0

#     stmt = _instance_segmentations_in_dataset_statement(
#         dataset_name=dset_name, max_area=136
#     )
#     assert len(db.scalars(stmt).all()) == 3

#     # check specifying both min size and max size
#     stmt = _instance_segmentations_in_dataset_statement(
#         dataset_name=dset_name, min_area=45, max_area=136
#     )
#     assert len(db.scalars(stmt).all()) == 3
#     stmt = _instance_segmentations_in_dataset_statement(
#         dataset_name=dset_name, min_area=50, max_area=100
#     )
#     assert len(db.scalars(stmt).all()) == 1


# @NOTE: Moved to `velour_api.backend.metrics.detections`
# def test___model_instance_segmentation_preds_statement(
#     db: Session,
#     gt_segs_create: list[schemas.GroundTruth],
#     pred_segs_create: list[schemas.Prediction],
# ):
#     crud.create_dataset(db, schemas.Dataset(name=dset_name))

#     crud.create_model(db, schemas.Model(name=model_name))
#     for gt in gt_segs_create:
#         crud.create_groundtruth(db, data=gt)
#     for pd in pred_segs_create:
#         crud.create_prediction(db, pd)

#     areas = db.scalars(
#         select(ST_Count(models.Annotation.shape)).where(
#             models.Annotation.is_instance
#         )
#     ).all()

#     assert sorted(areas) == [95, 279, 1077]

#     # sanity check no min_area and max_area arguments
#     stmt = _model_instance_segmentation_preds_statement(
#         dataset_name=dset_name, model_name=model_name
#     )
#     assert len(db.scalars(stmt).all()) == 3

#     # check min_area arg
#     stmt = _model_instance_segmentation_preds_statement(
#         dataset_name=dset_name, model_name=model_name, min_area=94
#     )
#     assert len(db.scalars(stmt).all()) == 3
#     stmt = _model_instance_segmentation_preds_statement(
#         dataset_name=dset_name, model_name=model_name, min_area=1078
#     )
#     assert len(db.scalars(stmt).all()) == 0

#     # check max_area argument
#     stmt = _model_instance_segmentation_preds_statement(
#         dataset_name=dset_name, model_name=model_name, max_area=94
#     )
#     assert len(db.scalars(stmt).all()) == 0

#     stmt = _model_instance_segmentation_preds_statement(
#         dataset_name=dset_name, model_name=model_name, max_area=1078
#     )
#     assert len(db.scalars(stmt).all()) == 3

#     # check specifying both min size and max size
#     stmt = _model_instance_segmentation_preds_statement(
#         dataset_name=dset_name,
#         model_name=model_name,
#         min_area=94,
#         max_area=1078,
#     )
#     assert len(db.scalars(stmt).all()) == 3
#     stmt = _model_instance_segmentation_preds_statement(
#         dataset_name=dset_name,
#         model_name=model_name,
#         min_area=200,
#         max_area=300,
#     )
#     assert len(db.scalars(stmt).all()) == 1


# @NOTE: Moved to `velour_api.backend.metrics.detections`
# def test___object_detections_in_dataset_statement(db: Session, groundtruths):
#     # the groundtruths argument is not used but that fixture creates groundtruth
#     # detections in the database

#     areas = db.scalars(ST_Area(models.GroundTruthDetection.boundary)).all()

#     # these are just to establish what the bounds on the areas of
#     # the groundtruth detections are
#     assert len(areas) == 20
#     assert min(areas) > 94
#     assert max(areas) < 326771
#     assert len([a for a in areas if a > 500 and a < 1200]) == 9

#     # sanity check no min_area and max_area arguments
#     stmt = _object_detections_in_dataset_statement(
#         dataset_name=dset_name, task=enums.Task.BBOX_OBJECT_DETECTION
#     )
#     assert len(db.scalars(stmt).all()) == 20

#     # check min_area arg
#     stmt = _object_detections_in_dataset_statement(
#         dataset_name=dset_name,
#         min_area=93,
#         task=enums.Task.BBOX_OBJECT_DETECTION,
#         task_for_area_computation=enums.Task.BBOX_OBJECT_DETECTION,
#     )
#     assert len(db.scalars(stmt).all()) == 20
#     stmt = _object_detections_in_dataset_statement(
#         dataset_name=dset_name,
#         min_area=326771,
#         task=enums.Task.BBOX_OBJECT_DETECTION,
#         task_for_area_computation=enums.Task.BBOX_OBJECT_DETECTION,
#     )
#     assert len(db.scalars(stmt).all()) == 0

#     # check max_area argument
#     stmt = _object_detections_in_dataset_statement(
#         dataset_name=dset_name,
#         max_area=93,
#         task=enums.Task.BBOX_OBJECT_DETECTION,
#         task_for_area_computation=enums.Task.BBOX_OBJECT_DETECTION,
#     )
#     assert len(db.scalars(stmt).all()) == 0

#     stmt = _object_detections_in_dataset_statement(
#         dataset_name=dset_name,
#         max_area=326771,
#         task=enums.Task.BBOX_OBJECT_DETECTION,
#         task_for_area_computation=enums.Task.BBOX_OBJECT_DETECTION,
#     )
#     assert len(db.scalars(stmt).all()) == 20

#     # check specifying both min size and max size
#     stmt = _object_detections_in_dataset_statement(
#         dataset_name=dset_name,
#         min_area=94,
#         max_area=326771,
#         task=enums.Task.BBOX_OBJECT_DETECTION,
#         task_for_area_computation=enums.Task.BBOX_OBJECT_DETECTION,
#     )
#     assert len(db.scalars(stmt).all()) == 20
#     stmt = _object_detections_in_dataset_statement(
#         dataset_name=dset_name,
#         min_area=500,
#         max_area=1200,
#         task=enums.Task.BBOX_OBJECT_DETECTION,
#         task_for_area_computation=enums.Task.BBOX_OBJECT_DETECTION,
#     )
#     assert len(db.scalars(stmt).all()) == 9


# @NOTE: Moved to `velour_api.backend.metrics.detections`
# def test__model_object_detection_preds_statement(
#     db: Session, groundtruths, predictions
# ):
#     # the groundtruths and predictions arguments are not used but the fixtures create predicted
#     # detections in the database

#     areas = db.scalars(ST_Area(models.Annotation.boundary)).all()

#     # these are just to establish what the bounds on the areas of
#     # the groundtruth detections are
#     assert len(areas) == 19
#     assert min(areas) > 94
#     assert max(areas) < 307274
#     assert len([a for a in areas if a > 500 and a < 1200]) == 9

#     # sanity check no min_area and max_area arguments
#     stmt = _model_object_detection_preds_statement(
#         dataset_name=dset_name,
#         model_name=model_name,
#         task=enums.Task.BBOX_OBJECT_DETECTION,
#     )
#     assert len(db.scalars(stmt).all()) == 19

#     # check min_area arg
#     stmt = _model_object_detection_preds_statement(
#         dataset_name=dset_name,
#         model_name=model_name,
#         min_area=93,
#         task=enums.Task.BBOX_OBJECT_DETECTION,
#         task_for_area_computation=enums.Task.BBOX_OBJECT_DETECTION,
#     )
#     assert len(db.scalars(stmt).all()) == 19
#     stmt = _model_object_detection_preds_statement(
#         dataset_name=dset_name,
#         model_name=model_name,
#         min_area=326771,
#         task=enums.Task.BBOX_OBJECT_DETECTION,
#         task_for_area_computation=enums.Task.BBOX_OBJECT_DETECTION,
#     )
#     assert len(db.scalars(stmt).all()) == 0

#     # check max_area argument
#     stmt = _model_object_detection_preds_statement(
#         dataset_name=dset_name,
#         model_name=model_name,
#         max_area=93,
#         task=enums.Task.BBOX_OBJECT_DETECTION,
#         task_for_area_computation=enums.Task.BBOX_OBJECT_DETECTION,
#     )
#     assert len(db.scalars(stmt).all()) == 0

#     stmt = _model_object_detection_preds_statement(
#         dataset_name=dset_name,
#         model_name=model_name,
#         max_area=326771,
#         task=enums.Task.BBOX_OBJECT_DETECTION,
#         task_for_area_computation=enums.Task.BBOX_OBJECT_DETECTION,
#     )
#     assert len(db.scalars(stmt).all()) == 19

#     # check specifying both min size and max size
#     stmt = _model_object_detection_preds_statement(
#         dataset_name=dset_name,
#         model_name=model_name,
#         min_area=94,
#         max_area=326771,
#         task=enums.Task.BBOX_OBJECT_DETECTION,
#         task_for_area_computation=enums.Task.BBOX_OBJECT_DETECTION,
#     )
#     assert len(db.scalars(stmt).all()) == 19
#     stmt = _model_object_detection_preds_statement(
#         dataset_name=dset_name,
#         model_name=model_name,
#         min_area=500,
#         max_area=1200,
#         task=enums.Task.BBOX_OBJECT_DETECTION,
#         task_for_area_computation=enums.Task.BBOX_OBJECT_DETECTION,
#     )
#     assert len(db.scalars(stmt).all()) == 9


# @NOTE: Moved to `velour_api.backend.metrics.detections`
# def test__filter_instance_segmentations_by_area(db: Session):
#     crud.create_dataset(db, schemas.Dataset(name=dset_name))

#     # triangle of area 150
#     poly1 = schemas.Polygon(polygon=[(10, 20), (10, 40), (25, 20)])
#     # rectangle of area 1050
#     poly2 = schemas.Polygon(
#         polygon=[(0, 5), (0, 40), (30, 40), (30, 5)]
#     )

#     img = schemas.Datum(uid="", height=1000, width=2000)

#     crud.create_groundtruth(
#         db,
#         data=list[schemas.GroundTruth](
#             dataset_name=dset_name,
#             segmentations=[
#                 schemas.Annotation(
#                     shape=[poly1],
#                     image=img,
#                     labels=[schemas.Label(key="k", value="v")],
#                     is_instance=True,
#                 ),
#                 schemas.Annotation(
#                     shape=[poly2],
#                     image=img,
#                     labels=[schemas.Label(key="k", value="v")],
#                     is_instance=True,
#                 ),
#             ],
#         ),
#     )

#     areas = db.scalars(ST_Count(models.Annotation.shape)).all()
#     assert sorted(areas) == [150, 1050]

#     base_stmt = "SELECT id FROM ground_truth_segmentation WHERE ground_truth_segmentation.is_instance"

#     # check filtering when use area determined by instance segmentation task
#     stmt = _filter_instance_segmentations_by_area(
#         stmt=base_stmt,
#         seg_table=models.Annotation,
#         task_for_area_computation=enums.Task.INSTANCE_SEGMENTATION,
#         min_area=100,
#         max_area=2000,
#     )
#     assert len(db.scalars(stmt).all()) == 2

#     stmt = _filter_instance_segmentations_by_area(
#         base_stmt,
#         seg_table=models.Annotation,
#         task_for_area_computation=enums.Task.INSTANCE_SEGMENTATION,
#         min_area=100,
#         max_area=200,
#     )
#     assert len(db.scalars(stmt).all()) == 1

#     stmt = _filter_instance_segmentations_by_area(
#         base_stmt,
#         seg_table=models.Annotation,
#         task_for_area_computation=enums.Task.INSTANCE_SEGMENTATION,
#         min_area=151,
#         max_area=2000,
#     )
#     assert len(db.scalars(stmt).all()) == 1

#     # now when we use bounding box detection task, the triangle becomes its circumscribing
#     # rectangle (with area ~300) so we should get both segmentations
#     stmt = _filter_instance_segmentations_by_area(
#         base_stmt,
#         seg_table=models.Annotation,
#         task_for_area_computation=enums.Task.BBOX_OBJECT_DETECTION,
#         min_area=280,
#         max_area=2000,
#     )
#     assert len(db.scalars(stmt).all()) == 2

#     stmt = _filter_instance_segmentations_by_area(
#         base_stmt,
#         seg_table=models.Annotation,
#         task_for_area_computation=enums.Task.BBOX_OBJECT_DETECTION,
#         min_area=301,
#         max_area=2000,
#     )
#     assert len(db.scalars(stmt).all()) == 1

#     # if we use polygon detection then the areas shouldn't change much (the area
#     # of the triangle actually becomes 163-- not sure if this is aliasing or what)
#     stmt = _filter_instance_segmentations_by_area(
#         base_stmt,
#         seg_table=models.Annotation,
#         task_for_area_computation=enums.Task.POLY_OBJECT_DETECTION,
#         min_area=149,
#         max_area=2000,
#     )
#     assert len(db.scalars(stmt).all()) == 2

#     stmt = _filter_instance_segmentations_by_area(
#         base_stmt,
#         seg_table=models.Annotation,
#         task_for_area_computation=enums.Task.POLY_OBJECT_DETECTION,
#         min_area=164,
#         max_area=2000,
#     )

#     assert len(db.scalars(stmt).all()) == 1


# @NOTE: Moved to `velour_api.backend.metrics.detections`
# def test__filter_object_detections_by_area(db: Session):
#     crud.create_dataset(db, schemas.Dataset(name=dset_name))

#     # triangle of area 150
#     boundary1 = [(10, 20), (10, 40), (25, 20)]
#     # rectangle of area 1050
#     boundary2 = [(0, 5), (0, 40), (30, 40), (30, 5)]

#     img = schemas.Datum(uid="", height=1000, width=2000)

#     crud.create_groundtruth(
#         db,
#         data=schemas.GroundTruth(
#             dataset_name=dset_name,
#             detections=[
#                 schemas.GroundTruthDetection(
#                     boundary=boundary1,
#                     image=img,
#                     labels=[schemas.Label(key="k", value="v")],
#                 ),
#                 schemas.GroundTruthDetection(
#                     boundary=boundary2,
#                     image=img,
#                     labels=[schemas.Label(key="k", value="v")],
#                 ),
#             ],
#         ),
#     )

#     areas = db.scalars(ST_Area(models.GroundTruthDetection.boundary)).all()
#     assert sorted(areas) == [150, 1050]

#     # make base statement. need WHERE here because of what `_filter_instance_segmentations_by_area` expects
#     base_stmt = "SELECT id FROM ground_truth_detection WHERE ground_truth_detection.id > 0"

#     # check filtering when use area determined by polygon detection task
#     stmt = _filter_object_detections_by_area(
#         base_stmt,
#         det_table=models.GroundTruthDetection,
#         task_for_area_computation=enums.Task.POLY_OBJECT_DETECTION,
#         min_area=100,
#         max_area=2000,
#     )
#     assert len(db.scalars(stmt).all()) == 2

#     stmt = _filter_object_detections_by_area(
#         base_stmt,
#         det_table=models.GroundTruthDetection,
#         task_for_area_computation=enums.Task.POLY_OBJECT_DETECTION,
#         min_area=100,
#         max_area=200,
#     )
#     assert len(db.scalars(stmt).all()) == 1

#     stmt = _filter_object_detections_by_area(
#         base_stmt,
#         det_table=models.GroundTruthDetection,
#         task_for_area_computation=enums.Task.POLY_OBJECT_DETECTION,
#         min_area=151,
#         max_area=2000,
#     )
#     assert len(db.scalars(stmt).all()) == 1

#     # now when we use bounding box detection task, the triangle becomes its circumscribing
#     # rectangle (with area 300) so we should get both segmentations
#     stmt = _filter_object_detections_by_area(
#         base_stmt,
#         det_table=models.GroundTruthDetection,
#         task_for_area_computation=enums.Task.BBOX_OBJECT_DETECTION,
#         min_area=299,
#         max_area=2000,
#     )
#     assert len(db.scalars(stmt).all()) == 2

#     stmt = _filter_object_detections_by_area(
#         base_stmt,
#         det_table=models.GroundTruthDetection,
#         task_for_area_computation=enums.Task.BBOX_OBJECT_DETECTION,
#         min_area=301,
#         max_area=2000,
#     )
#     assert len(db.scalars(stmt).all()) == 1

#     # check error if use the wrong task type
#     with pytest.raises(ValueError) as exc_info:
#         _filter_object_detections_by_area(
#             base_stmt,
#             det_table=models.GroundTruthDetection,
#             task_for_area_computation=enums.Task.INSTANCE_SEGMENTATION,
#             min_area=301,
#             max_area=2000,
#         )
#     assert "Expected task_for_area_computation to be" in str(exc_info)


# @NOTE: Moved to `velour_api.backend.metrics.detections`
# def test__filter_instance_segmentations_by_area_using_mask(db: Session):
#     crud.create_dataset(db, schemas.Dataset(name=dset_name))

#     # approximate triangle of area 150
#     mask = np.zeros((1000, 2000), dtype=bool)
#     for i in range(10):
#         for j in range(30):
#             if i + j < 20:
#                 mask[i, j] = True
#     assert mask.sum() == 155
#     mask_bytes = pil_to_bytes(Image.fromarray(mask))
#     b64_mask = b64encode(mask_bytes).decode()

#     # rectangle of area 1050
#     boundary = [(0, 5), (0, 40), (30, 40), (30, 5)]

#     img = schemas.Datum(uid="", height=1000, width=2000)

#     crud.create_groundtruth(
#         db,
#         data=list[schemas.GroundTruth](
#             dataset_name=dset_name,
#             segmentations=[
#                 schemas.Annotation(
#                     shape=b64_mask,
#                     image=img,
#                     labels=[schemas.Label(key="k", value="v")],
#                     is_instance=True,
#                 ),
#                 schemas.Annotation(
#                     shape=[schemas.Polygon(polygon=boundary)],
#                     image=img,
#                     labels=[schemas.Label(key="k", value="v")],
#                     is_instance=True,
#                 ),
#             ],
#         ),
#     )

#     areas = db.scalars(ST_Count(models.Annotation.shape)).all()
#     assert sorted(areas) == [155, 1050]

#     base_stmt = "SELECT id FROM ground_truth_segmentation WHERE ground_truth_segmentation.is_instance"

#     # check filtering when use area determined by polygon detection task
#     stmt = _filter_instance_segmentations_by_area(
#         base_stmt,
#         seg_table=models.Annotation,
#         task_for_area_computation=enums.Task.POLY_OBJECT_DETECTION,
#         min_area=100,
#         max_area=2000,
#     )
#     assert len(db.scalars(stmt).all()) == 2

#     stmt = _filter_instance_segmentations_by_area(
#         base_stmt,
#         seg_table=models.Annotation,
#         task_for_area_computation=enums.Task.POLY_OBJECT_DETECTION,
#         min_area=100,
#         max_area=200,
#     )
#     assert len(db.scalars(stmt).all()) == 1

#     stmt = _filter_instance_segmentations_by_area(
#         base_stmt,
#         seg_table=models.Annotation,
#         task_for_area_computation=enums.Task.POLY_OBJECT_DETECTION,
#         min_area=170,  # this won't pass at 156 due to aliasing
#         max_area=2000,
#     )
#     assert len(db.scalars(stmt).all()) == 1

#     # now when we use bounding box detection task, the triangle becomes its circumscribing
#     # rectangle (with area 200) so we should get both segmentations
#     stmt = _filter_instance_segmentations_by_area(
#         base_stmt,
#         seg_table=models.Annotation,
#         task_for_area_computation=enums.Task.BBOX_OBJECT_DETECTION,
#         min_area=160,
#         max_area=2000,
#     )

#     assert len(db.scalars(stmt).all()) == 2

#     stmt = _filter_instance_segmentations_by_area(
#         base_stmt,
#         seg_table=models.Annotation,
#         task_for_area_computation=enums.Task.BBOX_OBJECT_DETECTION,
#         min_area=300,
#         max_area=2000,
#     )
#     assert len(db.scalars(stmt).all()) == 1

#     stmt = _filter_instance_segmentations_by_area(
#         base_stmt,
#         seg_table=models.Annotation,
#         task_for_area_computation=enums.Task.BBOX_OBJECT_DETECTION,
#         min_area=3000,
#         max_area=10000,
#     )
#     assert len(db.scalars(stmt).all()) == 0


# @NOTE: `velour-api.backend` - this will be replaced by stateflow
# @TODO: Implement test for finalizing empty dataset
# def test__validate_and_update_evaluation_settings_task_type_for_detection_no_groundtruth(
#     db: Session,
# ):
#     """Test runtime error when there's no groundtruth data"""
#     crud.create_dataset(db, schemas.Dataset(name=dset_name))

#     crud.create_model(db, schemas.Model(name=model_name))
#     crud.finalize(db, dset_name)
#     crud.finalize(db, model_name=model_name, dataset_name=dset_name)

#     evaluation_settings = schemas.EvaluationSettings(
#         model_name=model_name, dataset_name=dset_name
#     )

#     with pytest.raises(RuntimeError) as exc_info:
#         _validate_and_update_evaluation_settings_task_type_for_detection(
#             db, evaluation_settings
#         )
#     assert "The dataset does not have any annotations to support" in str(
#         exc_info
#     )


# @NOTE: `velour-api.backend` - this will be replaced by stateflow
# @TODO: Implement test for finalizing empty model
# def test__validate_and_update_evaluation_settings_task_type_for_detection_no_predictions(
#     db: Session, gt_dets_create
# ):
#     """Test runtime error when there's no prediction data"""
#     crud.create_dataset(db, schemas.Dataset(name=dset_name))

#     crud.create_model(db, schemas.Model(name=model_name))

#     crud.create_groundtruth(db, gt_dets_create)

#     crud.finalize(db, dset_name)
#     crud.finalize(db, model_name=model_name, dataset_name=dset_name)

#     evaluation_settings = schemas.EvaluationSettings(
#         model_name=model_name, dataset_name=dset_name
#     )

#     with pytest.raises(RuntimeError) as exc_info:
#         _validate_and_update_evaluation_settings_task_type_for_detection(
#             db, evaluation_settings
#         )
#     assert "The model does not have any inferences to support" in str(exc_info)


# @NOTE: `velour-api.backend` - this will be replaced by stateflow
# @TODO: Implement test that checks the set of valid commands over given dataset/model pairing
# def test__validate_and_update_evaluation_settings_task_type_for_detection_multiple_groundtruth_types(
#     db: Session, gt_dets_create, gt_segs_create
# ):
#     crud.create_dataset(db, schemas.Dataset(name=dset_name))

#     crud.create_model(db, schemas.Model(name=model_name))

#     crud.create_groundtruth(db, gt_dets_create)
#     for gt in gt_segs_create:
#         crud.create_groundtruth(db, gt)

#     crud.finalize(db, dset_name)
#     crud.finalize(db, model_name=model_name, dataset_name=dset_name)

#     evaluation_settings = schemas.EvaluationSettings(
#         model_name=model_name, dataset_name=dset_name
#     )

#     with pytest.raises(RuntimeError) as exc_info:
#         _validate_and_update_evaluation_settings_task_type_for_detection(
#             db, evaluation_settings
#         )
#     assert "The dataset has the following tasks compatible" in str(exc_info)

#     # now specify task types for dataset and check we get an error since model
#     # has no inferences
#     evaluation_settings = schemas.EvaluationSettings(
#         model_name=model_name,
#         dataset_name=dset_name,
#         dataset_gt_task_type=enums.Task.BBOX_OBJECT_DETECTION,
#     )
#     with pytest.raises(RuntimeError) as exc_info:
#         _validate_and_update_evaluation_settings_task_type_for_detection(
#             db, evaluation_settings
#         )
#     assert "The model does not have any inferences to support" in str(exc_info)


# @NOTE: `velour-api.backend` - this will be replaced by stateflow
# @TODO: Implement test that checks the set of valid commands over given dataset/model pairing
# def test__validate_and_update_evaluation_settings_task_type_for_detection_multiple_prediction_types(
#     db: Session, gt_dets_create, pred_dets_create, pred_segs_create
# ):
#     crud.create_dataset(db, schemas.Dataset(name=dset_name))

#     crud.create_model(db, schemas.Model(name=model_name))

#     crud.create_groundtruth(db, gt_dets_create)
#     crud.create_prediction(db, pred_dets_create)
#     for pd in pred_segs_create:
#         crud.create_prediction(db, pd)

#     crud.finalize(db, dset_name)
#     crud.finalize(db, model_name=model_name, dataset_name=dset_name)

#     evaluation_settings = schemas.EvaluationSettings(
#         model_name=model_name, dataset_name=dset_name
#     )

#     with pytest.raises(RuntimeError) as exc_info:
#         _validate_and_update_evaluation_settings_task_type_for_detection(
#             db, evaluation_settings
#         )
#     assert "The model has the following tasks compatible" in str(exc_info)

#     # now specify task type for model and check there's no error and that
#     # the dataset task type was made explicit
#     evaluation_settings = schemas.EvaluationSettings(
#         model_name=model_name,
#         dataset_name=dset_name,
#         model_pred_task_type=enums.Task.BBOX_OBJECT_DETECTION,
#     )
#     assert evaluation_settings.dataset_gt_task_type is None
#     _validate_and_update_evaluation_settings_task_type_for_detection(
#         db, evaluation_settings
#     )
#     assert (
#         evaluation_settings.dataset_gt_task_type
#         == enums.Task.POLY_OBJECT_DETECTION
#     )


# @NOTE: `velour_api.backend.io`
# @TODO: Implement a test that checks the existince and linking of metatata
# def test_create_datums_with_metadata(db: Session):
#     crud.create_dataset(
#         db,
#         schemas.Dataset(name=dset_name),
#     )

#     datums = [
#         schemas.Datum(
#             uid="uid1",
#             metadata=[
#                 schemas.MetaDatum(name="type", value=enums.DataType.TABULAR.value),
#                 schemas.MetaDatum(name="name1", value=0.7),
#                 schemas.MetaDatum(name="name2", value="a string"),
#             ],
#         ),
#         schemas.Datum(
#             uid="uid2",
#             metadata=[
#                 schemas.MetaDatum(name="type", value=enums.DataType.TABULAR.value),
#                 schemas.MetaDatum(name="name2", value="a string"),
#                 schemas.MetaDatum(
#                     name="name3",
#                     value={
#                         "type": "Point",
#                         "coordinates": [-48.23456, 20.12345],
#                     },
#                 ),
#             ],
#         ),
#     ]
#     crud._create._add_datums_to_dataset(db, dset_name, datums)

#     # check there should only be three unique metadatums since two are the same
#     assert len(db.scalars(select(models.Metadatum)).all()) == 3

#     db_datums = crud.get_datums_in_dataset(db, dset_name)

#     assert len(db_datums) == 2

#     md1 = db_datums[0].datum_metadatum_links[0].metadatum
#     assert md1.name == "name1"
#     assert md1.numeric_value == 0.7
#     assert md1.string_value is None
#     assert md1.geo is None

#     md2 = db_datums[1].datum_metadatum_links[0].metadatum
#     assert md2.name == "name2"
#     assert md2.numeric_value is None
#     assert md2.string_value == "a string"
#     assert md2.geo is None

#     md3 = db_datums[1].datum_metadatum_links[1].metadatum
#     assert md3.name == "name3"
#     assert md3.numeric_value is None
#     assert md3.string_value is None
#     assert json.loads(db.scalar(ST_AsGeoJSON(md3.geo))) == {
#         "type": "Point",
#         "coordinates": [-48.23456, 20.12345],
#     }


""" READ """


def test_get_dataset(db: Session):
    with pytest.raises(exceptions.DatasetDoesNotExistError) as exc_info:
        crud.get_dataset(db, dset_name)
    assert "does not exist" in str(exc_info)

    crud.create_dataset(db, schemas.Dataset(name=dset_name))

    dset = crud.get_dataset(db, dset_name)
    assert dset.name == dset_name


def test_get_model(db: Session):
    with pytest.raises(exceptions.ModelDoesNotExistError) as exc_info:
        crud.get_model(db, model_name)
    assert "does not exist" in str(exc_info)

    crud.create_model(db, schemas.Model(name=model_name))
    model = crud.get_model(db, model_name)
    assert model.name == model_name


# @NOTE: Sub-functionality of `crud.get_info` 
# @TODO: Implement `crud.get_info`
# def test_get_dataset_info(
#     db: Session,
#     dataset_names: list[str],
#     model_names: list[str],
#     dataset_model_associations_create,
# ):
#     ds_meta1 = get_dataset_info(db, dataset_names[0])
#     assert ds_meta1.annotation_type == ["DETECTION"]
#     assert ds_meta1.number_of_classifications == 0
#     assert ds_meta1.number_of_bounding_boxes == 0
#     assert ds_meta1.number_of_bounding_polygons == 2
#     assert ds_meta1.number_of_segmentation_rasters == 0
#     assert ds_meta1.associated == [model_names[0]]

#     ds_meta2 = get_dataset_info(db, dataset_names[1])
#     assert ds_meta2.annotation_type == ["DETECTION"]
#     assert ds_meta2.number_of_classifications == 0
#     assert ds_meta2.number_of_bounding_boxes == 0
#     assert ds_meta2.number_of_bounding_polygons == 2
#     assert ds_meta2.number_of_segmentation_rasters == 0
#     assert ds_meta2.associated == [model_names[0], model_names[1]]

# @NOTE: Sub-functionality of `crud.get_info` 
# @TODO: Implement `crud.get_info`
# def test_get_model_info(
#     db: Session,
#     dataset_names: list[str],
#     model_names: list[str],
#     dataset_model_associations_create,
# ):
#     md_meta1 = get_model_info(db, model_names[0])
#     assert md_meta1.annotation_type == ["DETECTION"]
#     assert md_meta1.number_of_classifications == 0
#     assert md_meta1.number_of_bounding_boxes == 0
#     assert md_meta1.number_of_bounding_polygons == 4
#     assert md_meta1.number_of_segmentation_rasters == 0
#     assert md_meta1.associated == [dataset_names[0], dataset_names[1]]

#     md_meta2 = get_model_info(db, model_names[1])
#     assert md_meta2.annotation_type == ["DETECTION"]
#     assert md_meta2.number_of_classifications == 0
#     assert md_meta2.number_of_bounding_boxes == 0
#     assert md_meta2.number_of_bounding_polygons == 2
#     assert md_meta2.number_of_segmentation_rasters == 0
#     assert md_meta2.associated == [dataset_names[1]]


# @NOTE: Sub-functionality of `crud.get_info` 
# @TODO: Implement `crud.get_info`
# def test__get_associated_models(
#     db: Session,
#     dataset_names: list[str],
#     model_names: list[str],
#     dataset_model_associations_create,
# ):
#     # Get associated models for each dataset
#     assert _get_associated_models(db, dataset_names[0]) == [model_names[0]]
#     assert _get_associated_models(db, dataset_names[1]) == [
#         model_names[0],
#         model_names[1],
#     ]
#     assert _get_associated_models(db, "doesnt exist") == []

# @NOTE: Sub-functionality of `crud.get_info` 
# @TODO: Implement `crud.get_info`
# def test__get_associated_datasets(
#     db: Session,
#     dataset_names: list[str],
#     model_names: list[str],
#     dataset_model_associations_create,
# ):
#     # Get associated datasets for each model
#     assert _get_associated_datasets(db, model_names[0]) == [
#         dataset_names[0],
#         dataset_names[1],
#     ]
#     assert _get_associated_datasets(db, model_names[1]) == [dataset_names[1]]
#     assert _get_associated_datasets(db, "doesnt exist") == []

# @NOTE: Sub-functionality of `crud.get_info` 
# @TODO: Implement `crud.get_info`
# def test_get_label_distribution_from_dataset(
#     db: Session,
#     dataset_names: list[str],
#     dataset_model_associations_create,
# ):
#     ds1 = get_label_distribution_from_dataset(db, dataset_names[0])
#     assert len(ds1) == 2
#     assert ds1[0] == schemas.LabelDistribution(
#         label=schemas.Label(key="k1", value="v1"), count=1
#     )
#     assert ds1[1] == schemas.LabelDistribution(
#         label=schemas.Label(key="k2", value="v2"), count=2
#     )

#     ds2 = get_label_distribution_from_dataset(db, dataset_names[1])
#     assert len(ds2) == 2
#     assert ds1[0] == schemas.LabelDistribution(
#         label=schemas.Label(key="k1", value="v1"), count=1
#     )
#     assert ds1[1] == schemas.LabelDistribution(
#         label=schemas.Label(key="k2", value="v2"), count=2
#     )

# @NOTE: Sub-functionality of `crud.get_info` 
# @TODO: Implement `crud.get_info`
# def test_get_label_distribution_from_model(
#     db: Session,
#     model_names: list[str],
#     dataset_model_associations_create,
# ):

#     md1 = get_label_distribution_from_model(db, model_names[0])
#     assert len(md1) == 2
#     assert md1[0] == schemas.ScoredLabelDistribution(
#         label=schemas.Label(key="k1", value="v1"), scores=[0.6, 0.6], count=2
#     )
#     assert md1[1] == schemas.ScoredLabelDistribution(
#         label=schemas.Label(key="k2", value="v2"),
#         scores=[0.2, 0.9, 0.2, 0.9],
#         count=4,
#     )

#     md2 = get_label_distribution_from_model(db, model_names[1])
#     assert len(md2) == 2
#     assert md2[0] == schemas.ScoredLabelDistribution(
#         label=schemas.Label(key="k1", value="v1"), scores=[0.6], count=1
#     )
#     assert md2[1] == schemas.ScoredLabelDistribution(
#         label=schemas.Label(key="k2", value="v2"), scores=[0.2, 0.9], count=2
#     )

def test_get_all_labels(
    db: Session, gt_dets_create: schemas.GroundTruth
):
    crud.create_dataset(db, schemas.Dataset(name=dset_name))

    for gt in gt_dets_create:
        crud.create_groundtruth(db, gt)

    labels = crud.get_labels(db, request=schemas.Filter())

    assert len(labels) == 2
    assert set([(label.key, label.value) for label in labels]) == set(
        [("k1", "v1"), ("k2", "v2")]
    )


def test_get_labels_from_dataset(
    db: Session,
    dataset_names: list[str],
    dataset_model_associations_create,
):

    # Test get all from dataset 1
    ds1 = crud.get_labels(
        db, 
        schemas.Filter(
            filter_by_dataset_names=[dataset_names[0]]
        )
    )
    assert len(ds1) == 2
    assert schemas.Label(key="k1", value="v1") in ds1
    assert schemas.Label(key="k2", value="v2") in ds1

    # Test get all from dataset 2
    ds2 = crud.get_labels(
        db, 
        schemas.Filter(
            filter_by_dataset_names=[dataset_names[1]]
        )
    )
    assert len(ds2) == 2
    assert schemas.Label(key="k1", value="v1") in ds2
    assert schemas.Label(key="k2", value="v2") in ds2

    # Test get all but polygon labels from dataset 1
    ds1 = crud.get_labels(
        db, 
        schemas.Filter(
            filter_by_dataset_names=[dataset_names[0]],
            filter_by_task_types=[enums.AnnotationType.CLASSIFICATION],
            filter_by_annotation_types=[
                enums.AnnotationType.BOX,
                enums.AnnotationType.RASTER,
            ]
        )
    )
    assert ds1 == []

    # Test get only polygon labels from dataset 1
    ds1 = crud.get_labels(
        db,
        dataset_name=dataset_names[0],
        of_type=[enums.AnnotationType.POLYGON],
    )
    assert len(ds1) == 2
    assert schemas.Label(key="k1", value="v1") in ds1
    assert schemas.Label(key="k2", value="v2") in ds1


def test_get_labels_from_model(
    db: Session,
    model_names: list[str],
    dataset_model_associations_create,
):
    # Test get all labels from model 1
    md1 = crud.get_labels(db, model_name=model_names[0])
    assert len(md1) == 2
    assert schemas.Label(key="k1", value="v1") in md1
    assert schemas.Label(key="k2", value="v2") in md1

    # Test get all labels from model 2
    md2 = crud.get_labels(db, model_name=model_names[1])
    assert len(md2) == 2
    assert schemas.Label(key="k1", value="v1") in md2
    assert schemas.Label(key="k2", value="v2") in md2

    # Test get all but polygon labels from model 1
    md1 = crud.get_labels(
        db,
        model_name=model_names[0],
        metadatum_id=None,
        of_type=[
            enums.AnnotationType.CLASSIFICATION,
            enums.AnnotationType.BBOX,
            enums.AnnotationType.RASTER,
        ],
    )
    assert md1 == []

    # Test get only polygon labels from model 1
    md1 = crud.get_labels(
        db,
        model_name=model_names[0],
        metadatum_id=None,
        of_type=[enums.AnnotationType.POLYGON],
    )
    assert len(md1) == 2
    assert schemas.Label(key="k1", value="v1") in md1
    assert schemas.Label(key="k2", value="v2") in md1


def test_get_joint_labels(
    db: Session,
    dataset_names: list[str],
    model_names: list[str],
    dataset_model_associations_create,
):
    # Test get joint labels from dataset 1 and model 1
    assert set(
        crud.get_labels(
            db=db,
            dataset_name=dataset_names[0],
            model_name=model_names[0],
        )
    ) == set(
        [
            schemas.Label(key="k1", value="v1"),
            schemas.Label(key="k2", value="v2"),
        ]
    )

    # Test get joint labels from dataset 1 and model 1 where labels are not for polygons
    assert (
        set(
            crud.get_labels(
                db=db,
                dataset_name=dataset_names[0],
                model_name=model_names[0],
                of_type=[
                    enums.AnnotationType.CLASSIFICATION,
                    enums.AnnotationType.BBOX,
                    enums.AnnotationType.RASTER,
                ],
            )
        )
        == set()
    )

    # Test get joint labels from dataset 1 and model 1 where labels are only for polygons
    assert set(
        crud.get_labels(
            db=db,
            dataset_name=dataset_names[0],
            model_name=model_names[0],
            of_type=[enums.AnnotationType.POLYGON],
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
#     crud.create_dataset(
#         db,
#         schemas.Dataset(name=dset_name),
#     )

#     datums = [
#         schemas.Datum(
#             uid="uid1",
#             metadata=[
#                 schemas.MetaDatum(name="type", value=enums.DataType.TABULAR.value),
#                 schemas.MetaDatum(name="md1", value=0.7)
#             ],
#         ),
#         schemas.Datum(
#             uid="uid2",
#             metadata=[
#                 schemas.MetaDatum(name="type", value=enums.DataType.TABULAR.value),
#                 schemas.MetaDatum(name="md1", value="md1-val1"),
#                 schemas.MetaDatum(name="md2", value="md2-val1"),
#             ],
#         ),
#         schemas.Datum(
#             uid="uid3",
#             metadata=[
#                 schemas.MetaDatum(name="type", value=enums.DataType.TABULAR.value),
#                 schemas.MetaDatum(name="md1", value="md1-val1"),
#             ],
#         ),
#         schemas.Datum(
#             uid="uid4",
#             metadata=[
#                 schemas.MetaDatum(name="type", value=enums.DataType.TABULAR.value),
#                 schemas.MetaDatum(name="md1", value="md1-val2"),
#             ],
#         ),
#     ]

#     crud._create._add_datums_to_dataset(db, dset_name, datums)

#     string_ids = crud.get_string_metadata_ids(
#         db, dset_name, metadata_name="md1"
#     )

#     assert len(string_ids) == 2
#     # assert set([s[1] for s in string_vals_and_ids]) == {"md1-val1", "md1-val2"}

#     string_ids = crud.get_string_metadata_ids(
#         db, dset_name, metadata_name="md2"
#     )
#     assert len(string_ids) == 1
#     # assert [s[1] for s in string_vals_and_ids] == ["md2-val1"]

#     string_ids = crud.get_string_metadata_ids(
#         db, dset_name, metadata_name="md3"
#     )
#     assert string_ids == []
