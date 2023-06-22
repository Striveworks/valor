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

from velour_api import crud, enums, exceptions, models, ops, schemas
from velour_api.crud._create import (
    _instance_segmentations_in_dataset_statement,
    _model_instance_segmentation_preds_statement,
    _model_object_detection_preds_statement,
    _object_detections_in_dataset_statement,
    _validate_and_update_evaluation_settings_task_type_for_detection,
)
from velour_api.crud._read import (
    _filter_instance_segmentations_by_area,
    _filter_object_detections_by_area,
    _raster_to_png_b64,
)

dset_name = "test dataset"
model_name = "test model"


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
        models.Datum,
        models.GroundTruthDetection,
        models.LabeledGroundTruthDetection,
        models.PredictedDetection,
        models.Label,
        models.Dataset,
        models.GroundTruthClassification,
        models.PredictedClassification,
        models.LabeledGroundTruthSegmentation,
        models.LabeledPredictedDetection,
        models.PredictedSegmentation,
        models.LabeledPredictedSegmentation,
    ]:
        assert db.scalar(select(func.count(model_cls.id))) == 0


@pytest.fixture
def poly_without_hole() -> schemas.PolygonWithHole:
    # should have area 45.5
    return schemas.PolygonWithHole(
        polygon=[(14, 10), (19, 7), (21, 2), (12, 2)]
    )


@pytest.fixture
def poly_with_hole() -> schemas.PolygonWithHole:
    # should have area 100 - 8 = 92
    return schemas.PolygonWithHole(
        polygon=[(0, 10), (10, 10), (10, 0), (0, 0)],
        hole=[(2, 4), (2, 8), (6, 4)],
    )


@pytest.fixture
def gt_dets_create(img1: schemas.Image) -> schemas.GroundTruthDetectionsCreate:
    return schemas.GroundTruthDetectionsCreate(
        dataset_name=dset_name,
        detections=[
            schemas.GroundTruthDetection(
                boundary=[(10, 20), (10, 30), (20, 30), (20, 20), (10, 20)],
                labels=[
                    schemas.Label(key="k1", value="v1"),
                    schemas.Label(key="k2", value="v2"),
                ],
                image=img1,
            ),
            schemas.GroundTruthDetection(
                boundary=[(10, 20), (10, 30), (20, 30), (20, 20), (10, 20)],
                labels=[schemas.Label(key="k2", value="v2")],
                image=img1,
            ),
        ],
    )


@pytest.fixture
def pred_dets_create(img1: schemas.Image) -> schemas.PredictedDetectionsCreate:
    return schemas.PredictedDetectionsCreate(
        model_name=model_name,
        dataset_name=dset_name,
        detections=[
            schemas.PredictedDetection(
                boundary=[(107, 207), (107, 307), (207, 307), (207, 207)],
                scored_labels=[
                    schemas.ScoredLabel(
                        label=schemas.Label(key="k1", value="v1"), score=0.6
                    ),
                    schemas.ScoredLabel(
                        label=schemas.Label(key="k2", value="v2"), score=0.2
                    ),
                ],
                image=img1,
            ),
            schemas.PredictedDetection(
                boundary=[(107, 207), (107, 307), (207, 307), (207, 207)],
                scored_labels=[
                    schemas.ScoredLabel(
                        label=schemas.Label(key="k2", value="v2"), score=0.9
                    )
                ],
                image=img1,
            ),
        ],
    )


@pytest.fixture
def gt_segs_create(
    poly_with_hole, poly_without_hole, img1, img2
) -> schemas.GroundTruthSegmentationsCreate:
    return schemas.GroundTruthSegmentationsCreate(
        dataset_name=dset_name,
        segmentations=[
            schemas.GroundTruthSegmentation(
                is_instance=True,
                shape=[poly_with_hole],
                image=img1,
                labels=[schemas.Label(key="k1", value="v1")],
            ),
            schemas.GroundTruthSegmentation(
                is_instance=False,
                shape=[poly_without_hole],
                image=img2,
                labels=[schemas.Label(key="k1", value="v1")],
            ),
            schemas.GroundTruthSegmentation(
                is_instance=True,
                shape=[poly_without_hole],
                image=img2,
                labels=[schemas.Label(key="k3", value="v3")],
            ),
            schemas.GroundTruthSegmentation(
                is_instance=True,
                shape=[poly_with_hole, poly_without_hole],
                image=img1,
                labels=[schemas.Label(key="k1", value="v1")],
            ),
        ],
    )


@pytest.fixture
def pred_segs_create(
    mask_bytes1: bytes,
    mask_bytes2: bytes,
    mask_bytes3: bytes,
    img1: schemas.Image,
) -> schemas.PredictedSegmentationsCreate:
    b64_mask1 = b64encode(mask_bytes1).decode()
    b64_mask2 = b64encode(mask_bytes2).decode()
    b64_mask3 = b64encode(mask_bytes3).decode()
    return schemas.PredictedSegmentationsCreate(
        model_name=model_name,
        dataset_name=dset_name,
        segmentations=[
            schemas.PredictedSegmentation(
                base64_mask=b64_mask1,
                is_instance=True,
                image=img1,
                scored_labels=[
                    schemas.ScoredLabel(
                        label=schemas.Label(key="k1", value="v1"), score=0.43
                    )
                ],
            ),
            schemas.PredictedSegmentation(
                base64_mask=b64_mask2,
                is_instance=False,
                image=img1,
                scored_labels=[
                    schemas.ScoredLabel(
                        label=schemas.Label(key="k2", value="v2"), score=0.97
                    )
                ],
            ),
            schemas.PredictedSegmentation(
                base64_mask=b64_mask2,
                is_instance=True,
                image=img1,
                scored_labels=[
                    schemas.ScoredLabel(
                        label=schemas.Label(key="k2", value="v2"), score=0.74
                    )
                ],
            ),
            schemas.PredictedSegmentation(
                base64_mask=b64_mask3,
                is_instance=True,
                image=img1,
                scored_labels=[
                    schemas.ScoredLabel(
                        label=schemas.Label(key="k2", value="v2"), score=0.14
                    )
                ],
            ),
        ],
    )


@pytest.fixture
def gt_clfs_create(
    img1: schemas.Image, img2: schemas.Image
) -> schemas.GroundTruthClassificationsCreate:
    return schemas.GroundTruthClassificationsCreate(
        dataset_name=dset_name,
        classifications=[
            schemas.GroundTruthClassification(
                datum=img1,
                labels=[
                    schemas.Label(key="k1", value="v1"),
                    schemas.Label(key="k2", value="v2"),
                ],
            ),
            schemas.GroundTruthClassification(
                datum=img2,
                labels=[schemas.Label(key="k2", value="v3")],
            ),
        ],
    )


@pytest.fixture
def pred_clfs_create(
    img1: schemas.Image, img2: schemas.Image
) -> schemas.PredictedClassificationsCreate:
    return schemas.PredictedClassificationsCreate(
        model_name=model_name,
        dataset_name=dset_name,
        classifications=[
            schemas.PredictedClassification(
                datum=img1,
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
            ),
            schemas.PredictedClassification(
                datum=img2,
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
            ),
        ],
    )


def test_create_and_get_datasets(db: Session):
    crud.create_dataset(
        db,
        schemas.Dataset(name=dset_name, type=schemas.DatumTypes.IMAGE),
    )

    all_datasets = db.scalars(select(models.Dataset)).all()
    assert len(all_datasets) == 1
    assert all_datasets[0].name == dset_name

    with pytest.raises(exceptions.DatasetAlreadyExistsError) as exc_info:
        crud.create_dataset(
            db,
            schemas.Dataset(name=dset_name, type=schemas.DatumTypes.IMAGE),
        )
    assert "already exists" in str(exc_info)

    crud.create_dataset(
        db,
        schemas.Dataset(name="other dataset", type=schemas.DatumTypes.IMAGE),
    )
    datasets = crud.get_datasets(db)
    assert len(datasets) == 2
    assert set([d.name for d in datasets]) == {dset_name, "other dataset"}

    # clean up
    crud.delete_dataset(db, dset_name)
    crud.delete_dataset(db, "other dataset")


def test_create_and_get_models(db: Session):
    crud.create_model(
        db, schemas.Model(name=model_name, type=schemas.DatumTypes.IMAGE)
    )

    all_models = db.scalars(select(models.Model)).all()
    assert len(all_models) == 1
    assert all_models[0].name == model_name

    with pytest.raises(exceptions.ModelAlreadyExistsError) as exc_info:
        crud.create_model(
            db, schemas.Model(name=model_name, type=schemas.DatumTypes.IMAGE)
        )
    assert "already exists" in str(exc_info)

    crud.create_model(
        db, schemas.Model(name="other model", type=schemas.DatumTypes.IMAGE)
    )
    db_models = crud.get_models(db)
    assert len(db_models) == 2
    assert set([m.name for m in db_models]) == {model_name, "other model"}


def test_get_dataset(db: Session):
    with pytest.raises(exceptions.DatasetDoesNotExistError) as exc_info:
        crud.get_dataset(db, dset_name)
    assert "does not exist" in str(exc_info)

    crud.create_dataset(
        db,
        schemas.Dataset(name=dset_name, type=schemas.DatumTypes.IMAGE),
    )
    dset = crud.get_dataset(db, dset_name)
    assert dset.name == dset_name


def test_get_model(db: Session):
    with pytest.raises(exceptions.ModelDoesNotExistError) as exc_info:
        crud.get_model(db, model_name)
    assert "does not exist" in str(exc_info)

    crud.create_model(
        db, schemas.Model(name=model_name, type=schemas.DatumTypes.IMAGE)
    )
    model = crud.get_model(db, model_name)
    assert model.name == model_name


def test_create_ground_truth_detections_and_delete_dataset(
    db: Session, gt_dets_create: schemas.GroundTruthDetectionsCreate
):
    # sanity check nothing in db
    check_db_empty(db=db)

    crud.create_dataset(
        db,
        schemas.Dataset(name=dset_name, type=schemas.DatumTypes.IMAGE),
    )

    crud.create_groundtruth_detections(db, data=gt_dets_create)

    assert crud.number_of_rows(db, models.GroundTruthDetection) == 2
    assert crud.number_of_rows(db, models.Datum) == 1
    assert crud.number_of_rows(db, models.LabeledGroundTruthDetection) == 3
    assert crud.number_of_rows(db, models.Label) == 2

    # verify we get the same dets back
    dets = crud.get_groundtruth_detections_in_image(
        db, uid=gt_dets_create.detections[0].image.uid, dataset_name=dset_name
    )
    assert dets == gt_dets_create.detections

    # delete dataset and check the cascade worked
    crud.delete_dataset(db, dataset_name=dset_name)
    for model_cls in [
        models.Dataset,
        models.Datum,
        models.GroundTruthDetection,
        models.LabeledGroundTruthDetection,
    ]:
        assert crud.number_of_rows(db, model_cls) == 0

    # make sure labels are still there`
    assert crud.number_of_rows(db, models.Label) == 2


def test_create_predicted_detections_and_delete_model(
    db: Session,
    pred_dets_create: schemas.PredictedDetectionsCreate,
    gt_dets_create: schemas.GroundTruthDetectionsCreate,
):
    # check this gives an error since the dataset hasn't been defined yet
    with pytest.raises(exceptions.ModelDoesNotExistError) as exc_info:
        crud.create_predicted_detections(db, pred_dets_create)
    assert "does not exist" in str(exc_info)

    crud.create_model(
        db, schemas.Model(name=model_name, type=schemas.DatumTypes.IMAGE)
    )

    # check this gives an error since the images haven't been added yet
    with pytest.raises(exceptions.ImageDoesNotExistError) as exc_info:
        crud.create_predicted_detections(db, pred_dets_create)
    assert (
        "Image with uid 'uid1' does not exist in dataset 'test dataset'."
        in str(exc_info)
    )

    # create dataset, add images, and add predictions
    crud.create_dataset(
        db,
        schemas.Dataset(name=dset_name, type=schemas.DatumTypes.IMAGE),
    )

    # finalize early
    crud.finalize_dataset(db, dset_name)

    # check this gives an error since the images haven't been added yet
    with pytest.raises(exceptions.ImageDoesNotExistError) as exc_info:
        crud.create_predicted_detections(db, pred_dets_create)
    assert (
        "Image with uid 'uid1' does not exist in dataset 'test dataset'."
        in str(exc_info)
    )

    # check model has no entries
    assert crud.number_of_rows(db, models.PredictedDetection) == 0
    assert crud.number_of_rows(db, models.LabeledPredictedDetection) == 0

    # delete dataset
    crud.delete_dataset(db, dset_name)

    # create dataset and add a groundtruth
    crud.create_dataset(
        db,
        schemas.Dataset(name=dset_name, type=schemas.DatumTypes.IMAGE),
    )
    crud.create_groundtruth_detections(db, gt_dets_create)

    # finalize dataset
    crud.finalize_dataset(db, dset_name)

    # add prediction
    crud.create_predicted_detections(db, pred_dets_create)

    # check db has the added predictions
    assert crud.number_of_rows(db, models.PredictedDetection) == 2
    assert crud.number_of_rows(db, models.LabeledPredictedDetection) == 3

    # delete model and check all detections from it are gone
    crud.delete_model(db, model_name)
    assert crud.number_of_rows(db, models.Model) == 0
    assert crud.number_of_rows(db, models.PredictedDetection) == 0
    assert crud.number_of_rows(db, models.LabeledPredictedDetection) == 0


def test_create_detections_as_bbox_or_poly(db: Session, img1: schemas.Image):
    xmin, ymin, xmax, ymax = 50, 70, 120, 300
    det1 = schemas.GroundTruthDetection(
        boundary=[(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)],
        image=img1,
        labels=[schemas.Label(key="k", value="v")],
    )
    det2 = schemas.GroundTruthDetection(
        bbox=(xmin, ymin, xmax, ymax),
        image=img1,
        labels=[schemas.Label(key="k", value="v")],
    )

    crud.create_dataset(
        db,
        schemas.Dataset(name=dset_name, type=schemas.DatumTypes.IMAGE),
    )
    crud.create_groundtruth_detections(
        db,
        data=schemas.GroundTruthDetectionsCreate(
            dataset_name=dset_name, detections=[det1, det2]
        ),
    )

    dets = db.scalars(select(models.LabeledGroundTruthDetection)).all()
    assert len(dets) == 2
    assert set([det.detection.is_bbox for det in dets]) == {True, False}

    # check we get the same polygon
    assert db.scalar(ST_AsText(dets[0].detection.boundary)) == db.scalar(
        ST_AsText(dets[1].detection.boundary)
    )


def test_create_ground_truth_classifications_and_delete_dataset(
    db: Session, gt_clfs_create: schemas.GroundTruthClassificationsCreate
):
    crud.create_dataset(
        db,
        schemas.Dataset(name=dset_name, type=schemas.DatumTypes.IMAGE),
    )
    crud.create_ground_truth_classifications(db, gt_clfs_create)

    # should have three GroundTruthClassification rows since one image has two
    # labels and the other has one
    assert crud.number_of_rows(db, models.GroundTruthClassification) == 3
    assert crud.number_of_rows(db, models.Datum) == 2
    assert crud.number_of_rows(db, models.Label) == 3

    # delete dataset and check the cascade worked
    crud.delete_dataset(db, dataset_name=dset_name)
    for model_cls in [
        models.Dataset,
        models.Datum,
        models.GroundTruthClassification,
    ]:
        assert crud.number_of_rows(db, model_cls) == 0

    # make sure labels are still there`
    assert crud.number_of_rows(db, models.Label) == 3


def test_create_predicted_classifications_and_delete_model(
    db: Session,
    pred_clfs_create: schemas.PredictedClassification,
    gt_clfs_create: schemas.GroundTruthClassificationsCreate,
):
    # check this gives an error since no dataset exists yet
    with pytest.raises(exceptions.ModelDoesNotExistError) as exc_info:
        crud.create_predicted_image_classifications(db, pred_clfs_create)
    assert "does not exist" in str(exc_info)

    crud.create_model(
        db, schemas.Model(name=model_name, type=schemas.DatumTypes.IMAGE)
    )

    # create dataset without images or predictions
    crud.create_dataset(
        db,
        schemas.Dataset(name=dset_name, type=schemas.DatumTypes.IMAGE),
    )

    # finalize early
    crud.finalize_dataset(db, dset_name)

    # check this gives an error since the images haven't been added yet
    with pytest.raises(exceptions.ImageDoesNotExistError) as exc_info:
        crud.create_predicted_image_classifications(db, pred_clfs_create)
    assert "Image with uid" in str(exc_info)

    # reset dataset
    crud.delete_dataset(db, dset_name)

    # create dataset, add images, and add predictions
    crud.create_dataset(
        db,
        schemas.Dataset(name=dset_name, type=schemas.DatumTypes.IMAGE),
    )
    crud.create_ground_truth_classifications(db, gt_clfs_create)
    crud.finalize_dataset(db, dset_name)

    # create inference
    crud.create_predicted_image_classifications(db, pred_clfs_create)

    # check db has the added predictions
    assert crud.number_of_rows(db, models.PredictedClassification) == 6

    # delete model and check all detections from it are gone
    crud.delete_model(db, model_name)
    assert crud.number_of_rows(db, models.Model) == 0
    assert crud.number_of_rows(db, models.PredictedDetection) == 0
    assert crud.number_of_rows(db, models.LabeledPredictedDetection) == 0


def test_create_ground_truth_segmentations_and_delete_dataset(
    db: Session, gt_segs_create: schemas.GroundTruthSegmentationsCreate
):
    # sanity check nothing in db
    check_db_empty(db=db)

    crud.create_dataset(
        db,
        schemas.Dataset(name=dset_name, type=schemas.DatumTypes.IMAGE),
    )

    crud.create_groundtruth_segmentations(db, data=gt_segs_create)

    assert crud.number_of_rows(db, models.GroundTruthSegmentation) == 4
    assert crud.number_of_rows(db, models.Datum) == 2
    assert crud.number_of_rows(db, models.LabeledGroundTruthSegmentation) == 4
    assert crud.number_of_rows(db, models.Label) == 2

    # delete dataset and check the cascade worked
    crud.delete_dataset(db, dataset_name=dset_name)
    for model_cls in [
        models.Dataset,
        models.Datum,
        models.GroundTruthSegmentation,
        models.LabeledGroundTruthSegmentation,
    ]:
        assert crud.number_of_rows(db, model_cls) == 0

    # make sure labels are still there`
    assert crud.number_of_rows(db, models.Label) == 2


def test_create_predicted_segmentations_check_area_and_delete_model(
    db: Session,
    pred_segs_create: schemas.PredictedSegmentationsCreate,
    gt_segs_create: schemas.GroundTruthSegmentationsCreate,
):
    # check this gives an error since the model hasn't been added yet
    with pytest.raises(exceptions.ModelDoesNotExistError) as exc_info:
        crud.create_predicted_segmentations(db, pred_segs_create)
    assert "does not exist" in str(exc_info)

    crud.create_model(
        db, schemas.Model(name=model_name, type=schemas.DatumTypes.IMAGE)
    )

    # check this gives an error since the images haven't been added yet
    with pytest.raises(exceptions.ImageDoesNotExistError) as exc_info:
        crud.create_predicted_segmentations(db, pred_segs_create)
    assert "Image with uid" in str(exc_info)

    # create dataset, add images, and add predictions
    crud.create_dataset(
        db,
        schemas.Dataset(name=dset_name, type=schemas.DatumTypes.IMAGE),
    )
    crud.create_groundtruth_segmentations(db, gt_segs_create)
    crud.finalize_dataset(db, dset_name)
    crud.create_predicted_segmentations(db, pred_segs_create)

    # check db has the added predictions
    assert crud.number_of_rows(db, models.PredictedSegmentation) == 4
    assert crud.number_of_rows(db, models.LabeledPredictedSegmentation) == 4

    # grab the first one and check that the area of the raster
    # matches the area of the image
    img = crud.get_image(db, "uid1", dset_name)
    seg = img.predicted_segmentations[0]
    mask = bytes_to_pil(
        b64decode(pred_segs_create.segmentations[0].base64_mask)
    )
    assert ops._raster_area(db, seg.shape) == np.array(mask).sum()

    # delete model and check all detections from it are gone
    crud.delete_model(db, model_name)
    assert crud.number_of_rows(db, models.Model) == 0
    assert crud.number_of_rows(db, models.PredictedSegmentation) == 0
    assert crud.number_of_rows(db, models.LabeledPredictedSegmentation) == 0


def test_get_labels(
    db: Session, gt_dets_create: schemas.GroundTruthDetectionsCreate
):
    crud.create_dataset(
        db,
        schemas.Dataset(name=dset_name, type=schemas.DatumTypes.IMAGE),
    )
    crud.create_groundtruth_detections(db, data=gt_dets_create)
    labels = crud.get_detection_labels_in_dataset(db, dset_name)

    assert len(labels) == 2
    assert set([(label.key, label.value) for label in labels]) == set(
        [("k1", "v1"), ("k2", "v2")]
    )

    assert crud.get_detection_labels_in_dataset(db, "not a dataset") == []
    assert crud.get_classification_labels_in_dataset(db, dset_name) == []
    assert crud.get_segmentation_labels_in_dataset(db, dset_name) == []


def test_segmentation_area_no_hole(
    db: Session,
    poly_without_hole: schemas.PolygonWithHole,
    img1: schemas.Image,
):
    # sanity check nothing in db
    check_db_empty(db=db)

    crud.create_dataset(
        db,
        schemas.Dataset(name=dset_name, type=schemas.DatumTypes.IMAGE),
    )
    crud.create_groundtruth_segmentations(
        db,
        data=schemas.GroundTruthSegmentationsCreate(
            dataset_name=dset_name,
            segmentations=[
                schemas.GroundTruthSegmentation(
                    is_instance=True,
                    shape=[poly_without_hole],
                    image=img1,
                    labels=[schemas.Label(key="k1", value="v1")],
                )
            ],
        ),
    )

    segmentation = db.scalar(select(models.GroundTruthSegmentation))

    assert ops._raster_area(db, segmentation.shape) == math.ceil(
        45.5
    )  # area of mask will be an int


def test_segmentation_area_with_hole(
    db: Session, poly_with_hole: schemas.PolygonWithHole, img1: schemas.Image
):
    # sanity check nothing in db
    check_db_empty(db=db)

    crud.create_dataset(
        db,
        schemas.Dataset(name=dset_name, type=schemas.DatumTypes.IMAGE),
    )
    crud.create_groundtruth_segmentations(
        db,
        data=schemas.GroundTruthSegmentationsCreate(
            dataset_name=dset_name,
            segmentations=[
                schemas.GroundTruthSegmentation(
                    is_instance=False,
                    shape=[poly_with_hole],
                    image=img1,
                    labels=[schemas.Label(key="k1", value="v1")],
                )
            ],
        ),
    )

    segmentation = db.scalar(select(models.GroundTruthSegmentation))

    # give tolerance of 2 pixels because of poly -> mask conversion
    assert (ops._raster_area(db, segmentation.shape) - 92) <= 2


def test_segmentation_area_multi_polygon(
    db: Session,
    poly_with_hole: schemas.PolygonWithHole,
    poly_without_hole: schemas.PolygonWithHole,
    img1: schemas.Image,
):
    # sanity check nothing in db
    check_db_empty(db=db)

    crud.create_dataset(
        db,
        schemas.Dataset(name=dset_name, type=schemas.DatumTypes.IMAGE),
    )
    crud.create_groundtruth_segmentations(
        db,
        data=schemas.GroundTruthSegmentationsCreate(
            dataset_name=dset_name,
            segmentations=[
                schemas.GroundTruthSegmentation(
                    is_instance=True,
                    shape=[poly_with_hole, poly_without_hole],
                    image=img1,
                    labels=[schemas.Label(key="k1", value="v1")],
                )
            ],
        ),
    )

    segmentation = db.scalar(select(models.GroundTruthSegmentation))

    # the two shapes don't intersect so area should be sum of the areas
    # give tolerance of 2 pixels because of poly -> mask conversion
    assert (
        abs(ops._raster_area(db, segmentation.shape) - (math.ceil(45.5) + 92))
        <= 2
    )


def test__select_statement_from_poly(
    db: Session, poly_with_hole: schemas.PolygonWithHole, img: models.Datum
):
    gt_seg = db.scalar(
        insert(models.GroundTruthSegmentation)
        .values(
            [
                {
                    "shape": crud._create._select_statement_from_poly(
                        [poly_with_hole]
                    ),
                    "datum_id": img.id,
                    "is_instance": True,
                }
            ]
        )
        .returning(models.GroundTruthSegmentation)
    )
    db.add(gt_seg)
    db.commit()

    wkt = db.scalar(ST_AsText(ST_Polygon(gt_seg.shape)))

    # note the hole, which is a triangle, is jagged due to aliasing
    assert (
        wkt
        == "MULTIPOLYGON(((0 0,0 10,10 10,10 0,0 0),(2 4,2 8,3 8,3 7,4 7,4 6,5 6,5 5,6 5,6 4,2 4)))"
    )


def test_gt_seg_as_mask_or_polys(db: Session):
    """Check that a groundtruth segmentation can be created as a polygon or mask"""
    xmin, xmax, ymin, ymax = 11, 45, 37, 102
    h, w = 150, 200
    mask = np.zeros((h, w), dtype=bool)
    mask[ymin:ymax, xmin:xmax] = True
    mask_b64 = b64encode(np_to_bytes(mask)).decode()

    img = schemas.Image(uid="uid", height=h, width=w)

    poly = [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)]

    gt1 = schemas.GroundTruthSegmentation(
        is_instance=False,
        shape=mask_b64,
        image=img,
        labels=[schemas.Label(key="k1", value="v1")],
    )
    gt2 = schemas.GroundTruthSegmentation(
        is_instance=True,
        shape=[schemas.PolygonWithHole(polygon=poly)],
        image=img,
        labels=[schemas.Label(key="k1", value="v1")],
    )

    check_db_empty(db=db)

    crud.create_dataset(
        db,
        schemas.Dataset(name=dset_name, type=schemas.DatumTypes.IMAGE),
    )
    crud.create_groundtruth_segmentations(
        db,
        data=schemas.GroundTruthSegmentationsCreate(
            dataset_name=dset_name, segmentations=[gt1, gt2]
        ),
    )

    shapes = db.scalars(
        select(ST_AsText(ST_Polygon(models.GroundTruthSegmentation.shape)))
    ).all()

    assert len(shapes) == 2
    # check that the mask and polygon define the same polygons
    assert shapes[0] == shapes[1]

    # verify we get the same segmentations back
    segs = crud.get_groundtruth_segmentations_in_image(
        db, uid=img.uid, dataset_name=dset_name, are_instance=True
    )
    assert len(segs) == 1  # should just be one instance segmentation
    decoded_mask = bytes_to_pil(b64decode(segs[0].shape))
    decoded_mask_arr = np.array(decoded_mask)

    np.testing.assert_equal(decoded_mask_arr, mask)
    assert segs[0].image == gt1.image
    assert segs[0].labels == gt1.labels


def test_get_filtered_preds_statement_and_missing_labels(
    db: Session,
    gt_segs_create: schemas.GroundTruthDetectionsCreate,
    pred_segs_create: schemas.PredictedSegmentationsCreate,
):
    crud.create_dataset(
        db,
        schemas.Dataset(name=dset_name, type=schemas.DatumTypes.IMAGE),
    )
    crud.create_model(
        db, schemas.Model(name=model_name, type=schemas.DatumTypes.IMAGE)
    )

    # add three total ground truth segmentations, two of which are instance segmentations with
    # the same label.
    crud.create_groundtruth_segmentations(db, data=gt_segs_create)

    # finalize dataset
    crud.finalize_dataset(db, dset_name)

    # add three total predicted segmentations, two of which are instance segmentations
    crud.create_predicted_segmentations(db, pred_segs_create)

    gts_statement = crud._create._instance_segmentations_in_dataset_statement(
        dset_name
    )
    preds_statement = crud._read._model_instance_segmentation_preds_statement(
        model_name=model_name, dataset_name=dset_name
    )

    gts = db.scalars(gts_statement).all()
    preds = db.scalars(preds_statement).all()

    assert len(gts) == 3
    assert len(preds) == 3

    labels = crud._create._labels_in_query(db, gts_statement)
    assert len(labels) == 2

    # check get everything if the requested labels argument is empty
    (
        new_preds_statement,
        missing_pred_labels,
        ignored_pred_labels,
    ) = crud.get_filtered_preds_statement_and_missing_labels(
        db=db, gts_statement=gts_statement, preds_statement=preds_statement
    )

    gts = db.scalars(gts_statement).all()
    preds = db.scalars(new_preds_statement).all()

    assert len(gts) == 3
    # should not get the pred with label "k2", "v2" since its not
    # present in the groundtruths
    assert len(preds) == 1
    assert missing_pred_labels == [schemas.Label(key="k3", value="v3")]
    assert ignored_pred_labels == [schemas.Label(key="k2", value="v2")]


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
                dataset_gt_task_type=schemas.Task.BBOX_OBJECT_DETECTION,
                model_pred_task_type=schemas.Task.BBOX_OBJECT_DETECTION,
                label_key=label_key,
            ),
            iou_thresholds=[0.2, 0.6],
            ious_to_keep=[0.2],
        )

        (
            missing_pred_labels,
            ignored_pred_labels,
        ) = crud.validate_create_ap_metrics(db, request_info)

        return (
            crud.create_ap_metrics(
                db,
                request_info=request_info,
            ),
            missing_pred_labels,
            ignored_pred_labels,
        )

    # # check we get an error since the dataset is still a draft
    # with pytest.raises(exceptions.DatasetIsNotFinalizedError):
    #     method_to_test(label_key="class")

    # # finalize dataset
    # crud.finalize_dataset(db, "test dataset")

    # # now if we try again we should get an error that inferences aren't finalized
    # with pytest.raises(exceptions.InferencesAreNotFinalizedError):
    #     method_to_test(label_key="class")

    # verify we have no evaluations yet
    assert len(crud.get_model_evaluation_settings(db, model_name)) == 0

    # finalize inferences and try again
    # crud.finalize_inferences(db, model_name=model_name, dataset_name=dset_name)

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
        model_pred_task_type=enums.Task.BBOX_OBJECT_DETECTION,
        dataset_gt_task_type=enums.Task.BBOX_OBJECT_DETECTION,
        label_key="class",
        id=1,
    )
    assert model_evals[1] == schemas.EvaluationSettings(
        model_name=model_name,
        dataset_name=dset_name,
        model_pred_task_type=enums.Task.BBOX_OBJECT_DETECTION,
        dataset_gt_task_type=enums.Task.BBOX_OBJECT_DETECTION,
        label_key="class",
        min_area=min_area,
        max_area=max_area,
        id=2,
    )


def test_create_clf_metrics(db: Session, gt_clfs_create, pred_clfs_create):

    # create dataset
    crud.create_dataset(
        db,
        dataset=schemas.Dataset(name=dset_name, type=schemas.DatumTypes.IMAGE),
    )
    crud.create_ground_truth_classifications(db, gt_clfs_create)
    crud.finalize_dataset(db, dset_name)

    # create model
    crud.create_model(
        db, schemas.Model(name=model_name, type=schemas.DatumTypes.IMAGE)
    )
    crud.create_predicted_image_classifications(db, pred_clfs_create)
    crud.finalize_inferences(db, dataset_name=dset_name, model_name=model_name)

    request_info = schemas.ClfMetricsRequest(
        settings=schemas.EvaluationSettings(
            model_name=model_name, dataset_name=dset_name
        )
    )

    (
        missing_pred_keys,
        ignored_pred_keys,
    ) = crud.validate_create_clf_metrics(db, request_info=request_info)
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


def test__raster_to_png_b64(db: Session):
    # create a mask consisting of an ellipse with a whole in it
    w, h = 50, 100
    img = Image.new("1", size=(w, h))
    draw = ImageDraw.Draw(img)
    draw.ellipse((15, 40, 30, 70), fill=True)
    draw.ellipse((20, 50, 25, 60), fill=False)

    f = io.BytesIO()
    img.save(f, format="PNG")
    f.seek(0)
    b64_mask = b64encode(f.read()).decode()

    image = schemas.Image(uid="uid", height=h, width=w)
    crud.create_dataset(
        db,
        schemas.Dataset(name=dset_name, type=schemas.DatumTypes.IMAGE),
    )
    crud.create_groundtruth_segmentations(
        db,
        data=schemas.GroundTruthSegmentationsCreate(
            dataset_name=dset_name,
            segmentations=[
                schemas.GroundTruthSegmentation(
                    shape=b64_mask,
                    image=image,
                    labels=[schemas.Label(key="k", value="v")],
                    is_instance=True,
                )
            ],
        ),
    )

    seg = db.scalar(select(models.GroundTruthSegmentation))

    assert b64_mask == _raster_to_png_b64(db, seg.shape, image)


def test__instance_segmentations_in_dataset_statement(
    db: Session, gt_segs_create: schemas.GroundTruthSegmentationsCreate
):
    crud.create_dataset(
        db,
        schemas.Dataset(name=dset_name, type=schemas.DatumTypes.IMAGE),
    )
    crud.create_groundtruth_segmentations(db, data=gt_segs_create)

    areas = db.scalars(
        select(ST_Count(models.GroundTruthSegmentation.shape)).where(
            models.GroundTruthSegmentation.is_instance
        )
    ).all()

    assert sorted(areas) == [46, 90, 136]

    # sanity check no min_area and max_area arguments
    stmt = _instance_segmentations_in_dataset_statement(dataset_name=dset_name)
    assert len(db.scalars(stmt).all()) == 3

    # check min_area arg
    stmt = _instance_segmentations_in_dataset_statement(
        dataset_name=dset_name, min_area=45
    )
    assert len(db.scalars(stmt).all()) == 3
    stmt = _instance_segmentations_in_dataset_statement(
        dataset_name=dset_name, min_area=137
    )
    assert len(db.scalars(stmt).all()) == 0

    # check max_area argument
    stmt = _instance_segmentations_in_dataset_statement(
        dataset_name=dset_name, max_area=45
    )
    assert len(db.scalars(stmt).all()) == 0

    stmt = _instance_segmentations_in_dataset_statement(
        dataset_name=dset_name, max_area=136
    )
    assert len(db.scalars(stmt).all()) == 3

    # check specifying both min size and max size
    stmt = _instance_segmentations_in_dataset_statement(
        dataset_name=dset_name, min_area=45, max_area=136
    )
    assert len(db.scalars(stmt).all()) == 3
    stmt = _instance_segmentations_in_dataset_statement(
        dataset_name=dset_name, min_area=50, max_area=100
    )
    assert len(db.scalars(stmt).all()) == 1


def test___model_instance_segmentation_preds_statement(
    db: Session,
    gt_segs_create: schemas.GroundTruthSegmentationsCreate,
    pred_segs_create: schemas.PredictedSegmentationsCreate,
):
    crud.create_dataset(
        db,
        schemas.Dataset(name=dset_name, type=schemas.DatumTypes.IMAGE),
    )
    crud.create_groundtruth_segmentations(db, data=gt_segs_create)
    crud.finalize_dataset(db, dset_name)
    crud.create_model(
        db, schemas.Model(name=model_name, type=schemas.DatumTypes.IMAGE)
    )
    crud.create_predicted_segmentations(db, pred_segs_create)

    areas = db.scalars(
        select(ST_Count(models.PredictedSegmentation.shape)).where(
            models.PredictedSegmentation.is_instance
        )
    ).all()

    assert sorted(areas) == [95, 279, 1077]

    # sanity check no min_area and max_area arguments
    stmt = _model_instance_segmentation_preds_statement(
        dataset_name=dset_name, model_name=model_name
    )
    assert len(db.scalars(stmt).all()) == 3

    # check min_area arg
    stmt = _model_instance_segmentation_preds_statement(
        dataset_name=dset_name, model_name=model_name, min_area=94
    )
    assert len(db.scalars(stmt).all()) == 3
    stmt = _model_instance_segmentation_preds_statement(
        dataset_name=dset_name, model_name=model_name, min_area=1078
    )
    assert len(db.scalars(stmt).all()) == 0

    # check max_area argument
    stmt = _model_instance_segmentation_preds_statement(
        dataset_name=dset_name, model_name=model_name, max_area=94
    )
    assert len(db.scalars(stmt).all()) == 0

    stmt = _model_instance_segmentation_preds_statement(
        dataset_name=dset_name, model_name=model_name, max_area=1078
    )
    assert len(db.scalars(stmt).all()) == 3

    # check specifying both min size and max size
    stmt = _model_instance_segmentation_preds_statement(
        dataset_name=dset_name,
        model_name=model_name,
        min_area=94,
        max_area=1078,
    )
    assert len(db.scalars(stmt).all()) == 3
    stmt = _model_instance_segmentation_preds_statement(
        dataset_name=dset_name,
        model_name=model_name,
        min_area=200,
        max_area=300,
    )
    assert len(db.scalars(stmt).all()) == 1


def test___object_detections_in_dataset_statement(db: Session, groundtruths):
    # the groundtruths argument is not used but that fixture creates groundtruth
    # detections in the database

    areas = db.scalars(ST_Area(models.GroundTruthDetection.boundary)).all()

    # these are just to establish what the bounds on the areas of
    # the groundtruth detections are
    assert len(areas) == 20
    assert min(areas) > 94
    assert max(areas) < 326771
    assert len([a for a in areas if a > 500 and a < 1200]) == 9

    # sanity check no min_area and max_area arguments
    stmt = _object_detections_in_dataset_statement(
        dataset_name=dset_name, task=enums.Task.BBOX_OBJECT_DETECTION
    )
    assert len(db.scalars(stmt).all()) == 20

    # check min_area arg
    stmt = _object_detections_in_dataset_statement(
        dataset_name=dset_name,
        min_area=93,
        task=enums.Task.BBOX_OBJECT_DETECTION,
        task_for_area_computation=enums.Task.BBOX_OBJECT_DETECTION,
    )
    assert len(db.scalars(stmt).all()) == 20
    stmt = _object_detections_in_dataset_statement(
        dataset_name=dset_name,
        min_area=326771,
        task=enums.Task.BBOX_OBJECT_DETECTION,
        task_for_area_computation=enums.Task.BBOX_OBJECT_DETECTION,
    )
    assert len(db.scalars(stmt).all()) == 0

    # check max_area argument
    stmt = _object_detections_in_dataset_statement(
        dataset_name=dset_name,
        max_area=93,
        task=enums.Task.BBOX_OBJECT_DETECTION,
        task_for_area_computation=enums.Task.BBOX_OBJECT_DETECTION,
    )
    assert len(db.scalars(stmt).all()) == 0

    stmt = _object_detections_in_dataset_statement(
        dataset_name=dset_name,
        max_area=326771,
        task=enums.Task.BBOX_OBJECT_DETECTION,
        task_for_area_computation=enums.Task.BBOX_OBJECT_DETECTION,
    )
    assert len(db.scalars(stmt).all()) == 20

    # check specifying both min size and max size
    stmt = _object_detections_in_dataset_statement(
        dataset_name=dset_name,
        min_area=94,
        max_area=326771,
        task=enums.Task.BBOX_OBJECT_DETECTION,
        task_for_area_computation=enums.Task.BBOX_OBJECT_DETECTION,
    )
    assert len(db.scalars(stmt).all()) == 20
    stmt = _object_detections_in_dataset_statement(
        dataset_name=dset_name,
        min_area=500,
        max_area=1200,
        task=enums.Task.BBOX_OBJECT_DETECTION,
        task_for_area_computation=enums.Task.BBOX_OBJECT_DETECTION,
    )
    assert len(db.scalars(stmt).all()) == 9


def test__model_object_detection_preds_statement(
    db: Session, groundtruths, predictions
):
    # the groundtruths and predictions arguments are not used but the fixtures create predicted
    # detections in the database

    areas = db.scalars(ST_Area(models.PredictedDetection.boundary)).all()

    # these are just to establish what the bounds on the areas of
    # the groundtruth detections are
    assert len(areas) == 19
    assert min(areas) > 94
    assert max(areas) < 307274
    assert len([a for a in areas if a > 500 and a < 1200]) == 9

    # sanity check no min_area and max_area arguments
    stmt = _model_object_detection_preds_statement(
        dataset_name=dset_name,
        model_name=model_name,
        task=enums.Task.BBOX_OBJECT_DETECTION,
    )
    assert len(db.scalars(stmt).all()) == 19

    # check min_area arg
    stmt = _model_object_detection_preds_statement(
        dataset_name=dset_name,
        model_name=model_name,
        min_area=93,
        task=enums.Task.BBOX_OBJECT_DETECTION,
        task_for_area_computation=enums.Task.BBOX_OBJECT_DETECTION,
    )
    assert len(db.scalars(stmt).all()) == 19
    stmt = _model_object_detection_preds_statement(
        dataset_name=dset_name,
        model_name=model_name,
        min_area=326771,
        task=enums.Task.BBOX_OBJECT_DETECTION,
        task_for_area_computation=enums.Task.BBOX_OBJECT_DETECTION,
    )
    assert len(db.scalars(stmt).all()) == 0

    # check max_area argument
    stmt = _model_object_detection_preds_statement(
        dataset_name=dset_name,
        model_name=model_name,
        max_area=93,
        task=enums.Task.BBOX_OBJECT_DETECTION,
        task_for_area_computation=enums.Task.BBOX_OBJECT_DETECTION,
    )
    assert len(db.scalars(stmt).all()) == 0

    stmt = _model_object_detection_preds_statement(
        dataset_name=dset_name,
        model_name=model_name,
        max_area=326771,
        task=enums.Task.BBOX_OBJECT_DETECTION,
        task_for_area_computation=enums.Task.BBOX_OBJECT_DETECTION,
    )
    assert len(db.scalars(stmt).all()) == 19

    # check specifying both min size and max size
    stmt = _model_object_detection_preds_statement(
        dataset_name=dset_name,
        model_name=model_name,
        min_area=94,
        max_area=326771,
        task=enums.Task.BBOX_OBJECT_DETECTION,
        task_for_area_computation=enums.Task.BBOX_OBJECT_DETECTION,
    )
    assert len(db.scalars(stmt).all()) == 19
    stmt = _model_object_detection_preds_statement(
        dataset_name=dset_name,
        model_name=model_name,
        min_area=500,
        max_area=1200,
        task=enums.Task.BBOX_OBJECT_DETECTION,
        task_for_area_computation=enums.Task.BBOX_OBJECT_DETECTION,
    )
    assert len(db.scalars(stmt).all()) == 9


def test__filter_instance_segmentations_by_area(db: Session):
    crud.create_dataset(
        db,
        schemas.Dataset(name=dset_name, type=schemas.DatumTypes.IMAGE),
    )
    # triangle of area 150
    poly1 = schemas.PolygonWithHole(polygon=[(10, 20), (10, 40), (25, 20)])
    # rectangle of area 1050
    poly2 = schemas.PolygonWithHole(
        polygon=[(0, 5), (0, 40), (30, 40), (30, 5)]
    )

    img = schemas.Image(uid="", height=1000, width=2000)

    crud.create_groundtruth_segmentations(
        db,
        data=schemas.GroundTruthSegmentationsCreate(
            dataset_name=dset_name,
            segmentations=[
                schemas.GroundTruthSegmentation(
                    shape=[poly1],
                    image=img,
                    labels=[schemas.Label(key="k", value="v")],
                    is_instance=True,
                ),
                schemas.GroundTruthSegmentation(
                    shape=[poly2],
                    image=img,
                    labels=[schemas.Label(key="k", value="v")],
                    is_instance=True,
                ),
            ],
        ),
    )

    areas = db.scalars(ST_Count(models.GroundTruthSegmentation.shape)).all()
    assert sorted(areas) == [150, 1050]

    base_stmt = "SELECT id FROM ground_truth_segmentation WHERE ground_truth_segmentation.is_instance"

    # check filtering when use area determined by instance segmentation task
    stmt = _filter_instance_segmentations_by_area(
        stmt=base_stmt,
        seg_table=models.GroundTruthSegmentation,
        task_for_area_computation=enums.Task.INSTANCE_SEGMENTATION,
        min_area=100,
        max_area=2000,
    )
    assert len(db.scalars(stmt).all()) == 2

    stmt = _filter_instance_segmentations_by_area(
        base_stmt,
        seg_table=models.GroundTruthSegmentation,
        task_for_area_computation=enums.Task.INSTANCE_SEGMENTATION,
        min_area=100,
        max_area=200,
    )
    assert len(db.scalars(stmt).all()) == 1

    stmt = _filter_instance_segmentations_by_area(
        base_stmt,
        seg_table=models.GroundTruthSegmentation,
        task_for_area_computation=enums.Task.INSTANCE_SEGMENTATION,
        min_area=151,
        max_area=2000,
    )
    assert len(db.scalars(stmt).all()) == 1

    # now when we use bounding box detection task, the triangle becomes its circumscribing
    # rectangle (with area ~300) so we should get both segmentations
    stmt = _filter_instance_segmentations_by_area(
        base_stmt,
        seg_table=models.GroundTruthSegmentation,
        task_for_area_computation=enums.Task.BBOX_OBJECT_DETECTION,
        min_area=280,
        max_area=2000,
    )
    assert len(db.scalars(stmt).all()) == 2

    stmt = _filter_instance_segmentations_by_area(
        base_stmt,
        seg_table=models.GroundTruthSegmentation,
        task_for_area_computation=enums.Task.BBOX_OBJECT_DETECTION,
        min_area=301,
        max_area=2000,
    )
    assert len(db.scalars(stmt).all()) == 1

    # if we use polygon detection then the areas shouldn't change much (the area
    # of the triangle actually becomes 163-- not sure if this is aliasing or what)
    stmt = _filter_instance_segmentations_by_area(
        base_stmt,
        seg_table=models.GroundTruthSegmentation,
        task_for_area_computation=enums.Task.POLY_OBJECT_DETECTION,
        min_area=149,
        max_area=2000,
    )
    assert len(db.scalars(stmt).all()) == 2

    stmt = _filter_instance_segmentations_by_area(
        base_stmt,
        seg_table=models.GroundTruthSegmentation,
        task_for_area_computation=enums.Task.POLY_OBJECT_DETECTION,
        min_area=164,
        max_area=2000,
    )

    assert len(db.scalars(stmt).all()) == 1


def test__filter_object_detections_by_area(db: Session):
    crud.create_dataset(
        db,
        schemas.Dataset(name=dset_name, type=schemas.DatumTypes.IMAGE),
    )
    # triangle of area 150
    boundary1 = [(10, 20), (10, 40), (25, 20)]
    # rectangle of area 1050
    boundary2 = [(0, 5), (0, 40), (30, 40), (30, 5)]

    img = schemas.Image(uid="", height=1000, width=2000)

    crud.create_groundtruth_detections(
        db,
        data=schemas.GroundTruthDetectionsCreate(
            dataset_name=dset_name,
            detections=[
                schemas.GroundTruthDetection(
                    boundary=boundary1,
                    image=img,
                    labels=[schemas.Label(key="k", value="v")],
                ),
                schemas.GroundTruthDetection(
                    boundary=boundary2,
                    image=img,
                    labels=[schemas.Label(key="k", value="v")],
                ),
            ],
        ),
    )

    areas = db.scalars(ST_Area(models.GroundTruthDetection.boundary)).all()
    assert sorted(areas) == [150, 1050]

    # make base statement. need WHERE here because of what `_filter_instance_segmentations_by_area` expects
    base_stmt = "SELECT id FROM ground_truth_detection WHERE ground_truth_detection.id > 0"

    # check filtering when use area determined by polygon detection task
    stmt = _filter_object_detections_by_area(
        base_stmt,
        det_table=models.GroundTruthDetection,
        task_for_area_computation=enums.Task.POLY_OBJECT_DETECTION,
        min_area=100,
        max_area=2000,
    )
    assert len(db.scalars(stmt).all()) == 2

    stmt = _filter_object_detections_by_area(
        base_stmt,
        det_table=models.GroundTruthDetection,
        task_for_area_computation=enums.Task.POLY_OBJECT_DETECTION,
        min_area=100,
        max_area=200,
    )
    assert len(db.scalars(stmt).all()) == 1

    stmt = _filter_object_detections_by_area(
        base_stmt,
        det_table=models.GroundTruthDetection,
        task_for_area_computation=enums.Task.POLY_OBJECT_DETECTION,
        min_area=151,
        max_area=2000,
    )
    assert len(db.scalars(stmt).all()) == 1

    # now when we use bounding box detection task, the triangle becomes its circumscribing
    # rectangle (with area 300) so we should get both segmentations
    stmt = _filter_object_detections_by_area(
        base_stmt,
        det_table=models.GroundTruthDetection,
        task_for_area_computation=enums.Task.BBOX_OBJECT_DETECTION,
        min_area=299,
        max_area=2000,
    )
    assert len(db.scalars(stmt).all()) == 2

    stmt = _filter_object_detections_by_area(
        base_stmt,
        det_table=models.GroundTruthDetection,
        task_for_area_computation=enums.Task.BBOX_OBJECT_DETECTION,
        min_area=301,
        max_area=2000,
    )
    assert len(db.scalars(stmt).all()) == 1

    # check error if use the wrong task type
    with pytest.raises(ValueError) as exc_info:
        _filter_object_detections_by_area(
            base_stmt,
            det_table=models.GroundTruthDetection,
            task_for_area_computation=enums.Task.INSTANCE_SEGMENTATION,
            min_area=301,
            max_area=2000,
        )
    assert "Expected task_for_area_computation to be" in str(exc_info)


def test__filter_instance_segmentations_by_area_using_mask(db: Session):
    crud.create_dataset(
        db,
        schemas.Dataset(name=dset_name, type=schemas.DatumTypes.IMAGE),
    )
    # approximate triangle of area 150
    mask = np.zeros((1000, 2000), dtype=bool)
    for i in range(10):
        for j in range(30):
            if i + j < 20:
                mask[i, j] = True
    assert mask.sum() == 155
    mask_bytes = pil_to_bytes(Image.fromarray(mask))
    b64_mask = b64encode(mask_bytes).decode()

    # rectangle of area 1050
    boundary = [(0, 5), (0, 40), (30, 40), (30, 5)]

    img = schemas.Image(uid="", height=1000, width=2000)

    crud.create_groundtruth_segmentations(
        db,
        data=schemas.GroundTruthSegmentationsCreate(
            dataset_name=dset_name,
            segmentations=[
                schemas.GroundTruthSegmentation(
                    shape=b64_mask,
                    image=img,
                    labels=[schemas.Label(key="k", value="v")],
                    is_instance=True,
                ),
                schemas.GroundTruthSegmentation(
                    shape=[schemas.PolygonWithHole(polygon=boundary)],
                    image=img,
                    labels=[schemas.Label(key="k", value="v")],
                    is_instance=True,
                ),
            ],
        ),
    )

    areas = db.scalars(ST_Count(models.GroundTruthSegmentation.shape)).all()
    assert sorted(areas) == [155, 1050]

    base_stmt = "SELECT id FROM ground_truth_segmentation WHERE ground_truth_segmentation.is_instance"

    # check filtering when use area determined by polygon detection task
    stmt = _filter_instance_segmentations_by_area(
        base_stmt,
        seg_table=models.GroundTruthSegmentation,
        task_for_area_computation=enums.Task.POLY_OBJECT_DETECTION,
        min_area=100,
        max_area=2000,
    )
    assert len(db.scalars(stmt).all()) == 2

    stmt = _filter_instance_segmentations_by_area(
        base_stmt,
        seg_table=models.GroundTruthSegmentation,
        task_for_area_computation=enums.Task.POLY_OBJECT_DETECTION,
        min_area=100,
        max_area=200,
    )
    assert len(db.scalars(stmt).all()) == 1

    stmt = _filter_instance_segmentations_by_area(
        base_stmt,
        seg_table=models.GroundTruthSegmentation,
        task_for_area_computation=enums.Task.POLY_OBJECT_DETECTION,
        min_area=170,  # this won't pass at 156 due to aliasing
        max_area=2000,
    )
    assert len(db.scalars(stmt).all()) == 1

    # now when we use bounding box detection task, the triangle becomes its circumscribing
    # rectangle (with area 200) so we should get both segmentations
    stmt = _filter_instance_segmentations_by_area(
        base_stmt,
        seg_table=models.GroundTruthSegmentation,
        task_for_area_computation=enums.Task.BBOX_OBJECT_DETECTION,
        min_area=160,
        max_area=2000,
    )

    assert len(db.scalars(stmt).all()) == 2

    stmt = _filter_instance_segmentations_by_area(
        base_stmt,
        seg_table=models.GroundTruthSegmentation,
        task_for_area_computation=enums.Task.BBOX_OBJECT_DETECTION,
        min_area=300,
        max_area=2000,
    )
    assert len(db.scalars(stmt).all()) == 1

    stmt = _filter_instance_segmentations_by_area(
        base_stmt,
        seg_table=models.GroundTruthSegmentation,
        task_for_area_computation=enums.Task.BBOX_OBJECT_DETECTION,
        min_area=3000,
        max_area=10000,
    )
    assert len(db.scalars(stmt).all()) == 0


def test__validate_and_update_evaluation_settings_task_type_for_detection_no_groundtruth(
    db: Session,
):
    """Test runtime error when there's no groundtruth data"""
    crud.create_dataset(
        db,
        schemas.Dataset(name=dset_name, type=schemas.DatumTypes.IMAGE),
    )
    crud.finalize_dataset(db, dset_name)
    crud.create_model(
        db, schemas.Model(name=model_name, type=schemas.DatumTypes.IMAGE)
    )
    crud.finalize_inferences(db, model_name=model_name, dataset_name=dset_name)

    evaluation_settings = schemas.EvaluationSettings(
        model_name=model_name, dataset_name=dset_name
    )

    with pytest.raises(RuntimeError) as exc_info:
        _validate_and_update_evaluation_settings_task_type_for_detection(
            db, evaluation_settings
        )
    assert "The dataset does not have any annotations to support" in str(
        exc_info
    )


def test__validate_and_update_evaluation_settings_task_type_for_detection_no_predictions(
    db: Session, gt_dets_create
):
    """Test runtime error when there's no prediction data"""

    # create dataset
    crud.create_dataset(
        db,
        schemas.Dataset(name=dset_name, type=schemas.DatumTypes.IMAGE),
    )
    crud.create_groundtruth_detections(db, gt_dets_create)
    crud.finalize_dataset(db, dset_name)

    # create model
    crud.create_model(
        db, schemas.Model(name=model_name, type=schemas.DatumTypes.IMAGE)
    )
    crud.finalize_inferences(db, model_name=model_name, dataset_name=dset_name)

    evaluation_settings = schemas.EvaluationSettings(
        model_name=model_name, dataset_name=dset_name
    )

    with pytest.raises(RuntimeError) as exc_info:
        _validate_and_update_evaluation_settings_task_type_for_detection(
            db, evaluation_settings
        )
    assert "The model does not have any inferences to support" in str(exc_info)


def test__validate_and_update_evaluation_settings_task_type_for_detection_multiple_groundtruth_types(
    db: Session, gt_dets_create, gt_segs_create
):
    # create dataset
    crud.create_dataset(
        db,
        schemas.Dataset(name=dset_name, type=schemas.DatumTypes.IMAGE),
    )
    crud.create_groundtruth_detections(db, gt_dets_create)
    crud.create_groundtruth_segmentations(db, gt_segs_create)
    crud.finalize_dataset(db, dset_name)

    # create model
    crud.create_model(
        db, schemas.Model(name=model_name, type=schemas.DatumTypes.IMAGE)
    )
    crud.finalize_inferences(db, model_name=model_name, dataset_name=dset_name)

    evaluation_settings = schemas.EvaluationSettings(
        model_name=model_name, dataset_name=dset_name
    )

    with pytest.raises(RuntimeError) as exc_info:
        _validate_and_update_evaluation_settings_task_type_for_detection(
            db, evaluation_settings
        )
    assert "The dataset has the following tasks compatible" in str(exc_info)

    # now specify task types for dataset and check we get an error since model
    # has no inferences
    evaluation_settings = schemas.EvaluationSettings(
        model_name=model_name,
        dataset_name=dset_name,
        dataset_gt_task_type=enums.Task.BBOX_OBJECT_DETECTION,
    )
    with pytest.raises(RuntimeError) as exc_info:
        _validate_and_update_evaluation_settings_task_type_for_detection(
            db, evaluation_settings
        )
    assert "The model does not have any inferences to support" in str(exc_info)


def test__validate_and_update_evaluation_settings_task_type_for_detection_multiple_prediction_types(
    db: Session, gt_dets_create, pred_dets_create, pred_segs_create
):
    # create dataset
    crud.create_dataset(
        db,
        schemas.Dataset(name=dset_name, type=schemas.DatumTypes.IMAGE),
    )
    crud.create_groundtruth_detections(db, gt_dets_create)
    crud.finalize_dataset(db, dset_name)

    # create model
    crud.create_model(
        db, schemas.Model(name=model_name, type=schemas.DatumTypes.IMAGE)
    )
    crud.create_predicted_detections(db, pred_dets_create)
    crud.create_predicted_segmentations(db, pred_segs_create)
    crud.finalize_inferences(db, model_name=model_name, dataset_name=dset_name)

    evaluation_settings = schemas.EvaluationSettings(
        model_name=model_name, dataset_name=dset_name
    )

    with pytest.raises(RuntimeError) as exc_info:
        _validate_and_update_evaluation_settings_task_type_for_detection(
            db, evaluation_settings
        )
    assert "The model has the following tasks compatible" in str(exc_info)

    # now specify task type for model and check there's no error and that
    # the dataset task type was made explicit
    evaluation_settings = schemas.EvaluationSettings(
        model_name=model_name,
        dataset_name=dset_name,
        model_pred_task_type=enums.Task.BBOX_OBJECT_DETECTION,
    )
    assert evaluation_settings.dataset_gt_task_type is None
    _validate_and_update_evaluation_settings_task_type_for_detection(
        db, evaluation_settings
    )
    assert (
        evaluation_settings.dataset_gt_task_type
        == enums.Task.POLY_OBJECT_DETECTION
    )


def test_create_datums_with_metadata(db: Session):
    crud.create_dataset(
        db,
        schemas.Dataset(name=dset_name, type=schemas.DatumTypes.TABULAR),
    )

    datums = [
        schemas.Datum(
            uid="uid1",
            metadata=[
                schemas.DatumMetadatum(name="name1", value=0.7),
                schemas.DatumMetadatum(name="name2", value="a string"),
            ],
        ),
        schemas.Datum(
            uid="uid2",
            metadata=[
                schemas.DatumMetadatum(name="name2", value="a string"),
                schemas.DatumMetadatum(
                    name="name3",
                    value={
                        "type": "Point",
                        "coordinates": [-48.23456, 20.12345],
                    },
                ),
            ],
        ),
    ]
    crud._create._add_datums_to_dataset(db, dset_name, datums)

    # check there should only be three unique metadatums since two are the same
    assert len(db.scalars(select(models.Metadatum)).all()) == 3

    db_datums = crud.get_datums_in_dataset(db, dset_name)

    assert len(db_datums) == 2

    md1 = db_datums[0].datum_metadatum_links[0].metadatum
    assert md1.name == "name1"
    assert md1.numeric_value == 0.7
    assert md1.string_value is None
    assert md1.geo is None

    md2 = db_datums[1].datum_metadatum_links[0].metadatum
    assert md2.name == "name2"
    assert md2.numeric_value is None
    assert md2.string_value == "a string"
    assert md2.geo is None

    md3 = db_datums[1].datum_metadatum_links[1].metadatum
    assert md3.name == "name3"
    assert md3.numeric_value is None
    assert md3.string_value is None
    assert json.loads(db.scalar(ST_AsGeoJSON(md3.geo))) == {
        "type": "Point",
        "coordinates": [-48.23456, 20.12345],
    }
