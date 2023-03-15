import io
import math
from base64 import b64decode, b64encode

import numpy as np
import pytest
from geoalchemy2.functions import ST_AsText, ST_Polygon
from PIL import Image
from sqlalchemy import func, insert, select
from sqlalchemy.orm import Session

from velour_api import crud, enums, exceptions, models, ops, schemas

dset_name = "test dataset"
model_name = "test model"


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
        models.Image,
        models.GroundTruthDetection,
        models.LabeledGroundTruthDetection,
        models.PredictedDetection,
        models.Label,
        models.Dataset,
        models.GroundTruthImageClassification,
        models.PredictedImageClassification,
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
                boundary=[(10, 20), (10, 30), (20, 30), (20, 20)],
                labels=[
                    schemas.Label(key="k1", value="v1"),
                    schemas.Label(key="k2", value="v2"),
                ],
                image=img1,
            ),
            schemas.GroundTruthDetection(
                boundary=[(10, 20), (10, 30), (20, 30), (20, 20)],
                labels=[schemas.Label(key="k2", value="v2")],
                image=img1,
            ),
        ],
    )


@pytest.fixture
def pred_dets_create(img1: schemas.Image) -> schemas.PredictedDetectionsCreate:
    return schemas.PredictedDetectionsCreate(
        model_name=model_name,
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
        ],
    )


@pytest.fixture
def pred_segs_create(
    mask_bytes1: bytes, mask_bytes2: bytes, img1: schemas.Image
) -> schemas.PredictedSegmentationsCreate:
    b64_mask1 = b64encode(mask_bytes1).decode()
    b64_mask2 = b64encode(mask_bytes2).decode()
    return schemas.PredictedSegmentationsCreate(
        model_name=model_name,
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
        ],
    )


@pytest.fixture
def gt_clfs_create(
    img1: schemas.Image, img2: schemas.Image
) -> schemas.GroundTruthImageClassificationsCreate:
    return schemas.GroundTruthImageClassificationsCreate(
        dataset_name=dset_name,
        classifications=[
            schemas.ImageClassificationBase(
                image=img1,
                labels=[
                    schemas.Label(key="k1", value="v1"),
                    schemas.Label(key="k2", value="v2"),
                ],
            ),
            schemas.ImageClassificationBase(
                image=img2,
                labels=[schemas.Label(key="k2", value="v2")],
            ),
        ],
    )


@pytest.fixture
def pred_clfs_create(
    img1: schemas.Image, img2: schemas.Image
) -> schemas.PredictedImageClassificationsCreate:
    return schemas.PredictedImageClassificationsCreate(
        model_name=model_name,
        classifications=[
            schemas.PredictedImageClassification(
                image=img1,
                scored_labels=[
                    schemas.ScoredLabel(
                        label=schemas.Label(key="k1", value="v1"), score=0.2
                    ),
                    schemas.ScoredLabel(
                        label=schemas.Label(key="k4", value="v4"), score=0.5
                    ),
                ],
            ),
            schemas.PredictedImageClassification(
                image=img2,
                scored_labels=[
                    schemas.ScoredLabel(
                        label=schemas.Label(key="k2", value="v2"), score=0.3
                    ),
                    schemas.ScoredLabel(
                        label=schemas.Label(key="k3", value="v3"), score=0.87
                    ),
                ],
            ),
        ],
    )


def test_create_and_get_datasets(db: Session):
    crud.create_dataset(db, schemas.DatasetCreate(name=dset_name))

    all_datasets = db.scalars(select(models.Dataset)).all()
    assert len(all_datasets) == 1
    assert all_datasets[0].name == dset_name

    with pytest.raises(exceptions.DatasetAlreadyExistsError) as exc_info:
        crud.create_dataset(db, schemas.DatasetCreate(name=dset_name))
    assert "already exists" in str(exc_info)

    crud.create_dataset(db, schemas.DatasetCreate(name="other dataset"))
    datasets = crud.get_datasets(db)
    assert len(datasets) == 2
    assert set([d.name for d in datasets]) == {dset_name, "other dataset"}


def test_create_and_get_models(db: Session):
    crud.create_model(db, schemas.Model(name=model_name))

    all_models = db.scalars(select(models.Model)).all()
    assert len(all_models) == 1
    assert all_models[0].name == model_name

    with pytest.raises(exceptions.ModelAlreadyExistsError) as exc_info:
        crud.create_model(db, schemas.Model(name=model_name))
    assert "already exists" in str(exc_info)

    crud.create_model(db, schemas.Model(name="other model"))
    db_models = crud.get_models(db)
    assert len(db_models) == 2
    assert set([m.name for m in db_models]) == {model_name, "other model"}


def test_get_dataset(db: Session):
    with pytest.raises(exceptions.DatasetDoesNotExistError) as exc_info:
        crud.get_dataset(db, dset_name)
    assert "does not exist" in str(exc_info)

    crud.create_dataset(db, schemas.DatasetCreate(name=dset_name))
    dset = crud.get_dataset(db, dset_name)
    assert dset.name == dset_name


def test_get_model(db: Session):
    with pytest.raises(exceptions.ModelDoesNotExistError) as exc_info:
        crud.get_model(db, model_name)
    assert "does not exist" in str(exc_info)

    crud.create_model(db, schemas.Model(name=model_name))
    model = crud.get_model(db, model_name)
    assert model.name == model_name


def test_create_ground_truth_detections_and_delete_dataset(
    db: Session, gt_dets_create: schemas.GroundTruthDetectionsCreate
):
    # sanity check nothing in db
    check_db_empty(db=db)

    crud.create_dataset(db, schemas.DatasetCreate(name=dset_name))

    crud.create_groundtruth_detections(db, data=gt_dets_create)

    assert crud.number_of_rows(db, models.GroundTruthDetection) == 2
    assert crud.number_of_rows(db, models.Image) == 1
    assert crud.number_of_rows(db, models.LabeledGroundTruthDetection) == 3
    assert crud.number_of_rows(db, models.Label) == 2

    # delete dataset and check the cascade worked
    crud.delete_dataset(db, dataset_name=dset_name)
    for model_cls in [
        models.Dataset,
        models.Image,
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
    # check this gives an error since the model hasn't been added yet
    with pytest.raises(exceptions.ModelDoesNotExistError) as exc_info:
        crud.create_predicted_detections(db, pred_dets_create)
    assert "does not exist" in str(exc_info)

    crud.create_model(db, schemas.Model(name=model_name))

    # check this gives an error since the images haven't been added yet
    with pytest.raises(exceptions.ImageDoesNotExistError) as exc_info:
        crud.create_predicted_detections(db, pred_dets_create)
    assert "Image with uid" in str(exc_info)

    # create dataset, add images, and add predictions
    crud.create_dataset(db, schemas.DatasetCreate(name=dset_name))
    crud.create_groundtruth_detections(db, gt_dets_create)
    crud.create_predicted_detections(db, pred_dets_create)

    # check db has the added predictions
    assert crud.number_of_rows(db, models.PredictedDetection) == 2
    assert crud.number_of_rows(db, models.LabeledPredictedDetection) == 3

    # delete model and check all detections from it are gone
    crud.delete_model(db, model_name)
    assert crud.number_of_rows(db, models.Model) == 0
    assert crud.number_of_rows(db, models.PredictedDetection) == 0
    assert crud.number_of_rows(db, models.LabeledPredictedDetection) == 0


def test_create_ground_truth_classifications_and_delete_dataset(
    db: Session, gt_clfs_create: schemas.GroundTruthImageClassificationsCreate
):
    crud.create_dataset(db, schemas.DatasetCreate(name=dset_name))
    crud.create_ground_truth_image_classifications(db, gt_clfs_create)

    # should have three GroundTruthImageClassification rows since one image has two
    # labels and the other has one
    assert crud.number_of_rows(db, models.GroundTruthImageClassification) == 3
    assert crud.number_of_rows(db, models.Image) == 2
    assert crud.number_of_rows(db, models.Label) == 2

    # delete dataset and check the cascade worked
    crud.delete_dataset(db, dataset_name=dset_name)
    for model_cls in [
        models.Dataset,
        models.Image,
        models.GroundTruthImageClassification,
    ]:
        assert crud.number_of_rows(db, model_cls) == 0

    # make sure labels are still there`
    assert crud.number_of_rows(db, models.Label) == 2


def test_create_predicted_classifications_and_delete_model(
    db: Session,
    pred_clfs_create: schemas.PredictedImageClassification,
    gt_clfs_create: schemas.GroundTruthImageClassificationsCreate,
):
    # check this gives an error since the model hasn't been added yet
    with pytest.raises(exceptions.ModelDoesNotExistError) as exc_info:
        crud.create_predicted_image_classifications(db, pred_clfs_create)
    assert "does not exist" in str(exc_info)

    crud.create_model(db, schemas.Model(name=model_name))

    # check this gives an error since the images haven't been added yet
    with pytest.raises(exceptions.ImageDoesNotExistError) as exc_info:
        crud.create_predicted_image_classifications(db, pred_clfs_create)
    assert "Image with uid" in str(exc_info)

    # create dataset, add images, and add predictions
    crud.create_dataset(db, schemas.DatasetCreate(name=dset_name))
    crud.create_ground_truth_image_classifications(db, gt_clfs_create)
    crud.create_predicted_image_classifications(db, pred_clfs_create)

    # check db has the added predictions
    assert crud.number_of_rows(db, models.PredictedImageClassification) == 4

    # delete model and check all detections from it are gone
    crud.delete_model(db, model_name)
    assert crud.number_of_rows(db, models.Model) == 0
    assert crud.number_of_rows(db, models.PredictedDetection) == 0
    assert crud.number_of_rows(db, models.LabeledPredictedDetection) == 0


def test_create_ground_truth_segmentations_and_delete_dataset(
    db: Session, gt_segs_create: schemas.GroundTruthDetectionsCreate
):
    # sanity check nothing in db
    check_db_empty(db=db)

    crud.create_dataset(db, schemas.DatasetCreate(name=dset_name))

    crud.create_groundtruth_segmentations(db, data=gt_segs_create)

    assert crud.number_of_rows(db, models.GroundTruthSegmentation) == 3
    assert crud.number_of_rows(db, models.Image) == 2
    assert crud.number_of_rows(db, models.LabeledGroundTruthSegmentation) == 3
    assert crud.number_of_rows(db, models.Label) == 2

    # delete dataset and check the cascade worked
    crud.delete_dataset(db, dataset_name=dset_name)
    for model_cls in [
        models.Dataset,
        models.Image,
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

    crud.create_model(db, schemas.Model(name=model_name))

    # check this gives an error since the images haven't been added yet
    with pytest.raises(exceptions.ImageDoesNotExistError) as exc_info:
        crud.create_predicted_segmentations(db, pred_segs_create)
    assert "Image with uid" in str(exc_info)

    # create dataset, add images, and add predictions
    crud.create_dataset(db, schemas.DatasetCreate(name=dset_name))
    crud.create_groundtruth_segmentations(db, gt_segs_create)
    crud.create_predicted_segmentations(db, pred_segs_create)

    # check db has the added predictions
    assert crud.number_of_rows(db, models.PredictedSegmentation) == 3
    assert crud.number_of_rows(db, models.LabeledPredictedSegmentation) == 3

    # grab the first one and check that the area of the raster
    # matches the area of the image
    img = crud.get_image(db, "uid1")
    seg = img.predicted_segmentations[0]
    mask = bytes_to_pil(
        b64decode(pred_segs_create.segmentations[0].base64_mask)
    )
    assert ops.seg_area(db, seg) == np.array(mask).sum()

    # delete model and check all detections from it are gone
    crud.delete_model(db, model_name)
    assert crud.number_of_rows(db, models.Model) == 0
    assert crud.number_of_rows(db, models.PredictedSegmentation) == 0
    assert crud.number_of_rows(db, models.LabeledPredictedSegmentation) == 0


def test_get_labels(
    db: Session, gt_dets_create: schemas.GroundTruthDetectionsCreate
):
    crud.create_dataset(db, schemas.DatasetCreate(name=dset_name))
    crud.create_groundtruth_detections(db, data=gt_dets_create)
    labels = crud.get_labels_in_dataset(db, dset_name)

    assert len(labels) == 2
    assert set([(label.key, label.value) for label in labels]) == set(
        [("k1", "v1"), ("k2", "v2")]
    )


def test_segmentation_area_no_hole(
    db: Session,
    poly_without_hole: schemas.PolygonWithHole,
    img1: schemas.Image,
):
    # sanity check nothing in db
    check_db_empty(db=db)

    crud.create_dataset(db, schemas.DatasetCreate(name=dset_name))
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

    assert ops.seg_area(db, segmentation) == math.ceil(
        45.5
    )  # area of mask will be an int


def test_segmentation_area_with_hole(
    db: Session, poly_with_hole: schemas.PolygonWithHole, img1: schemas.Image
):
    # sanity check nothing in db
    check_db_empty(db=db)

    crud.create_dataset(db, schemas.DatasetCreate(name=dset_name))
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
    assert (ops.seg_area(db, segmentation) - 92) <= 2


def test_segmentation_area_multi_polygon(
    db: Session,
    poly_with_hole: schemas.PolygonWithHole,
    poly_without_hole: schemas.PolygonWithHole,
    img1: schemas.Image,
):
    # sanity check nothing in db
    check_db_empty(db=db)

    crud.create_dataset(db, schemas.DatasetCreate(name=dset_name))
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
    assert abs(ops.seg_area(db, segmentation) - (math.ceil(45.5) + 92)) <= 2


def test__select_statement_from_poly(
    db: Session, poly_with_hole: schemas.PolygonWithHole, img: models.Image
):
    gt_seg = db.scalar(
        insert(models.GroundTruthSegmentation)
        .values(
            [
                {
                    "shape": crud._select_statement_from_poly(
                        [poly_with_hole]
                    ),
                    "image_id": img.id,
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


def test_gt_seg_as_mask_or_polys(db: Session, img1: schemas.Image):
    """Check that a groundtruth segmentation can be created as a polygon or mask"""
    xmin, xmax, ymin, ymax = 11, 45, 37, 102
    h, w = 150, 200
    mask = np.zeros((h, w), dtype=bool)
    mask[ymin:ymax, xmin:xmax] = True
    mask_b64 = b64encode(np_to_bytes(mask)).decode()

    poly = [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)]

    gt1 = schemas.GroundTruthSegmentation(
        is_instance=False,
        shape=mask_b64,
        image=img1,
        labels=[schemas.Label(key="k1", value="v1")],
    )
    gt2 = schemas.GroundTruthSegmentation(
        is_instance=True,
        shape=[schemas.PolygonWithHole(polygon=poly)],
        image=img1,
        labels=[schemas.Label(key="k1", value="v1")],
    )

    check_db_empty(db=db)

    crud.create_dataset(db, schemas.DatasetCreate(name=dset_name))
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


def test_validate_requested_labels_and_get_new_defining_statements_and_missing_labels(
    db: Session,
    gt_segs_create: schemas.GroundTruthDetectionsCreate,
    pred_segs_create: schemas.PredictedSegmentationsCreate,
):
    crud.create_dataset(db, schemas.DatasetCreate(name=dset_name))
    crud.create_model(db, schemas.Model(name=model_name))

    # add three total ground truth segmentations, two of which are instance segmentations with
    # the same label.
    crud.create_groundtruth_segmentations(db, data=gt_segs_create)
    # add three total predicted segmentations, two of which are instance segmentations
    crud.create_predicted_segmentations(db, pred_segs_create)

    gts_statement = crud.instance_segmentations_in_dataset_statement(dset_name)
    preds_statement = crud.model_instance_segmentation_preds_statement(
        model_name=model_name, dataset_name=dset_name
    )

    gts = db.scalars(gts_statement).all()
    preds = db.scalars(preds_statement).all()

    assert len(gts) == 2
    assert len(preds) == 2

    labels = crud.labels_in_query(db, gts_statement)
    assert len(labels) == 2

    # now query just the one with label "k1", "v1"
    (
        new_gts_statement,
        new_preds_statement,
        missing_pred_labels,
        ignored_pred_labels,
    ) = crud.validate_requested_labels_and_get_new_defining_statements_and_missing_labels(
        db=db,
        gts_statement=gts_statement,
        preds_statement=preds_statement,
        requested_labels=[schemas.Label(key="k1", value="v1")],
    )

    gts = db.scalars(new_gts_statement).all()
    preds = db.scalars(new_preds_statement).all()

    assert len(gts) == 1
    assert (gts[0].label.key, gts[0].label.value) == ("k1", "v1")
    assert len(preds) == 1
    assert (preds[0].label.key, preds[0].label.value) == ("k1", "v1")
    assert missing_pred_labels == []
    assert ignored_pred_labels == [schemas.Label(key="k2", value="v2")]

    # # check error when requesting a label that doesn't exist
    with pytest.raises(ValueError) as exc_info:
        crud.validate_requested_labels_and_get_new_defining_statements_and_missing_labels(
            db=db,
            gts_statement=gts_statement,
            preds_statement=preds_statement,
            requested_labels=[schemas.Label(key="k1", value="v2")],
        )
    assert "The following label key/value pairs are missing" in str(exc_info)

    # check get everything if the requested labels argument is empty
    (
        new_gts_statement,
        new_preds_statement,
        missing_pred_labels,
        ignored_pred_labels,
    ) = crud.validate_requested_labels_and_get_new_defining_statements_and_missing_labels(
        db=db, gts_statement=gts_statement, preds_statement=preds_statement
    )

    gts = db.scalars(new_gts_statement).all()
    preds = db.scalars(new_preds_statement).all()

    assert len(gts) == 2
    # should not get the pred with label "k2", "v2" since its not
    # present in the groundtruths
    assert len(preds) == 1
    assert missing_pred_labels == [schemas.Label(key="k3", value="v3")]
    assert ignored_pred_labels == [schemas.Label(key="k2", value="v2")]


def test_create_ap_metrics(db: Session, groundtruths, predictions):
    # the groundtruths and predictions arguments are not used but
    # those fixtures create the necessary dataset, model, groundtruths, and predictions
    (
        ap_metric_ids,
        missing_pred_labels,
        ignored_pred_labels,
    ) = crud.create_ap_metrics(
        db,
        request_info=schemas.APRequest(
            parameters=schemas.MetricParameters(
                model_name="test model",
                dataset_name="test dataset",
                model_pred_type=enums.Task.OBJECT_DETECTION,
                dataset_gt_type=enums.Task.OBJECT_DETECTION,
            )
        ),
        iou_thresholds=[0.2, 0.6],
    )

    assert missing_pred_labels == []
    assert ignored_pred_labels == [schemas.Label(key="class", value="3")]

    metrics = db.scalars(
        select(models.APMetric).where(models.APMetric.id.in_(ap_metric_ids))
    ).all()

    for metric in metrics:
        assert metric.iou_threshold in [0.2, 0.6, [0.2, 0.6]]

    # should be five labels (since thats how many are in groundtruth set)
    assert len(set(m.label_id for m in metrics)) == 5
