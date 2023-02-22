import io

import numpy as np
import pytest
from geoalchemy2.functions import ST_Area
from PIL import Image
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from velour_api import crud, models, ops, schemas

dset_name = "test dataset"
model_name = "test model"


def random_mask_bytes(size: tuple[int, int]) -> bytes:
    mask = np.random.randint(0, 2, size=size, dtype=bool)
    mask = Image.fromarray(mask)
    f = io.BytesIO()
    mask.save(f, format="PNG")
    f.seek(0)
    return f.read()


def bytes_to_pil(b: bytes) -> Image.Image:
    f = io.BytesIO(b)
    img = Image.open(f)
    return img


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
    return schemas.PolygonWithHole(polygon=[(4, 10), (9, 7), (11, 2), (2, 2)])


@pytest.fixture
def poly_with_hole() -> schemas.PolygonWithHole:
    # should have area 100 - 8 = 92
    return schemas.PolygonWithHole(
        polygon=[(0, 10), (10, 10), (10, 0), (0, 0)],
        hole=[(2, 4), (2, 8), (6, 4)],
    )


@pytest.fixture
def gt_dets_create() -> schemas.GroundTruthDetectionsCreate:
    return schemas.GroundTruthDetectionsCreate(
        dataset_name=dset_name,
        detections=[
            schemas.GroundTruthDetection(
                boundary=[(10, 20), (10, 30), (20, 30), (20, 20)],
                labels=[
                    schemas.Label(key="k1", value="v1"),
                    schemas.Label(key="k2", value="v2"),
                ],
                image=schemas.Image(uri="uri1"),
            ),
            schemas.GroundTruthDetection(
                boundary=[(10, 20), (10, 30), (20, 30), (20, 20)],
                labels=[schemas.Label(key="k2", value="v2")],
                image=schemas.Image(uri="uri1"),
            ),
        ],
    )


@pytest.fixture
def pred_dets_create() -> schemas.PredictedDetectionsCreate:
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
                image=schemas.Image(uri="uri1"),
            ),
            schemas.PredictedDetection(
                boundary=[(107, 207), (107, 307), (207, 307), (207, 207)],
                scored_labels=[
                    schemas.ScoredLabel(
                        label=schemas.Label(key="k2", value="v2"), score=0.9
                    )
                ],
                image=schemas.Image(uri="uri1"),
            ),
        ],
    )


@pytest.fixture
def gt_segs_create(
    poly_with_hole, poly_without_hole
) -> schemas.GroundTruthSegmentationsCreate:
    return schemas.GroundTruthSegmentationsCreate(
        dataset_name=dset_name,
        segmentations=[
            schemas.GroundTruthSegmentation(
                shape=[poly_with_hole],
                image=schemas.Image(uri="uri1"),
                labels=[schemas.Label(key="k1", value="v1")],
            ),
            schemas.GroundTruthSegmentation(
                shape=[poly_without_hole],
                image=schemas.Image(uri="uri2"),
                labels=[schemas.Label(key="k1", value="v1")],
            ),
        ],
    )


@pytest.fixture
def pred_segs_create() -> schemas.PredictedSegmentationsCreate:
    mask_bytes1 = random_mask_bytes(size=(32, 64))
    mask_bytes2 = random_mask_bytes(size=(16, 12))
    return schemas.PredictedSegmentationsCreate(
        model_name=model_name,
        segmentations=[
            schemas.PredictedSegmentation(
                shape=mask_bytes1,
                image=schemas.Image(uri="uri1"),
                scored_labels=[
                    schemas.ScoredLabel(
                        label=schemas.Label(key="k1", value="v1"), score=0.43
                    )
                ],
            ),
            schemas.PredictedSegmentation(
                shape=mask_bytes2,
                image=schemas.Image(uri="uri1"),
                scored_labels=[
                    schemas.ScoredLabel(
                        label=schemas.Label(key="k2", value="v2"), score=0.97
                    )
                ],
            ),
        ],
    )


@pytest.fixture
def gt_clfs_create() -> schemas.GroundTruthImageClassificationsCreate:
    return schemas.GroundTruthImageClassificationsCreate(
        dataset_name=dset_name,
        classifications=[
            schemas.ImageClassificationBase(
                image=schemas.Image(uri="uri1"),
                labels=[
                    schemas.Label(key="k1", value="v1"),
                    schemas.Label(key="k2", value="v2"),
                ],
            ),
            schemas.ImageClassificationBase(
                image=schemas.Image(uri="uri2"),
                labels=[schemas.Label(key="k2", value="v2")],
            ),
        ],
    )


@pytest.fixture
def pred_clfs_create() -> schemas.PredictedImageClassificationsCreate:
    return schemas.PredictedImageClassificationsCreate(
        model_name=model_name,
        classifications=[
            schemas.PredictedImageClassification(
                image=schemas.Image(uri="uri1"),
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
                image=schemas.Image(uri="uri2"),
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

    with pytest.raises(crud.DatasetAlreadyExistsError) as exc_info:
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

    with pytest.raises(crud.ModelAlreadyExistsError) as exc_info:
        crud.create_model(db, schemas.Model(name=model_name))
    assert "already exists" in str(exc_info)

    crud.create_model(db, schemas.Model(name="other model"))
    db_models = crud.get_models(db)
    assert len(db_models) == 2
    assert set([m.name for m in db_models]) == {model_name, "other model"}


def test_get_dataset(db: Session):
    with pytest.raises(crud.DatasetDoesNotExistError) as exc_info:
        crud.get_dataset(db, dset_name)
    assert "does not exist" in str(exc_info)

    crud.create_dataset(db, schemas.DatasetCreate(name=dset_name))
    dset = crud.get_dataset(db, dset_name)
    assert dset.name == dset_name


def test_get_model(db: Session):
    with pytest.raises(crud.ModelDoesNotExistError) as exc_info:
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
    with pytest.raises(crud.ModelDoesNotExistError) as exc_info:
        crud.create_predicted_detections(db, pred_dets_create)
    assert "does not exist" in str(exc_info)

    crud.create_model(db, schemas.Model(name=model_name))

    # check this gives an error since the images haven't been added yet
    with pytest.raises(crud.ImageDoesNotExistError) as exc_info:
        crud.create_predicted_detections(db, pred_dets_create)
    assert "Image with uri" in str(exc_info)

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
    with pytest.raises(crud.ModelDoesNotExistError) as exc_info:
        crud.create_predicted_image_classifications(db, pred_clfs_create)
    assert "does not exist" in str(exc_info)

    crud.create_model(db, schemas.Model(name=model_name))

    # check this gives an error since the images haven't been added yet
    with pytest.raises(crud.ImageDoesNotExistError) as exc_info:
        crud.create_predicted_image_classifications(db, pred_clfs_create)
    assert "Image with uri" in str(exc_info)

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

    assert crud.number_of_rows(db, models.GroundTruthSegmentation) == 2
    assert crud.number_of_rows(db, models.Image) == 2
    assert crud.number_of_rows(db, models.LabeledGroundTruthSegmentation) == 2
    assert crud.number_of_rows(db, models.Label) == 1

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
    assert crud.number_of_rows(db, models.Label) == 1


def test_create_predicted_segmentations_check_area_and_delete_model(
    db: Session,
    pred_segs_create: schemas.PredictedSegmentationsCreate,
    gt_segs_create: schemas.GroundTruthSegmentationsCreate,
):
    # check this gives an error since the model hasn't been added yet
    with pytest.raises(crud.ModelDoesNotExistError) as exc_info:
        crud.create_predicted_segmentations(db, pred_segs_create)
    assert "does not exist" in str(exc_info)

    crud.create_model(db, schemas.Model(name=model_name))

    # check this gives an error since the images haven't been added yet
    with pytest.raises(crud.ImageDoesNotExistError) as exc_info:
        crud.create_predicted_segmentations(db, pred_segs_create)
    assert "Image with uri" in str(exc_info)

    # create dataset, add images, and add predictions
    crud.create_dataset(db, schemas.DatasetCreate(name=dset_name))
    crud.create_groundtruth_segmentations(db, gt_segs_create)
    crud.create_predicted_segmentations(db, pred_segs_create)

    # check db has the added predictions
    assert crud.number_of_rows(db, models.PredictedSegmentation) == 2
    assert crud.number_of_rows(db, models.LabeledPredictedSegmentation) == 2

    # grab the first one and check that the area of the raster
    # matches the area of the image
    img = crud.get_image(db, "uri1")
    seg = img.predicted_segmentations[0]
    mask = bytes_to_pil(pred_segs_create.segmentations[0].shape)
    assert ops.pred_seg_area(db, seg) == np.array(mask).sum()

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
    db: Session, poly_without_hole: schemas.PolygonWithHole
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
                    shape=[poly_without_hole],
                    image=schemas.Image(uri="uri1"),
                    labels=[schemas.Label(key="k1", value="v1")],
                )
            ],
        ),
    )

    segmentation = db.scalar(select(models.GroundTruthSegmentation))

    assert db.scalar(ST_Area(segmentation.shape)) == 45.5


def test_segmentation_area_with_hole(
    db: Session, poly_with_hole: schemas.PolygonWithHole
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
                    shape=[poly_with_hole],
                    image=schemas.Image(uri="uri1"),
                    labels=[schemas.Label(key="k1", value="v1")],
                )
            ],
        ),
    )

    segmentation = db.scalar(select(models.GroundTruthSegmentation))

    assert db.scalar(ST_Area(segmentation.shape)) == 92


def test_segmentation_area_multi_polygon(
    db: Session,
    poly_with_hole: schemas.PolygonWithHole,
    poly_without_hole: schemas.PolygonWithHole,
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
                    shape=[poly_with_hole, poly_without_hole],
                    image=schemas.Image(uri="uri1"),
                    labels=[schemas.Label(key="k1", value="v1")],
                )
            ],
        ),
    )

    segmentation = db.scalar(select(models.GroundTruthSegmentation))

    assert db.scalar(ST_Area(segmentation.shape)) == 45.5 + 92
