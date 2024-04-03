from base64 import b64encode

import pytest
from sqlalchemy.orm import Session

from valor_api import crud, enums, schemas
from valor_api.backend import models
from valor_api.backend.core.label import (
    create_labels,
    fetch_label,
    fetch_matching_labels,
    get_disjoint_keys,
    get_disjoint_labels,
    get_joint_keys,
    get_joint_labels,
    get_label_keys,
    get_labels,
)
from valor_api.crud import (
    create_dataset,
    create_groundtruth,
    create_model,
    create_prediction,
)


@pytest.fixture
def semantic_seg_gt_anns1(
    img1_gt_mask_bytes1: bytes,
) -> schemas.Annotation:
    return schemas.Annotation(
        task_type=enums.TaskType.SEMANTIC_SEGMENTATION,
        raster=schemas.Raster(mask=b64encode(img1_gt_mask_bytes1).decode()),
        labels=[
            schemas.Label(key="semsegk1", value="semsegv1"),
            schemas.Label(key="semsegk2", value="semsegv2"),
        ],
    )


@pytest.fixture
def semantic_seg_gt_anns2(
    img2_gt_mask_bytes1: bytes,
) -> schemas.Annotation:
    return schemas.Annotation(
        task_type=enums.TaskType.SEMANTIC_SEGMENTATION,
        raster=schemas.Raster(mask=b64encode(img2_gt_mask_bytes1).decode()),
        labels=[
            schemas.Label(key="semsegk2", value="semsegv2"),
            schemas.Label(key="semsegk3", value="semsegv3"),
        ],
    )


@pytest.fixture
def semantic_seg_pred_anns1(img1_gt_mask_bytes1: bytes) -> schemas.Annotation:
    return schemas.Annotation(
        task_type=enums.TaskType.SEMANTIC_SEGMENTATION,
        raster=schemas.Raster(mask=b64encode(img1_gt_mask_bytes1).decode()),
        labels=[
            schemas.Label(key="semsegk1", value="semsegv1"),
            schemas.Label(key="semsegk2", value="semsegv2"),
        ],
    )


@pytest.fixture
def semantic_seg_pred_anns2(img2_gt_mask_bytes1: bytes) -> schemas.Annotation:
    return schemas.Annotation(
        task_type=enums.TaskType.SEMANTIC_SEGMENTATION,
        raster=schemas.Raster(mask=b64encode(img2_gt_mask_bytes1).decode()),
        labels=[
            schemas.Label(key="semsegk2", value="semsegv2"),
            schemas.Label(key="semsegk3_pred", value="semsegv3_pred"),
        ],
    )


@pytest.fixture
def instance_seg_gt_anns1(
    img1_gt_mask_bytes1: bytes,
) -> schemas.Annotation:
    return schemas.Annotation(
        task_type=enums.TaskType.OBJECT_DETECTION,
        raster=schemas.Raster(mask=b64encode(img1_gt_mask_bytes1).decode()),
        labels=[
            schemas.Label(key="inssegk1", value="inssegv1"),
            schemas.Label(key="inssegk2", value="inssegv2"),
        ],
    )


@pytest.fixture
def instance_seg_gt_anns2(
    img2_gt_mask_bytes1: bytes,
) -> schemas.Annotation:
    return schemas.Annotation(
        task_type=enums.TaskType.OBJECT_DETECTION,
        raster=schemas.Raster(mask=b64encode(img2_gt_mask_bytes1).decode()),
        labels=[
            schemas.Label(key="inssegk2", value="inssegv2"),
            schemas.Label(key="inssegk3", value="inssegv3"),
        ],
    )


@pytest.fixture
def simple_labels(db: Session) -> list[schemas.Label]:
    labels = [
        schemas.Label(key="animal", value="dog"),
        schemas.Label(key="animal", value="cat"),
    ]
    create_labels(db, labels)
    return labels


@pytest.fixture
def labels_with_common_values(db: Session) -> list[schemas.Label]:
    labels = [
        schemas.Label(key="stoplight_color", value="red"),
        schemas.Label(key="stoplight_color", value="green"),
        schemas.Label(key="stoplight_color", value="yellow"),
        schemas.Label(key="car_color", value="red"),
        schemas.Label(key="car_color", value="green"),
        schemas.Label(key="car_color", value="blue"),
    ]
    create_labels(db, labels)
    return labels


@pytest.fixture
def create_dataset_model(db: Session, dataset_name: str, model_name: str):
    create_dataset(db=db, dataset=schemas.Dataset(name=dataset_name))
    create_model(db=db, model=schemas.Model(name=model_name))
    create_groundtruth(
        db=db,
        groundtruth=schemas.GroundTruth(
            datum=schemas.Datum(uid="123", dataset_name=dataset_name),
            annotations=[
                schemas.Annotation(
                    task_type=enums.TaskType.CLASSIFICATION,
                    labels=[
                        schemas.Label(key="k1", value="v1"),
                        schemas.Label(key="k1", value="v2"),
                        schemas.Label(key="k2", value="v3"),
                    ],
                )
            ],
        ),
    )
    create_prediction(
        db=db,
        prediction=schemas.Prediction(
            model_name=model_name,
            datum=schemas.Datum(uid="123", dataset_name=dataset_name),
            annotations=[
                schemas.Annotation(
                    task_type=enums.TaskType.CLASSIFICATION,
                    labels=[
                        schemas.Label(key="k1", value="v2", score=0.1),
                        schemas.Label(key="k1", value="v3", score=0.9),
                        schemas.Label(key="k3", value="v3", score=1.0),
                    ],
                )
            ],
        ),
    )


def test_fetch_label(db: Session, simple_labels: list[schemas.Label]):
    for label in simple_labels:
        fetched_label = fetch_label(db, label)
        assert fetched_label is not None
        assert fetched_label.key == label.key
        assert fetched_label.value == label.value

    # fetch label that doesnt exist
    assert fetch_label(db, schemas.Label(key="k1234", value="v1234")) is None


def test_fetch_matching_labels(db: Session, simple_labels):
    # check all exist
    db_labels = {
        (label.key, label.value)
        for label in fetch_matching_labels(db, simple_labels)
    }
    og_labels = {(label.key, label.value) for label in simple_labels}
    assert db_labels == og_labels

    # search for specific label (animal, dog)
    search = [schemas.Label(key="animal", value="dog")]
    labels = fetch_matching_labels(db, search)
    assert len(labels) == 1
    assert labels[0].key == "animal"
    assert labels[0].value == "dog"

    # search for label that doesnt exist
    search = [schemas.Label(key="animal", value="elephant")]
    assert len(fetch_matching_labels(db, search)) == 0


def test_fetch_matching_labels_from_labels_with_common_values(
    db: Session,
    labels_with_common_values,
):
    # check all exist
    db_labels = {
        (label.key, label.value)
        for label in fetch_matching_labels(db, labels_with_common_values)
    }
    og_labels = {
        (label.key, label.value) for label in labels_with_common_values
    }
    assert db_labels == og_labels

    # search for specific label (car_color, red)
    search = [schemas.Label(key="car_color", value="red")]
    labels = fetch_matching_labels(db, search)
    assert len(labels) == 1
    assert labels[0].key == "car_color"
    assert labels[0].value == "red"

    # search for label that doesnt exist
    search = [schemas.Label(key="animal", value="elephant")]
    assert len(fetch_matching_labels(db, search)) == 0

    # search for labels (car_color, red) and (stoplight_color, green)
    # validates that key-value pairings are respected.
    search = [
        schemas.Label(key="car_color", value="red"),
        schemas.Label(key="stoplight_color", value="green"),
    ]
    labels = fetch_matching_labels(db, search)
    assert {("car_color", "red"), ("stoplight_color", "green")} == {
        (label.key, label.value) for label in labels
    }
    assert len(labels) == 2


def test_create_labels_with_duplicates(db: Session):
    labels = [
        schemas.Label(key="stoplight_color", value="red"),
        schemas.Label(key="stoplight_color", value="red"),
    ]
    created_labels = create_labels(db, labels)
    assert len(db.query(models.Label).all()) == 1
    assert len(created_labels) == 2
    assert created_labels[0].id == created_labels[1].id


def test_get_labels(
    db: Session,
    create_dataset_model,
):
    assert len(db.query(models.Label).all()) == 5
    labels = get_labels(db)
    assert len(labels) == 5
    assert set(labels) == {
        schemas.Label(key="k1", value="v1"),
        schemas.Label(key="k1", value="v2"),
        schemas.Label(key="k2", value="v3"),
        schemas.Label(key="k1", value="v2"),
        schemas.Label(key="k1", value="v3"),
        schemas.Label(key="k3", value="v3"),
    }

    pred_labels = get_labels(db, ignore_groundtruths=True)
    assert len(pred_labels) == 3
    assert set(pred_labels) == {
        schemas.Label(key="k1", value="v2"),
        schemas.Label(key="k1", value="v3"),
        schemas.Label(key="k3", value="v3"),
    }

    gt_labels = get_labels(db, ignore_predictions=True)
    assert len(gt_labels) == 3
    assert set(gt_labels) == {
        schemas.Label(key="k1", value="v1"),
        schemas.Label(key="k1", value="v2"),
        schemas.Label(key="k2", value="v3"),
    }


def test_get_labels_filtered(
    db: Session,
    create_dataset_model,
):
    assert len(db.query(models.Label).all()) == 5

    filters = schemas.Filter(label_keys=["k1"])

    labels = get_labels(db, filters=filters)
    assert len(labels) == 3
    assert set(labels) == {
        schemas.Label(key="k1", value="v1"),
        schemas.Label(key="k1", value="v2"),
        schemas.Label(key="k1", value="v3"),
    }
    pred_labels = get_labels(db, filters=filters, ignore_groundtruths=True)
    assert len(pred_labels) == 2
    assert set(pred_labels) == {
        schemas.Label(key="k1", value="v2"),
        schemas.Label(key="k1", value="v3"),
    }

    gt_labels = get_labels(db, filters=filters, ignore_predictions=True)
    assert len(gt_labels) == 2
    assert set(gt_labels) == {
        schemas.Label(key="k1", value="v1"),
        schemas.Label(key="k1", value="v2"),
    }


def test_get_label_keys(
    db: Session,
    create_dataset_model,
):
    assert len(db.query(models.Label).all()) == 5

    labels = get_label_keys(db)
    assert len(labels) == 3
    assert set(labels) == {"k1", "k2", "k3"}

    pred_labels = get_label_keys(db, ignore_groundtruths=True)
    assert len(pred_labels) == 2
    assert set(pred_labels) == {"k1", "k3"}

    gt_labels = get_label_keys(db, ignore_predictions=True)
    assert len(gt_labels) == 2
    assert set(gt_labels) == {"k1", "k2"}


def test_get_label_keys_filtered(
    db: Session,
    create_dataset_model,
):
    assert len(db.query(models.Label).all()) == 5

    filters = schemas.Filter(label_keys=["k1"])

    labels = get_label_keys(db, filters=filters)
    assert len(labels) == 1
    assert set(labels) == {"k1"}

    pred_labels = get_label_keys(db, filters=filters, ignore_groundtruths=True)
    assert len(pred_labels) == 1
    assert set(pred_labels) == {"k1"}

    gt_labels = get_label_keys(db, filters=filters, ignore_predictions=True)
    assert len(gt_labels) == 1
    assert set(gt_labels) == {"k1"}


def test_get_joint_labels(
    db: Session,
    dataset_name: str,
    model_name: str,
    create_dataset_model,
):
    labels = get_joint_labels(
        db=db,
        lhs=schemas.Filter(
            dataset_names=[dataset_name],
            task_types=[enums.TaskType.CLASSIFICATION],
            require_box=False,
            require_polygon=False,
            require_raster=False,
        ),
        rhs=schemas.Filter(
            model_names=[model_name],
            task_types=[enums.TaskType.CLASSIFICATION],
            require_box=False,
            require_polygon=False,
            require_raster=False,
        ),
    )
    assert len(labels) == 1
    assert labels[0] == schemas.Label(
        key="k1",
        value="v2",
    )


def test_get_joint_keys(
    db: Session,
    dataset_name: str,
    model_name: str,
    create_dataset_model,
):
    keys = get_joint_keys(
        db=db,
        lhs=schemas.Filter(
            dataset_names=[dataset_name],
            task_types=[enums.TaskType.CLASSIFICATION],
        ),
        rhs=schemas.Filter(
            model_names=[model_name],
            task_types=[enums.TaskType.CLASSIFICATION],
        ),
    )
    assert len(keys) == 1
    assert set(keys) == {"k1"}


def test_get_disjoint_labels(
    db: Session,
    dataset_name: str,
    model_name: str,
    create_dataset_model,
):
    ds_unique, md_unique = get_disjoint_labels(
        db=db,
        lhs=schemas.Filter(
            dataset_names=[dataset_name],
            task_types=[enums.TaskType.CLASSIFICATION],
            require_box=False,
            require_polygon=False,
            require_raster=False,
        ),
        rhs=schemas.Filter(
            model_names=[model_name],
            task_types=[enums.TaskType.CLASSIFICATION],
            require_box=False,
            require_polygon=False,
            require_raster=False,
        ),
    )
    assert len(ds_unique) == 2
    assert set(ds_unique) == {
        schemas.Label(key="k1", value="v1"),
        schemas.Label(key="k2", value="v3"),
    }

    assert len(md_unique) == 2
    assert set(md_unique) == {
        schemas.Label(key="k1", value="v3"),
        schemas.Label(key="k3", value="v3"),
    }


def test_get_disjoint_keys(
    db: Session,
    dataset_name: str,
    model_name: str,
    create_dataset_model,
):
    ds_unique, md_unique = get_disjoint_keys(
        db=db,
        lhs=schemas.Filter(
            dataset_names=[dataset_name],
            task_types=[enums.TaskType.CLASSIFICATION],
        ),
        rhs=schemas.Filter(
            model_names=[model_name],
            task_types=[enums.TaskType.CLASSIFICATION],
        ),
    )
    assert len(ds_unique) == 1
    assert set(ds_unique) == {"k2"}

    assert len(md_unique) == 1
    assert set(md_unique) == {"k3"}


def test_label_functions(
    db: Session,
    dataset_name: str,
    model_name: str,
    img1: schemas.Datum,
    img2: schemas.Datum,
    semantic_seg_gt_anns1: schemas.Annotation,
    semantic_seg_gt_anns2: schemas.Annotation,
    semantic_seg_pred_anns1: schemas.Annotation,
    semantic_seg_pred_anns2: schemas.Annotation,
    instance_seg_gt_anns1: schemas.Annotation,
    instance_seg_gt_anns2: schemas.Annotation,
):
    """Tests the label query methods"""
    crud.create_dataset(db=db, dataset=schemas.Dataset(name=dataset_name))
    crud.create_model(db=db, model=schemas.Model(name=model_name))

    datum1 = img1
    datum2 = img2

    gts = [
        schemas.GroundTruth(
            datum=datum1,
            annotations=[
                semantic_seg_gt_anns1,
                instance_seg_gt_anns1,
            ],
        ),
        schemas.GroundTruth(
            datum=datum2,
            annotations=[
                semantic_seg_gt_anns2,
                instance_seg_gt_anns2,
            ],
        ),
    ]
    pds = [
        schemas.Prediction(
            model_name=model_name,
            datum=datum1,
            annotations=[
                semantic_seg_pred_anns1,
            ],
        ),
        schemas.Prediction(
            model_name=model_name,
            datum=datum2,
            annotations=[
                semantic_seg_pred_anns2,
            ],
        ),
    ]

    for gt in gts:
        crud.create_groundtruth(db=db, groundtruth=gt)

    for pred in pds:
        crud.create_prediction(db=db, prediction=pred)

    assert get_label_keys(
        db,
        schemas.Filter(
            dataset_names=[dataset_name],
            task_types=[enums.TaskType.SEMANTIC_SEGMENTATION],
        ),
        ignore_predictions=True,
    ) == {"semsegk1", "semsegk2", "semsegk3"}

    assert get_labels(
        db,
        schemas.Filter(
            dataset_names=[dataset_name],
            task_types=[enums.TaskType.SEMANTIC_SEGMENTATION],
            require_raster=True,
        ),
        ignore_predictions=True,
    ) == {
        schemas.Label(key="semsegk1", value="semsegv1"),
        schemas.Label(key="semsegk2", value="semsegv2"),
        schemas.Label(key="semsegk3", value="semsegv3"),
    }
    assert (
        get_labels(
            db,
            schemas.Filter(
                dataset_names=[dataset_name],
                task_types=[enums.TaskType.SEMANTIC_SEGMENTATION],
                require_polygon=True,
            ),
            ignore_predictions=True,
        )
        == set()
    )

    assert get_label_keys(
        db,
        schemas.Filter(
            model_names=[model_name],
            dataset_names=[dataset_name],
            task_types=[enums.TaskType.SEMANTIC_SEGMENTATION],
        ),
        ignore_groundtruths=True,
    ) == {"semsegk1", "semsegk2", "semsegk3_pred"}

    assert get_labels(
        db,
        schemas.Filter(
            model_names=[model_name],
            dataset_names=[dataset_name],
            require_raster=True,
            task_types=[enums.TaskType.SEMANTIC_SEGMENTATION],
        ),
        ignore_groundtruths=True,
    ) == {
        schemas.Label(key="semsegk1", value="semsegv1"),
        schemas.Label(key="semsegk2", value="semsegv2"),
        schemas.Label(key="semsegk3_pred", value="semsegv3_pred"),
    }

    assert (
        get_labels(
            db,
            schemas.Filter(
                model_names=[model_name],
                dataset_names=[dataset_name],
                require_polygon=True,
                task_types=[enums.TaskType.SEMANTIC_SEGMENTATION],
            ),
            ignore_groundtruths=True,
        )
        == set()
    )

    assert (
        get_label_keys(
            db,
            schemas.Filter(
                dataset_names=[dataset_name],
                task_types=[enums.TaskType.CLASSIFICATION],
            ),
            ignore_predictions=True,
        )
        == set()
    )
    assert (
        get_labels(
            db,
            schemas.Filter(
                model_names=[model_name],
                dataset_names=[dataset_name],
                task_types=[enums.TaskType.CLASSIFICATION],
            ),
            ignore_groundtruths=True,
        )
        == set()
    )

    assert get_label_keys(
        db,
        schemas.Filter(
            dataset_names=[dataset_name],
            task_types=[enums.TaskType.OBJECT_DETECTION],
        ),
        ignore_predictions=True,
    ) == {"inssegk1", "inssegk2", "inssegk3"}

    assert get_labels(
        db,
        schemas.Filter(
            dataset_names=[dataset_name],
            require_raster=True,
            task_types=[enums.TaskType.OBJECT_DETECTION],
        ),
        ignore_predictions=True,
    ) == {
        schemas.Label(key="inssegk1", value="inssegv1"),
        schemas.Label(key="inssegk2", value="inssegv2"),
        schemas.Label(key="inssegk3", value="inssegv3"),
    }
    assert get_labels(
        db,
        schemas.Filter(
            dataset_names=[dataset_name],
            require_raster=True,
            task_types=[
                enums.TaskType.OBJECT_DETECTION,
                enums.TaskType.SEMANTIC_SEGMENTATION,
            ],
        ),
        ignore_predictions=True,
    ) == {
        schemas.Label(key="inssegk1", value="inssegv1"),
        schemas.Label(key="inssegk2", value="inssegv2"),
        schemas.Label(key="inssegk3", value="inssegv3"),
        schemas.Label(key="semsegk1", value="semsegv1"),
        schemas.Label(key="semsegk2", value="semsegv2"),
        schemas.Label(key="semsegk3", value="semsegv3"),
    }

    assert get_label_keys(
        db,
        schemas.Filter(
            dataset_names=[dataset_name],
            task_types=[enums.TaskType.SEMANTIC_SEGMENTATION],
        ),
        ignore_predictions=True,
    ) == {"semsegk1", "semsegk2", "semsegk3"}

    assert get_label_keys(
        db,
        schemas.Filter(
            model_names=[model_name],
            dataset_names=[dataset_name],
            task_types=[enums.TaskType.SEMANTIC_SEGMENTATION],
        ),
        ignore_groundtruths=True,
    ) == {"semsegk1", "semsegk2", "semsegk3_pred"}

    assert (
        get_labels(
            db,
            schemas.Filter(
                model_names=[model_name],
                dataset_names=[dataset_name],
                task_types=[enums.TaskType.OBJECT_DETECTION],
            ),
            ignore_groundtruths=True,
        )
        == set()
    )

    assert get_labels(
        db,
        schemas.Filter(
            model_names=[model_name],
            dataset_names=[dataset_name],
            require_raster=True,
            task_types=[
                enums.TaskType.SEMANTIC_SEGMENTATION,
                enums.TaskType.OBJECT_DETECTION,
            ],
        ),
        ignore_groundtruths=True,
    ) == {
        schemas.Label(key="semsegk1", value="semsegv1"),
        schemas.Label(key="semsegk2", value="semsegv2"),
        schemas.Label(key="semsegk3_pred", value="semsegv3_pred"),
    }
    assert (
        get_labels(
            db,
            schemas.Filter(
                model_names=[model_name],
                dataset_names=[dataset_name],
                require_raster=True,
                task_types=[enums.TaskType.OBJECT_DETECTION],
            ),
            ignore_groundtruths=True,
        )
        == set()
    )
