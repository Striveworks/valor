from base64 import b64encode

import pytest
from sqlalchemy.orm import Session

from valor_api import crud, enums, schemas
from valor_api.backend import models
from valor_api.backend.core.label import (
    create_labels,
    fetch_label,
    get_disjoint_keys,
    get_disjoint_labels,
    get_joint_keys,
    get_joint_labels,
    get_label_keys,
    get_labels,
    get_paginated_labels,
)
from valor_api.crud import (
    create_dataset,
    create_groundtruths,
    create_model,
    create_predictions,
)


@pytest.fixture
def semantic_seg_gt_anns1(
    img1_gt_mask_bytes1: bytes,
) -> schemas.Annotation:
    return schemas.Annotation(
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
        raster=schemas.Raster(mask=b64encode(img2_gt_mask_bytes1).decode()),
        labels=[
            schemas.Label(key="semsegk2", value="semsegv2"),
            schemas.Label(key="semsegk3", value="semsegv3"),
        ],
    )


@pytest.fixture
def semantic_seg_pred_anns1(img1_gt_mask_bytes1: bytes) -> schemas.Annotation:
    return schemas.Annotation(
        raster=schemas.Raster(mask=b64encode(img1_gt_mask_bytes1).decode()),
        labels=[
            schemas.Label(key="semsegk1", value="semsegv1"),
            schemas.Label(key="semsegk2", value="semsegv2"),
        ],
    )


@pytest.fixture
def semantic_seg_pred_anns2(img2_gt_mask_bytes1: bytes) -> schemas.Annotation:
    return schemas.Annotation(
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
        raster=schemas.Raster(mask=b64encode(img1_gt_mask_bytes1).decode()),
        labels=[
            schemas.Label(key="inssegk1", value="inssegv1"),
            schemas.Label(key="inssegk2", value="inssegv2"),
        ],
        is_instance=True,
    )


@pytest.fixture
def instance_seg_gt_anns2(
    img2_gt_mask_bytes1: bytes,
) -> schemas.Annotation:
    return schemas.Annotation(
        raster=schemas.Raster(mask=b64encode(img2_gt_mask_bytes1).decode()),
        labels=[
            schemas.Label(key="inssegk2", value="inssegv2"),
            schemas.Label(key="inssegk3", value="inssegv3"),
        ],
        is_instance=True,
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
    create_groundtruths(
        db=db,
        groundtruths=[
            schemas.GroundTruth(
                dataset_name=dataset_name,
                datum=schemas.Datum(uid="123"),
                annotations=[
                    schemas.Annotation(
                        labels=[
                            schemas.Label(key="k1", value="v1"),
                            schemas.Label(key="k1", value="v2"),
                            schemas.Label(key="k2", value="v3"),
                        ],
                    )
                ],
            )
        ],
    )
    create_predictions(
        db=db,
        predictions=[
            schemas.Prediction(
                dataset_name=dataset_name,
                model_name=model_name,
                datum=schemas.Datum(uid="123"),
                annotations=[
                    schemas.Annotation(
                        labels=[
                            schemas.Label(key="k1", value="v2", score=0.1),
                            schemas.Label(key="k1", value="v3", score=0.9),
                            schemas.Label(key="k3", value="v3", score=1.0),
                        ],
                    )
                ],
            )
        ],
    )


def test_fetch_label(db: Session, simple_labels: list[schemas.Label]):
    for label in simple_labels:
        fetched_label = fetch_label(db, label)
        assert fetched_label is not None
        assert fetched_label.key == label.key
        assert fetched_label.value == label.value

    # fetch label that doesnt exist
    assert fetch_label(db, schemas.Label(key="k1234", value="v1234")) is None


def test_create_labels_with_duplicates(db: Session):
    labels = [
        schemas.Label(key="stoplight_color", value="red"),
        schemas.Label(key="stoplight_color", value="red"),
    ]
    created_labels = create_labels(db, labels)
    assert len(db.query(models.Label).all()) == 1
    assert len(created_labels) == 1
    assert ("stoplight_color", "red") in created_labels


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


def test_get_paginated_labels(
    db: Session,
    create_dataset_model,
):
    assert len(db.query(models.Label).all()) == 5
    labels, headers = get_paginated_labels(db)
    assert len(labels) == 5
    assert set(labels) == {
        schemas.Label(key="k1", value="v1"),
        schemas.Label(key="k2", value="v3"),
        schemas.Label(key="k1", value="v2"),
        schemas.Label(key="k1", value="v3"),
        schemas.Label(key="k3", value="v3"),
    }
    assert headers == {"content-range": "items 0-4/5"}

    # test that we can reconstitute the full set using paginated calls

    first_set, header = get_paginated_labels(db, offset=1, limit=2)
    assert len(first_set) == 2
    assert header == {"content-range": "items 1-2/5"}

    second_set, header = get_paginated_labels(db, offset=0, limit=1)
    assert len(second_set) == 1
    assert header == {"content-range": "items 0-0/5"}

    third_set, header = get_paginated_labels(db, offset=3, limit=2)
    assert len(third_set) == 2
    assert header == {"content-range": "items 3-4/5"}

    combined_set = first_set | second_set | third_set

    assert combined_set == {
        schemas.Label(key="k1", value="v1"),
        schemas.Label(key="k2", value="v3"),
        schemas.Label(key="k1", value="v2"),
        schemas.Label(key="k1", value="v3"),
        schemas.Label(key="k3", value="v3"),
    }

    # test that we get an error if the offset is set too high
    with pytest.raises(ValueError):
        _ = get_paginated_labels(db, offset=100, limit=1)


def test_get_labels_filtered(
    db: Session,
    create_dataset_model,
):
    assert len(db.query(models.Label).all()) == 5

    filters = schemas.Filter(
        labels=schemas.Condition(
            lhs=schemas.Symbol.LABEL_KEY,
            rhs=schemas.Value.infer("k1"),
            op=schemas.FilterOperator.EQ,
        ),
    )

    labels = get_labels(db, filters=filters)
    assert len(labels) == 3
    assert labels == {
        schemas.Label(key="k1", value="v1"),
        schemas.Label(key="k1", value="v2"),
        schemas.Label(key="k1", value="v3"),
    }
    pred_labels = get_labels(db, filters=filters, ignore_groundtruths=True)
    assert len(pred_labels) == 2
    assert pred_labels == {
        schemas.Label(key="k1", value="v2"),
        schemas.Label(key="k1", value="v3"),
    }

    gt_labels = get_labels(db, filters=filters, ignore_predictions=True)
    assert len(gt_labels) == 2
    assert gt_labels == {
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

    filters = schemas.Filter(
        labels=schemas.Condition(
            lhs=schemas.Symbol.LABEL_KEY,
            rhs=schemas.Value.infer("k1"),
            op=schemas.FilterOperator.EQ,
        ),
    )

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
            labels=schemas.LogicalFunction(
                args=[
                    schemas.Condition(
                        lhs=schemas.Symbol.DATASET_NAME,
                        rhs=schemas.Value.infer(dataset_name),
                        op=schemas.FilterOperator.EQ,
                    ),
                    schemas.Condition(
                        lhs=schemas.Symbol.TASK_TYPE,
                        rhs=schemas.Value(
                            type=schemas.SupportedType.TASK_TYPE,
                            value=enums.TaskType.CLASSIFICATION,
                        ),
                        op=schemas.FilterOperator.CONTAINS,
                    ),
                    schemas.Condition(
                        lhs=schemas.Symbol.BOX,
                        op=schemas.FilterOperator.ISNULL,
                    ),
                    schemas.Condition(
                        lhs=schemas.Symbol.POLYGON,
                        op=schemas.FilterOperator.ISNULL,
                    ),
                    schemas.Condition(
                        lhs=schemas.Symbol.RASTER,
                        op=schemas.FilterOperator.ISNULL,
                    ),
                ],
                op=schemas.LogicalOperator.AND
            )
        ),
        rhs=schemas.Filter(
            labels=schemas.LogicalFunction(
                args=[
                    schemas.Condition(
                        lhs=schemas.Symbol.MODEL_NAME,
                        rhs=schemas.Value.infer(model_name),
                        op=schemas.FilterOperator.EQ,
                    ),
                    schemas.Condition(
                        lhs=schemas.Symbol.TASK_TYPE,
                        rhs=schemas.Value(
                            type=schemas.SupportedType.TASK_TYPE,
                            value=enums.TaskType.CLASSIFICATION,
                        ),
                        op=schemas.FilterOperator.CONTAINS,
                    ),
                    schemas.Condition(
                        lhs=schemas.Symbol.BOX,
                        op=schemas.FilterOperator.ISNULL,
                    ),
                    schemas.Condition(
                        lhs=schemas.Symbol.POLYGON,
                        op=schemas.FilterOperator.ISNULL,
                    ),
                    schemas.Condition(
                        lhs=schemas.Symbol.RASTER,
                        op=schemas.FilterOperator.ISNULL,
                    ),
                ],
                op=schemas.LogicalOperator.AND
            )
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
            labels=schemas.LogicalFunction(
                args=[
                    schemas.Condition(
                        lhs=schemas.Symbol.DATASET_NAME,
                        rhs=schemas.Value.infer(dataset_name),
                        op=schemas.FilterOperator.EQ,
                    ),
                    schemas.Condition(
                        lhs=schemas.Symbol.TASK_TYPE,
                        rhs=schemas.Value(
                            type=schemas.SupportedType.TASK_TYPE,
                            value=enums.TaskType.CLASSIFICATION,
                        ),
                        op=schemas.FilterOperator.CONTAINS,
                    ),
                ],
                op=schemas.LogicalOperator.AND
            )
        ),
        rhs=schemas.Filter(
            labels=schemas.LogicalFunction(
                args=[
                    schemas.Condition(
                        lhs=schemas.Symbol.MODEL_NAME,
                        rhs=schemas.Value.infer(model_name),
                        op=schemas.FilterOperator.EQ,
                    ),
                    schemas.Condition(
                        lhs=schemas.Symbol.TASK_TYPE,
                        rhs=schemas.Value(
                            type=schemas.SupportedType.TASK_TYPE,
                            value=enums.TaskType.CLASSIFICATION,
                        ),
                        op=schemas.FilterOperator.CONTAINS,
                    ),
                ],
                op=schemas.LogicalOperator.AND
            )
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
            labels=schemas.LogicalFunction(
                args=[
                    schemas.Condition(
                        lhs=schemas.Symbol.DATASET_NAME,
                        rhs=schemas.Value.infer(dataset_name),
                        op=schemas.FilterOperator.EQ,
                    ),
                    schemas.Condition(
                        lhs=schemas.Symbol.TASK_TYPE,
                        rhs=schemas.Value(
                            type=schemas.SupportedType.TASK_TYPE,
                            value=enums.TaskType.CLASSIFICATION,
                        ),
                        op=schemas.FilterOperator.CONTAINS,
                    ),
                    schemas.Condition(
                        lhs=schemas.Symbol.BOX,
                        op=schemas.FilterOperator.ISNULL,
                    ),
                    schemas.Condition(
                        lhs=schemas.Symbol.POLYGON,
                        op=schemas.FilterOperator.ISNULL,
                    ),
                    schemas.Condition(
                        lhs=schemas.Symbol.RASTER,
                        op=schemas.FilterOperator.ISNULL,
                    ),
                ],
                op=schemas.LogicalOperator.AND
            )
        ),
        rhs=schemas.Filter(
            labels=schemas.LogicalFunction(
                args=[
                    schemas.Condition(
                        lhs=schemas.Symbol.MODEL_NAME,
                        rhs=schemas.Value.infer(model_name),
                        op=schemas.FilterOperator.EQ,
                    ),
                    schemas.Condition(
                        lhs=schemas.Symbol.TASK_TYPE,
                        rhs=schemas.Value(
                            type=schemas.SupportedType.TASK_TYPE,
                            value=enums.TaskType.CLASSIFICATION,
                        ),
                        op=schemas.FilterOperator.CONTAINS,
                    ),
                    schemas.Condition(
                        lhs=schemas.Symbol.BOX,
                        op=schemas.FilterOperator.ISNULL,
                    ),
                    schemas.Condition(
                        lhs=schemas.Symbol.POLYGON,
                        op=schemas.FilterOperator.ISNULL,
                    ),
                    schemas.Condition(
                        lhs=schemas.Symbol.RASTER,
                        op=schemas.FilterOperator.ISNULL,
                    ),
                ],
                op=schemas.LogicalOperator.AND
            )
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
            labels=schemas.LogicalFunction(
                args=[
                    schemas.Condition(
                        lhs=schemas.Symbol.MODEL_NAME,
                        rhs=schemas.Value.infer(model_name),
                        op=schemas.FilterOperator.EQ,
                    ),
                    schemas.Condition(
                        lhs=schemas.Symbol.TASK_TYPE,
                        rhs=schemas.Value(
                            type=schemas.SupportedType.TASK_TYPE,
                            value=enums.TaskType.CLASSIFICATION,
                        ),
                        op=schemas.FilterOperator.CONTAINS,
                    ),
                ],
                op=schemas.LogicalOperator.AND
            )
        ),
        rhs=schemas.Filter(
            labels=schemas.LogicalFunction(
                args=[
                    schemas.Condition(
                        lhs=schemas.Symbol.MODEL_NAME,
                        rhs=schemas.Value.infer(model_name),
                        op=schemas.FilterOperator.EQ,
                    ),
                    schemas.Condition(
                        lhs=schemas.Symbol.TASK_TYPE,
                        rhs=schemas.Value(
                            type=schemas.SupportedType.TASK_TYPE,
                            value=enums.TaskType.CLASSIFICATION,
                        ),
                        op=schemas.FilterOperator.CONTAINS,
                    ),
                ],
                op=schemas.LogicalOperator.AND
            )
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
            dataset_name=dataset_name,
            datum=datum1,
            annotations=[
                semantic_seg_gt_anns1,
                instance_seg_gt_anns1,
            ],
        ),
        schemas.GroundTruth(
            dataset_name=dataset_name,
            datum=datum2,
            annotations=[
                semantic_seg_gt_anns2,
                instance_seg_gt_anns2,
            ],
        ),
    ]
    pds = [
        schemas.Prediction(
            dataset_name=dataset_name,
            model_name=model_name,
            datum=datum1,
            annotations=[
                semantic_seg_pred_anns1,
            ],
        ),
        schemas.Prediction(
            dataset_name=dataset_name,
            model_name=model_name,
            datum=datum2,
            annotations=[
                semantic_seg_pred_anns2,
            ],
        ),
    ]

    for gt in gts:
        crud.create_groundtruths(db=db, groundtruths=[gt])

    for pred in pds:
        crud.create_predictions(db=db, predictions=[pred])

    assert get_label_keys(
        db,
        schemas.Filter(
            labels=schemas.LogicalFunction(
                args=[
                    schemas.Condition(
                        lhs=schemas.Symbol.DATASET_NAME,
                        rhs=schemas.Value.infer(dataset_name),
                        op=schemas.FilterOperator.EQ,
                    ),
                    schemas.Condition(
                        lhs=schemas.Symbol.TASK_TYPE,
                        rhs=schemas.Value(
                            type=schemas.SupportedType.TASK_TYPE,
                            value=enums.TaskType.SEMANTIC_SEGMENTATION,
                        ),
                        op=schemas.FilterOperator.CONTAINS,
                    ),
                ],
                op=schemas.LogicalOperator.AND
            )
        ),
        ignore_predictions=True,
    ) == {"semsegk1", "semsegk2", "semsegk3"}

    assert get_labels(
        db,
        schemas.Filter(
            labels=schemas.LogicalFunction(
                args=[
                    schemas.Condition(
                        lhs=schemas.Symbol.DATASET_NAME,
                        rhs=schemas.Value.infer(dataset_name),
                        op=schemas.FilterOperator.EQ,
                    ),
                    schemas.Condition(
                        lhs=schemas.Symbol.TASK_TYPE,
                        rhs=schemas.Value(
                            type=schemas.SupportedType.TASK_TYPE,
                            value=enums.TaskType.SEMANTIC_SEGMENTATION,
                        ),
                        op=schemas.FilterOperator.CONTAINS,
                    ),
                    schemas.Condition(
                        lhs=schemas.Symbol.RASTER,
                        op=schemas.FilterOperator.ISNOTNULL,
                    ),
                ],
                op=schemas.LogicalOperator.AND
            )
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
                labels=schemas.LogicalFunction(
                    args=[
                        schemas.Condition(
                            lhs=schemas.Symbol.DATASET_NAME,
                            rhs=schemas.Value.infer(dataset_name),
                            op=schemas.FilterOperator.EQ,
                        ),
                        schemas.Condition(
                            lhs=schemas.Symbol.TASK_TYPE,
                            rhs=schemas.Value(
                                type=schemas.SupportedType.TASK_TYPE,
                                value=enums.TaskType.SEMANTIC_SEGMENTATION,
                            ),
                            op=schemas.FilterOperator.CONTAINS,
                        ),
                        schemas.Condition(
                            lhs=schemas.Symbol.POLYGON,
                            op=schemas.FilterOperator.ISNOTNULL,
                        ),
                    ],
                    op=schemas.LogicalOperator.AND
                )
            ),
            ignore_predictions=True,
        )
        == set()
    )

    assert get_label_keys(
        db,
        schemas.Filter(
            labels=schemas.LogicalFunction(
                args=[
                    schemas.Condition(
                        lhs=schemas.Symbol.DATASET_NAME,
                        rhs=schemas.Value.infer(dataset_name),
                        op=schemas.FilterOperator.EQ,
                    ),
                    schemas.Condition(
                        lhs=schemas.Symbol.MODEL_NAME,
                        rhs=schemas.Value.infer(model_name),
                        op=schemas.FilterOperator.EQ,
                    ),
                    schemas.Condition(
                        lhs=schemas.Symbol.TASK_TYPE,
                        rhs=schemas.Value(
                            type=schemas.SupportedType.TASK_TYPE,
                            value=enums.TaskType.SEMANTIC_SEGMENTATION,
                        ),
                        op=schemas.FilterOperator.CONTAINS,
                    ),
                ],
                op=schemas.LogicalOperator.AND
            )
        ),
        ignore_groundtruths=True,
    ) == {"semsegk1", "semsegk2", "semsegk3_pred"}

    assert get_labels(
        db,
        schemas.Filter(
            labels=schemas.LogicalFunction(
                args=[
                    schemas.Condition(
                        lhs=schemas.Symbol.DATASET_NAME,
                        rhs=schemas.Value.infer(dataset_name),
                        op=schemas.FilterOperator.EQ,
                    ),
                    schemas.Condition(
                        lhs=schemas.Symbol.MODEL_NAME,
                        rhs=schemas.Value.infer(model_name),
                        op=schemas.FilterOperator.EQ,
                    ),
                    schemas.Condition(
                        lhs=schemas.Symbol.TASK_TYPE,
                        rhs=schemas.Value(
                            type=schemas.SupportedType.TASK_TYPE,
                            value=enums.TaskType.SEMANTIC_SEGMENTATION,
                        ),
                        op=schemas.FilterOperator.CONTAINS,
                    ),
                    schemas.Condition(
                        lhs=schemas.Symbol.RASTER,
                        op=schemas.FilterOperator.ISNOTNULL,
                    ),
                ],
                op=schemas.LogicalOperator.AND
            )
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
                labels=schemas.LogicalFunction(
                    args=[
                        schemas.Condition(
                            lhs=schemas.Symbol.DATASET_NAME,
                            rhs=schemas.Value.infer(dataset_name),
                            op=schemas.FilterOperator.EQ,
                        ),
                        schemas.Condition(
                            lhs=schemas.Symbol.MODEL_NAME,
                            rhs=schemas.Value.infer(model_name),
                            op=schemas.FilterOperator.EQ,
                        ),
                        schemas.Condition(
                            lhs=schemas.Symbol.TASK_TYPE,
                            rhs=schemas.Value(
                                type=schemas.SupportedType.TASK_TYPE,
                                value=enums.TaskType.SEMANTIC_SEGMENTATION,
                            ),
                            op=schemas.FilterOperator.CONTAINS,
                        ),
                        schemas.Condition(
                            lhs=schemas.Symbol.POLYGON,
                            op=schemas.FilterOperator.ISNOTNULL,
                        ),
                    ],
                    op=schemas.LogicalOperator.AND
                )
            ),
            ignore_groundtruths=True,
        )
        == set()
    )

    assert (
        get_label_keys(
            db,
            schemas.Filter(
                labels=schemas.LogicalFunction(
                    args=[
                        schemas.Condition(
                            lhs=schemas.Symbol.DATASET_NAME,
                            rhs=schemas.Value.infer(dataset_name),
                            op=schemas.FilterOperator.EQ,
                        ),
                        schemas.Condition(
                            lhs=schemas.Symbol.TASK_TYPE,
                            rhs=schemas.Value(
                                type=schemas.SupportedType.TASK_TYPE,
                                value=enums.TaskType.CLASSIFICATION,
                            ),
                            op=schemas.FilterOperator.CONTAINS,
                        ),
                    ],
                    op=schemas.LogicalOperator.AND
                )
            ),
            ignore_predictions=True,
        )
        == set()
    )
    assert (
        get_labels(
            db,
            schemas.Filter(
                labels=schemas.LogicalFunction(
                    args=[
                        schemas.Condition(
                            lhs=schemas.Symbol.DATASET_NAME,
                            rhs=schemas.Value.infer(dataset_name),
                            op=schemas.FilterOperator.EQ,
                        ),
                        schemas.Condition(
                            lhs=schemas.Symbol.MODEL_NAME,
                            rhs=schemas.Value.infer(model_name),
                            op=schemas.FilterOperator.EQ,
                        ),
                        schemas.Condition(
                            lhs=schemas.Symbol.TASK_TYPE,
                            rhs=schemas.Value(
                                type=schemas.SupportedType.TASK_TYPE,
                                value=enums.TaskType.CLASSIFICATION,
                            ),
                            op=schemas.FilterOperator.CONTAINS,
                        ),
                    ],
                    op=schemas.LogicalOperator.AND
                )
            ),
            ignore_groundtruths=True,
        )
        == set()
    )

    assert get_label_keys(
        db,
        schemas.Filter(
            labels=schemas.LogicalFunction(
                args=[
                    schemas.Condition(
                        lhs=schemas.Symbol.DATASET_NAME,
                        rhs=schemas.Value.infer(dataset_name),
                        op=schemas.FilterOperator.EQ,
                    ),
                    schemas.Condition(
                        lhs=schemas.Symbol.TASK_TYPE,
                        rhs=schemas.Value(
                            type=schemas.SupportedType.TASK_TYPE,
                            value=enums.TaskType.OBJECT_DETECTION,
                        ),
                        op=schemas.FilterOperator.CONTAINS,
                    ),
                ],
                op=schemas.LogicalOperator.AND
            )
        ),
        ignore_predictions=True,
    ) == {"inssegk1", "inssegk2", "inssegk3"}

    assert get_labels(
        db,
        schemas.Filter(
            labels=schemas.LogicalFunction(
                args=[
                    schemas.Condition(
                        lhs=schemas.Symbol.DATASET_NAME,
                        rhs=schemas.Value.infer(dataset_name),
                        op=schemas.FilterOperator.EQ,
                    ),
                    schemas.Condition(
                        lhs=schemas.Symbol.TASK_TYPE,
                        rhs=schemas.Value(
                            type=schemas.SupportedType.TASK_TYPE,
                            value=enums.TaskType.OBJECT_DETECTION,
                        ),
                        op=schemas.FilterOperator.CONTAINS,
                    ),
                    schemas.Condition(
                        lhs=schemas.Symbol.RASTER,
                        op=schemas.FilterOperator.ISNOTNULL,
                    ),
                ],
                op=schemas.LogicalOperator.AND
            )
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
            labels=schemas.LogicalFunction(
                args=[
                    schemas.Condition(
                        lhs=schemas.Symbol.DATASET_NAME,
                        rhs=schemas.Value.infer(dataset_name),
                        op=schemas.FilterOperator.EQ,
                    ),
                    schemas.LogicalFunction(
                        args=[
                            schemas.Condition(
                                lhs=schemas.Symbol.TASK_TYPE,
                                rhs=schemas.Value(
                                    type=schemas.SupportedType.TASK_TYPE,
                                    value=enums.TaskType.OBJECT_DETECTION,
                                ),
                                op=schemas.FilterOperator.CONTAINS,
                            ),
                            schemas.Condition(
                                lhs=schemas.Symbol.TASK_TYPE,
                                rhs=schemas.Value(
                                    type=schemas.SupportedType.TASK_TYPE,
                                    value=enums.TaskType.SEMANTIC_SEGMENTATION,
                                ),
                                op=schemas.FilterOperator.CONTAINS,
                            ),
                        ],
                        op=schemas.LogicalOperator.OR,
                    ),
                    schemas.Condition(
                        lhs=schemas.Symbol.RASTER,
                        op=schemas.FilterOperator.ISNOTNULL,
                    ),
                ],
                op=schemas.LogicalOperator.AND
            )
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
            labels=schemas.LogicalFunction(
                args=[
                    schemas.Condition(
                        lhs=schemas.Symbol.DATASET_NAME,
                        rhs=schemas.Value.infer(dataset_name),
                        op=schemas.FilterOperator.EQ,
                    ),
                    schemas.Condition(
                        lhs=schemas.Symbol.TASK_TYPE,
                        rhs=schemas.Value(
                            type=schemas.SupportedType.TASK_TYPE,
                            value=enums.TaskType.SEMANTIC_SEGMENTATION,
                        ),
                        op=schemas.FilterOperator.CONTAINS,
                    ),
                ],
                op=schemas.LogicalOperator.AND
            )
        ),
        ignore_predictions=True,
    ) == {"semsegk1", "semsegk2", "semsegk3"}

    assert get_label_keys(
        db,
        schemas.Filter(
            labels=schemas.LogicalFunction(
                args=[
                    schemas.Condition(
                        lhs=schemas.Symbol.DATASET_NAME,
                        rhs=schemas.Value.infer(dataset_name),
                        op=schemas.FilterOperator.EQ,
                    ),
                    schemas.Condition(
                        lhs=schemas.Symbol.MODEL_NAME,
                        rhs=schemas.Value.infer(model_name),
                        op=schemas.FilterOperator.EQ,
                    ),
                    schemas.Condition(
                        lhs=schemas.Symbol.TASK_TYPE,
                        rhs=schemas.Value(
                            type=schemas.SupportedType.TASK_TYPE,
                            value=enums.TaskType.SEMANTIC_SEGMENTATION,
                        ),
                        op=schemas.FilterOperator.CONTAINS,
                    ),
                ],
                op=schemas.LogicalOperator.AND
            )
        ),
        ignore_groundtruths=True,
    ) == {"semsegk1", "semsegk2", "semsegk3_pred"}

    assert (
        get_labels(
            db,
            schemas.Filter(
                labels=schemas.LogicalFunction(
                    args=[
                        schemas.Condition(
                            lhs=schemas.Symbol.DATASET_NAME,
                            rhs=schemas.Value.infer(dataset_name),
                            op=schemas.FilterOperator.EQ,
                        ),
                        schemas.Condition(
                            lhs=schemas.Symbol.MODEL_NAME,
                            rhs=schemas.Value.infer(model_name),
                            op=schemas.FilterOperator.EQ,
                        ),
                        schemas.Condition(
                            lhs=schemas.Symbol.TASK_TYPE,
                            rhs=schemas.Value(
                                type=schemas.SupportedType.TASK_TYPE,
                                value=enums.TaskType.OBJECT_DETECTION,
                            ),
                            op=schemas.FilterOperator.CONTAINS,
                        ),
                    ],
                    op=schemas.LogicalOperator.AND
                )
            ),
            ignore_groundtruths=True,
        )
        == set()
    )

    assert get_labels(
        db,
        schemas.Filter(
            labels=schemas.LogicalFunction(
                args=[
                    schemas.Condition(
                        lhs=schemas.Symbol.DATASET_NAME,
                        rhs=schemas.Value.infer(dataset_name),
                        op=schemas.FilterOperator.EQ,
                    ),
                    schemas.Condition(
                        lhs=schemas.Symbol.MODEL_NAME,
                        rhs=schemas.Value.infer(model_name),
                        op=schemas.FilterOperator.EQ,
                    ),
                    schemas.LogicalFunction(
                        args=[
                            schemas.Condition(
                                lhs=schemas.Symbol.TASK_TYPE,
                                rhs=schemas.Value(
                                    type=schemas.SupportedType.TASK_TYPE,
                                    value=enums.TaskType.OBJECT_DETECTION,
                                ),
                                op=schemas.FilterOperator.CONTAINS,
                            ),
                            schemas.Condition(
                                lhs=schemas.Symbol.TASK_TYPE,
                                rhs=schemas.Value(
                                    type=schemas.SupportedType.TASK_TYPE,
                                    value=enums.TaskType.SEMANTIC_SEGMENTATION,
                                ),
                                op=schemas.FilterOperator.CONTAINS,
                            ),
                        ],
                        op=schemas.LogicalOperator.OR,
                    ),
                    schemas.Condition(
                        lhs=schemas.Symbol.RASTER,
                        op=schemas.FilterOperator.ISNOTNULL,
                    ),
                ],
                op=schemas.LogicalOperator.AND
            )
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
                labels=schemas.LogicalFunction(
                    args=[
                        schemas.Condition(
                            lhs=schemas.Symbol.DATASET_NAME,
                            rhs=schemas.Value.infer(dataset_name),
                            op=schemas.FilterOperator.EQ,
                        ),
                        schemas.Condition(
                            lhs=schemas.Symbol.MODEL_NAME,
                            rhs=schemas.Value.infer(model_name),
                            op=schemas.FilterOperator.EQ,
                        ),
                        schemas.Condition(
                            lhs=schemas.Symbol.TASK_TYPE,
                            rhs=schemas.Value(
                                type=schemas.SupportedType.TASK_TYPE,
                                value=enums.TaskType.OBJECT_DETECTION,
                            ),
                            op=schemas.FilterOperator.CONTAINS,
                        ),
                        schemas.Condition(
                            lhs=schemas.Symbol.RASTER,
                            op=schemas.FilterOperator.ISNOTNULL,
                        ),
                    ],
                    op=schemas.LogicalOperator.AND
                )
            ),
            ignore_groundtruths=True,
        )
        == set()
    )
