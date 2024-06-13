import pytest
from sqlalchemy.orm import Session

from valor_api import crud, enums, exceptions, schemas


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


def test_get_labels(
    db: Session,
    dataset_name: str,
    groundtruth_detections: list[schemas.GroundTruth],
):
    crud.create_dataset(db=db, dataset=schemas.Dataset(name=dataset_name))

    for gt in groundtruth_detections:
        crud.create_groundtruths(db=db, groundtruths=[gt])

    labels, headers = crud.get_labels(db=db)

    assert len(labels) == 2
    assert set([(label.key, label.value) for label in labels]) == set(
        [("k1", "v1"), ("k2", "v2")]
    )
    assert headers == {"content-range": "items 0-1/2"}


def test_get_labels_from_dataset(
    db: Session,
    dataset_name: str,
    dataset_model_create,
):
    # Test get all from dataset 1
    ds1, headers = crud.get_labels(
        db=db,
        filters=schemas.Filter(
            labels=schemas.Condition(
                lhs=schemas.Symbol(name=schemas.SupportedSymbol.DATASET_NAME),
                rhs=schemas.Value.infer(dataset_name),
                op=schemas.FilterOperator.EQ,
            ),
        ),
        ignore_prediction_labels=True,
    )
    assert len(ds1) == 2
    assert schemas.Label(key="k1", value="v1") in ds1
    assert schemas.Label(key="k2", value="v2") in ds1
    assert headers == {"content-range": "items 0-1/2"}

    # NEGATIVE - Test filter by task type
    # This should be same result as previous b/c dataset only has Obj Dets
    ds1, _ = crud.get_labels(
        db=db,
        filters=schemas.Filter(
            groundtruths=schemas.LogicalFunction(
                args=[
                    schemas.Condition(
                        lhs=schemas.Symbol(
                            name=schemas.SupportedSymbol.DATASET_NAME
                        ),
                        rhs=schemas.Value.infer(dataset_name),
                        op=schemas.FilterOperator.EQ,
                    ),
                    schemas.LogicalFunction(
                        args=[
                            schemas.Condition(
                                lhs=schemas.Symbol(
                                    name=schemas.SupportedSymbol.TASK_TYPE
                                ),
                                rhs=schemas.Value.infer(
                                    enums.TaskType.OBJECT_DETECTION
                                ),
                                op=schemas.FilterOperator.CONTAINS,
                            ),
                            schemas.Condition(
                                lhs=schemas.Symbol(
                                    name=schemas.SupportedSymbol.TASK_TYPE
                                ),
                                rhs=schemas.Value.infer(
                                    enums.TaskType.SEMANTIC_SEGMENTATION
                                ),
                                op=schemas.FilterOperator.CONTAINS,
                            ),
                        ],
                        op=schemas.LogicalOperator.OR,
                    ),
                ],
                op=schemas.LogicalOperator.AND,
            )
        ),
        ignore_prediction_labels=True,
    )
    assert len(ds1) == 2
    assert schemas.Label(key="k1", value="v1") in ds1
    assert schemas.Label(key="k2", value="v2") in ds1

    # POSITIVE - Test filter by task type
    ds1, _ = crud.get_labels(
        db=db,
        filters=schemas.Filter(
            labels=schemas.LogicalFunction(
                args=[
                    schemas.Condition(
                        lhs=schemas.Symbol(
                            name=schemas.SupportedSymbol.DATASET_NAME
                        ),
                        rhs=schemas.Value.infer(dataset_name),
                        op=schemas.FilterOperator.EQ,
                    ),
                    schemas.Condition(
                        lhs=schemas.Symbol(
                            name=schemas.SupportedSymbol.TASK_TYPE
                        ),
                        rhs=schemas.Value.infer(
                            enums.TaskType.OBJECT_DETECTION
                        ),
                        op=schemas.FilterOperator.CONTAINS,
                    ),
                ],
                op=schemas.LogicalOperator.AND,
            )
        ),
        ignore_prediction_labels=True,
    )
    assert len(ds1) == 2
    assert schemas.Label(key="k1", value="v1") in ds1
    assert schemas.Label(key="k2", value="v2") in ds1

    # NEGATIVE - Test filter by annotation type
    ds1, _ = crud.get_labels(
        db=db,
        filters=schemas.Filter(
            labels=schemas.LogicalFunction(
                args=[
                    schemas.Condition(
                        lhs=schemas.Symbol(
                            name=schemas.SupportedSymbol.DATASET_NAME
                        ),
                        rhs=schemas.Value.infer(dataset_name),
                        op=schemas.FilterOperator.EQ,
                    ),
                    schemas.Condition(
                        lhs=schemas.Symbol(name=schemas.SupportedSymbol.BOX),
                        op=schemas.FilterOperator.ISNULL,
                    ),
                ],
                op=schemas.LogicalOperator.AND,
            )
        ),
        ignore_prediction_labels=True,
    )
    assert len(ds1) == 1
    assert schemas.Label(key="k2", value="v2") in ds1

    # POSITIVE - Test filter by annotation type
    ds1, _ = crud.get_labels(
        db=db,
        filters=schemas.Filter(
            labels=schemas.LogicalFunction(
                args=[
                    schemas.Condition(
                        lhs=schemas.Symbol(
                            name=schemas.SupportedSymbol.DATASET_NAME
                        ),
                        rhs=schemas.Value.infer(dataset_name),
                        op=schemas.FilterOperator.EQ,
                    ),
                    schemas.Condition(
                        lhs=schemas.Symbol(
                            name=schemas.SupportedSymbol.POLYGON
                        ),
                        op=schemas.FilterOperator.ISNOTNULL,
                    ),
                ],
                op=schemas.LogicalOperator.AND,
            )
        ),
        ignore_prediction_labels=True,
    )
    assert len(ds1) == 1
    assert schemas.Label(key="k2", value="v2") in ds1

    # POSITIVE - Test filter by annotation type
    ds1, _ = crud.get_labels(
        db=db,
        filters=schemas.Filter(
            labels=schemas.LogicalFunction(
                args=[
                    schemas.Condition(
                        lhs=schemas.Symbol(
                            name=schemas.SupportedSymbol.DATASET_NAME
                        ),
                        rhs=schemas.Value.infer(dataset_name),
                        op=schemas.FilterOperator.EQ,
                    ),
                    schemas.Condition(
                        lhs=schemas.Symbol(name=schemas.SupportedSymbol.BOX),
                        op=schemas.FilterOperator.ISNOTNULL,
                    ),
                ],
                op=schemas.LogicalOperator.AND,
            )
        ),
        ignore_prediction_labels=True,
    )
    assert len(ds1) == 2
    assert schemas.Label(key="k1", value="v1") in ds1
    assert schemas.Label(key="k2", value="v2") in ds1


def test_get_labels_from_model(
    db: Session,
    model_name: str,
    dataset_model_create,
):
    # Test get all labels from model 1
    md1, _ = crud.get_labels(
        db=db,
        filters=schemas.Filter(
            labels=schemas.Condition(
                lhs=schemas.Symbol(name=schemas.SupportedSymbol.MODEL_NAME),
                rhs=schemas.Value.infer(model_name),
                op=schemas.FilterOperator.EQ,
            )
        ),
        ignore_groundtruth_labels=True,
    )
    assert len(md1) == 4
    assert schemas.Label(key="k1", value="v1") in md1
    assert schemas.Label(key="k1", value="v2") in md1
    assert schemas.Label(key="k2", value="v1") in md1
    assert schemas.Label(key="k2", value="v2") in md1

    # Test get all but polygon labels from model 1
    md1, _ = crud.get_labels(
        db=db,
        filters=schemas.Filter(
            labels=schemas.LogicalFunction(
                args=[
                    schemas.Condition(
                        lhs=schemas.Symbol(
                            name=schemas.SupportedSymbol.MODEL_NAME
                        ),
                        rhs=schemas.Value.infer(model_name),
                        op=schemas.FilterOperator.EQ,
                    ),
                    schemas.Condition(
                        lhs=schemas.Symbol(
                            name=schemas.SupportedSymbol.TASK_TYPE
                        ),
                        rhs=schemas.Value.infer(enums.TaskType.CLASSIFICATION),
                        op=schemas.FilterOperator.CONTAINS,
                    ),
                ],
                op=schemas.LogicalOperator.AND,
            )
        ),
        ignore_groundtruth_labels=True,
    )
    assert md1 == set()

    # Test get only polygon labels from model 1
    md1, _ = crud.get_labels(
        db=db,
        filters=schemas.Filter(
            labels=schemas.LogicalFunction(
                args=[
                    schemas.Condition(
                        lhs=schemas.Symbol(
                            name=schemas.SupportedSymbol.MODEL_NAME
                        ),
                        rhs=schemas.Value.infer(model_name),
                        op=schemas.FilterOperator.EQ,
                    ),
                    schemas.Condition(
                        lhs=schemas.Symbol(name=schemas.SupportedSymbol.BOX),
                        op=schemas.FilterOperator.ISNOTNULL,
                    ),
                ],
                op=schemas.LogicalOperator.AND,
            )
        ),
        ignore_groundtruth_labels=True,
    )
    assert len(md1) == 4
    assert schemas.Label(key="k1", value="v1") in md1
    assert schemas.Label(key="k1", value="v2") in md1
    assert schemas.Label(key="k2", value="v1") in md1
    assert schemas.Label(key="k2", value="v2") in md1


def test_get_dataset_summary(
    db: Session, dataset_name: str, dataset_model_create
):
    summary = crud.get_dataset_summary(db=db, name=dataset_name)
    assert summary.name == dataset_name
    assert summary.num_datums == 2
    assert summary.num_annotations == 6
    assert summary.num_bounding_boxes == 3
    assert summary.num_polygons == 1
    assert summary.num_rasters == 1
    assert set(summary.task_types) == {
        enums.TaskType.OBJECT_DETECTION,
        enums.TaskType.CLASSIFICATION,
        enums.TaskType.EMPTY,
    }
    assert summary.datum_metadata == [
        {
            "width": 32,
            "height": 80,
        },
        {
            "width": 200,
            "height": 100,
        },
    ]
    assert summary.annotation_metadata == [
        {"int_key": 1},
        {
            "string_key": "string_val",
            "int_key": 1,
        },
    ]
