from typing import Sequence

import numpy
import pytest
from sqlalchemy import distinct, func
from sqlalchemy.exc import ArgumentError
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import InstrumentedAttribute

from valor_api import crud, schemas
from valor_api.backend import models
from valor_api.backend.query.ops import generate_query, generate_select
from valor_api.backend.query.types import LabelSourceAlias
from valor_api.schemas.filters import (
    Filter,
    Symbol,
    Value,
    Condition,
    LogicalFunction,
    FilterOperator,
    LogicalOperator,
    SupportedType,
)

dset_name = "dataset1"
model_name1 = "model1"
model_name2 = "model2"
datum_uid1 = "uid1"
datum_uid2 = "uid2"
datum_uid3 = "uid3"
datum_uid4 = "uid4"


@pytest.fixture
def geospatial_coordinates() -> dict[
    str,
    dict,
]:
    return {
        SupportedType.POINT: {"type": "Point", "coordinates": [125.2750725, 38.760525]},
        "polygon1": {
            "type": "Polygon",
            "coordinates": [
                [
                    [-10, -10],
                    [10, -10],
                    [10, 10],
                    [-10, 10],
                    [-10, -10],
                ]
            ],
        },
        "polygon2": {
            "type": "Polygon",
            "coordinates": [
                [
                    [20, 20],
                    [20, 30],
                    [30, 30],
                    [30, 20],
                    [20, 20],
                ]
            ],
        },
        "polygon3": {
            "type": "Polygon",
            "coordinates": [
                [
                    [80, 80],
                    [100, 80],
                    [90, 120],
                    [80, 80],
                ]
            ],
        },
        SupportedType.MULTIPOLYGON: {
            "type": "MultiPolygon",
            "coordinates": [
                [
                    [
                        [50, 50],
                        [70, 50],
                        [70, 70],
                        [50, 70],
                        [50, 50],
                    ],
                    [
                        [30, 30],
                        [35, 30],
                        [35, 35],
                        [30, 35],
                        [30, 30],
                    ],
                ],
                [
                    [
                        [10, 10],
                        [20, 10],
                        [20, 20],
                        [10, 20],
                        [10, 10],
                    ],
                ],
            ],
        },
    }


@pytest.fixture
def metadata_1(geospatial_coordinates) -> dict[str, int | float | str | dict]:
    return {
        "some_numeric_attribute": 0.4,
        "some_str_attribute": "abc",
        "height": 10,
        "width": 10,
        "some_bool_attribute": True,
        "some_geo_attribute": {
            "type": "geojson",
            "value": geospatial_coordinates["polygon1"],
        },
    }


@pytest.fixture
def metadata_2(geospatial_coordinates) -> dict[str, int | float | str | dict]:
    return {
        "some_numeric_attribute": 0.6,
        "some_str_attribute": "abc",
        "height": 10,
        "width": 10,
        "some_bool_attribute": False,
        "some_geo_attribute": {
            "type": "geojson",
            "value": geospatial_coordinates[SupportedType.MULTIPOLYGON],
        },
    }


@pytest.fixture
def metadata_3(geospatial_coordinates) -> dict[str, int | float | str | dict]:
    return {
        "some_numeric_attribute": 0.4,
        "some_str_attribute": "xyz",
        "height": 10,
        "width": 10,
        "some_bool_attribute": True,
        "some_geo_attribute": {
            "type": "geojson",
            "value": geospatial_coordinates["polygon2"],
        },
    }


@pytest.fixture
def metadata_4(geospatial_coordinates) -> dict[str, int | float | str | dict]:
    return {
        "some_numeric_attribute": 0.6,
        "some_str_attribute": "xyz",
        "height": 10,
        "width": 10,
        "some_bool_attribute": False,
        "some_geo_attribute": {
            "type": "geojson",
            "value": geospatial_coordinates["polygon3"],
        },
    }


@pytest.fixture
def label_dog() -> schemas.Label:
    return schemas.Label(key="class", value="dog")


@pytest.fixture
def label_cat() -> schemas.Label:
    return schemas.Label(key="class", value="cat")


@pytest.fixture
def label_tree() -> schemas.Label:
    return schemas.Label(key="class", value="tree")


@pytest.fixture
def raster_1():
    r = numpy.zeros((10, 10))
    r = r != 0
    r[5:] = True
    return schemas.Raster.from_numpy(r)


@pytest.fixture
def raster_2():
    r = numpy.zeros((10, 10))
    r = r != 0
    r[9:] = True
    return schemas.Raster.from_numpy(r)


@pytest.fixture
def datum_1(metadata_1) -> schemas.Datum:
    return schemas.Datum(
        uid=datum_uid1,
        metadata=metadata_1,
    )


@pytest.fixture
def datum_2(metadata_2) -> schemas.Datum:
    return schemas.Datum(
        uid=datum_uid2,
        metadata=metadata_2,
    )


@pytest.fixture
def datum_3(metadata_3) -> schemas.Datum:
    return schemas.Datum(
        uid=datum_uid3,
        metadata=metadata_3,
    )


@pytest.fixture
def datum_4(metadata_4) -> schemas.Datum:
    return schemas.Datum(
        uid=datum_uid4,
        metadata=metadata_4,
    )


@pytest.fixture
def groundtruth_annotations_cat(
    label_cat,
    raster_1,
    raster_2,
    metadata_1,
    metadata_2,
) -> list[schemas.Annotation]:
    return [
        schemas.Annotation(
            labels=[label_cat],
        ),
        schemas.Annotation(
            labels=[label_cat],
            bounding_box=schemas.Box.from_extrema(
                xmin=0, ymin=0, xmax=10, ymax=10
            ),
            is_instance=True,
            metadata=metadata_1,
        ),
        schemas.Annotation(
            labels=[label_cat],
            bounding_box=schemas.Box.from_extrema(
                xmin=0, ymin=0, xmax=1, ymax=50
            ),
            is_instance=True,
            metadata=metadata_2,
        ),
        schemas.Annotation(
            labels=[label_cat],
            raster=raster_1,
            metadata=metadata_1,
            is_instance=True,
        ),
        schemas.Annotation(
            labels=[label_cat],
            raster=raster_2,
            metadata=metadata_2,
            is_instance=True,
        ),
    ]


@pytest.fixture
def groundtruth_annotations_dog(
    label_dog,
    raster_1,
    raster_2,
    metadata_3,
    metadata_4,
) -> list[schemas.Annotation]:
    return [
        schemas.Annotation(
            labels=[label_dog],
        ),
        schemas.Annotation(
            labels=[label_dog],
            bounding_box=schemas.Box.from_extrema(
                xmin=0, ymin=0, xmax=10, ymax=10
            ),
            is_instance=True,
            metadata=metadata_3,
        ),
        schemas.Annotation(
            labels=[label_dog],
            bounding_box=schemas.Box.from_extrema(
                xmin=0, ymin=0, xmax=1, ymax=50
            ),
            is_instance=True,
            metadata=metadata_4,
        ),
        schemas.Annotation(
            labels=[label_dog],
            raster=raster_1,
            metadata=metadata_3,
            is_instance=True,
        ),
        schemas.Annotation(
            labels=[label_dog],
            raster=raster_2,
            metadata=metadata_4,
            is_instance=True,
        ),
    ]


@pytest.fixture
def prediction_annotations_cat(
    raster_1,
    raster_2,
    metadata_1,
    metadata_2,
) -> list[schemas.Annotation]:
    return [
        schemas.Annotation(
            labels=[
                schemas.Label(key="class", value="cat", score=0.9),
                schemas.Label(key="class", value="dog", score=0.1),
            ],
        ),
        schemas.Annotation(
            labels=[
                schemas.Label(key="class", value="cat", score=0.8),
                schemas.Label(key="class", value="dog", score=0.2),
            ],
            bounding_box=schemas.Box.from_extrema(
                xmin=0, ymin=0, xmax=10, ymax=10
            ),
            is_instance=True,
            metadata=metadata_1,
        ),
        schemas.Annotation(
            labels=[
                schemas.Label(key="class", value="cat", score=0.7),
                schemas.Label(key="class", value="dog", score=0.3),
            ],
            bounding_box=schemas.Box.from_extrema(
                xmin=0, ymin=0, xmax=1, ymax=50
            ),
            is_instance=True,
            metadata=metadata_2,
        ),
        schemas.Annotation(
            labels=[
                schemas.Label(key="class", value="cat", score=0.75),
                schemas.Label(key="class", value="dog", score=0.25),
            ],
            raster=raster_1,
            metadata=metadata_1,
            is_instance=True,
        ),
        schemas.Annotation(
            labels=[
                schemas.Label(key="class", value="cat", score=0.95),
                schemas.Label(key="class", value="dog", score=0.05),
            ],
            raster=raster_2,
            metadata=metadata_2,
            is_instance=True,
        ),
    ]


@pytest.fixture
def prediction_annotations_dog(
    raster_1,
    raster_2,
    metadata_3,
    metadata_4,
) -> list[schemas.Annotation]:
    return [
        schemas.Annotation(
            labels=[
                schemas.Label(key="class", value="cat", score=0.1),
                schemas.Label(key="class", value="dog", score=0.9),
            ],
        ),
        schemas.Annotation(
            labels=[
                schemas.Label(key="class", value="cat", score=0.2),
                schemas.Label(key="class", value="dog", score=0.8),
            ],
            bounding_box=schemas.Box.from_extrema(
                xmin=0, ymin=0, xmax=10, ymax=10
            ),
            is_instance=True,
            metadata=metadata_3,
        ),
        schemas.Annotation(
            labels=[
                schemas.Label(key="class", value="cat", score=0.3),
                schemas.Label(key="class", value="dog", score=0.7),
            ],
            bounding_box=schemas.Box.from_extrema(
                xmin=0, ymin=0, xmax=1, ymax=50
            ),
            is_instance=True,
            metadata=metadata_4,
        ),
        schemas.Annotation(
            labels=[
                schemas.Label(key="class", value="cat", score=0.25),
                schemas.Label(key="class", value="dog", score=0.75),
            ],
            raster=raster_1,
            metadata=metadata_3,
            is_instance=True,
        ),
        schemas.Annotation(
            labels=[
                schemas.Label(key="class", value="cat", score=0.05),
                schemas.Label(key="class", value="dog", score=0.95),
            ],
            raster=raster_2,
            metadata=metadata_4,
            is_instance=True,
        ),
    ]


@pytest.fixture
def groundtruth_cat_datum_1(
    datum_1,
    groundtruth_annotations_cat,
) -> schemas.GroundTruth:
    return schemas.GroundTruth(
        dataset_name=dset_name,
        datum=datum_1,
        annotations=groundtruth_annotations_cat,
    )


@pytest.fixture
def groundtruth_cat_datum_2(
    datum_2,
    groundtruth_annotations_cat,
) -> schemas.GroundTruth:
    return schemas.GroundTruth(
        dataset_name=dset_name,
        datum=datum_2,
        annotations=groundtruth_annotations_cat,
    )


@pytest.fixture
def groundtruth_dog_datum_3(
    datum_3,
    groundtruth_annotations_dog,
) -> schemas.GroundTruth:
    return schemas.GroundTruth(
        dataset_name=dset_name,
        datum=datum_3,
        annotations=groundtruth_annotations_dog,
    )


@pytest.fixture
def groundtruth_dog_datum_4(
    datum_4,
    groundtruth_annotations_dog,
) -> schemas.GroundTruth:
    return schemas.GroundTruth(
        dataset_name=dset_name,
        datum=datum_4,
        annotations=groundtruth_annotations_dog,
    )


@pytest.fixture
def prediction_cat_datum1_model1(
    datum_1,
    prediction_annotations_cat,
) -> schemas.Prediction:
    return schemas.Prediction(
        dataset_name=dset_name,
        model_name=model_name1,
        datum=datum_1,
        annotations=prediction_annotations_cat,
    )


@pytest.fixture
def prediction_cat_datum2_model1(
    datum_2,
    prediction_annotations_cat,
) -> schemas.Prediction:
    return schemas.Prediction(
        dataset_name=dset_name,
        model_name=model_name1,
        datum=datum_2,
        annotations=prediction_annotations_cat,
    )


@pytest.fixture
def prediction_dog_datum3_model1(
    datum_3,
    prediction_annotations_dog,
) -> schemas.Prediction:
    return schemas.Prediction(
        dataset_name=dset_name,
        model_name=model_name1,
        datum=datum_3,
        annotations=prediction_annotations_dog,
    )


@pytest.fixture
def prediction_dog_datum4_model1(
    datum_4,
    prediction_annotations_dog,
) -> schemas.Prediction:
    return schemas.Prediction(
        dataset_name=dset_name,
        model_name=model_name1,
        datum=datum_4,
        annotations=prediction_annotations_dog,
    )


@pytest.fixture
def prediction_dog_datum1_model2(
    datum_1,
    prediction_annotations_dog,
) -> schemas.Prediction:
    return schemas.Prediction(
        dataset_name=dset_name,
        model_name=model_name2,
        datum=datum_1,
        annotations=prediction_annotations_dog,
    )


@pytest.fixture
def prediction_dog_datum2_model2(
    datum_2,
    prediction_annotations_dog,
) -> schemas.Prediction:
    return schemas.Prediction(
        dataset_name=dset_name,
        model_name=model_name2,
        datum=datum_2,
        annotations=prediction_annotations_dog,
    )


@pytest.fixture
def prediction_cat_datum3_model2(
    datum_3,
    prediction_annotations_cat,
) -> schemas.Prediction:
    return schemas.Prediction(
        dataset_name=dset_name,
        model_name=model_name2,
        datum=datum_3,
        annotations=prediction_annotations_cat,
    )


@pytest.fixture
def prediction_cat_datum4_model2(
    datum_4,
    prediction_annotations_cat,
) -> schemas.Prediction:
    return schemas.Prediction(
        dataset_name=dset_name,
        model_name=model_name2,
        datum=datum_4,
        annotations=prediction_annotations_cat,
    )


@pytest.fixture
def dataset_sim(
    db: Session,
    metadata_1,
    groundtruth_cat_datum_1,
    groundtruth_cat_datum_2,
    groundtruth_dog_datum_3,
    groundtruth_dog_datum_4,
):
    crud.create_dataset(
        db=db,
        dataset=schemas.Dataset(
            name=dset_name,
            metadata=metadata_1,
        ),
    )
    crud.create_groundtruths(
        db=db,
        groundtruths=[
            groundtruth_cat_datum_1,
            groundtruth_cat_datum_2,
            groundtruth_dog_datum_3,
            groundtruth_dog_datum_4,
        ],
    )
    crud.finalize(db=db, dataset_name=dset_name)


@pytest.fixture
def model_sim(
    db: Session,
    dataset_sim,
    metadata_1,
    metadata_4,
    prediction_cat_datum1_model1,
    prediction_cat_datum2_model1,
    prediction_dog_datum3_model1,
    prediction_dog_datum4_model1,
    prediction_dog_datum1_model2,
    prediction_dog_datum2_model2,
    prediction_cat_datum3_model2,
    prediction_cat_datum4_model2,
):
    crud.create_model(
        db=db,
        model=schemas.Model(
            name=model_name1,
            metadata=metadata_1,
        ),
    )
    crud.create_predictions(
        db=db,
        predictions=[
            prediction_cat_datum1_model1,
            prediction_cat_datum2_model1,
            prediction_dog_datum3_model1,
            prediction_dog_datum4_model1,
        ],
    )
    crud.finalize(db=db, dataset_name=dset_name, model_name=model_name1)

    crud.create_model(
        db=db,
        model=schemas.Model(
            name=model_name2,
            metadata=metadata_4,
        ),
    )
    crud.create_predictions(
        db=db,
        predictions=[
            prediction_dog_datum1_model2,
            prediction_dog_datum2_model2,
            prediction_cat_datum3_model2,
            prediction_cat_datum4_model2,
        ],
    )
    crud.finalize(db=db, dataset_name=dset_name, model_name=model_name2)


def create_dataset_filter(name: str) -> Condition:
    return Condition(
        lhs=Symbol.DATASET_NAME,
        rhs=Value.infer(name),
        op=FilterOperator.EQ,
    )


def create_model_filter(name: str) -> Condition:
    return Condition(
        lhs=Symbol.MODEL_NAME,
        rhs=Value.infer(name),
        op=FilterOperator.EQ,
    )


def create_datum_filter(uid: str) -> Condition:
    return Condition(
        lhs=Symbol.DATUM_UID,
        rhs=Value.infer(uid),
        op=FilterOperator.EQ,
    )


def create_label_filter(key: str, value: str) -> LogicalFunction:
    return LogicalFunction(
        args=[
            Condition(
                lhs=Symbol.LABEL_KEY,
                rhs=Value.infer(key),
                op=FilterOperator.EQ,
            ),
            Condition(
                lhs=Symbol.LABEL_VALUE,
                rhs=Value.infer(value),
                op=FilterOperator.EQ,
            ),
        ],
        op=LogicalOperator.AND,
    )


def test_query_datasets(
    db: Session,
    model_sim,
):
    # Check that passing a non-InstrumentedAttribute returns None
    with pytest.raises(ArgumentError):
        generate_select("not a valid attribute")

    # Q: Get names for datasets where label class=cat exists in groundtruths.
    f = Filter(groundtruths=create_label_filter(key="class", value="cat"))
    dataset_names = generate_select(
        distinct(models.Dataset.name),
        filters=f,
        label_source=models.GroundTruth,
    )
    dataset_names = db.query(dataset_names.subquery()).all()
    assert len(dataset_names) == 1
    assert (dset_name,) in dataset_names

    # Q: Get names for datasets where label=tree exists in groundtruths
    f = Filter(groundtruths=create_label_filter(key="class", value="tree"))
    dataset_names = generate_select(
        distinct(models.Dataset.name),
        filters=f,
        label_source=models.GroundTruth,
    )
    dataset_names = db.query(dataset_names.subquery()).all()
    assert len(dataset_names) == 0


def test_query_models(
    db: Session,
    model_sim,
):
    # Q: Get names for all models that operate over a dataset.
    f = Filter(
        predictions=create_dataset_filter(dset_name),
    )
    model_names = generate_select(models.Model.name, filters=f).distinct()
    model_names = db.query(model_names.subquery()).all()
    assert len(model_names) == 2
    assert (model_name1,) in model_names
    assert (model_name2,) in model_names

    # Q: Get names for models that operate over dataset that doesn't exist.
    f = Filter(predictions=create_dataset_filter("invalid"))
    model_names = generate_select(models.Model.name, filters=f).distinct()
    model_names = db.query(model_names.subquery()).all()
    assert len(model_names) == 0

    # Q: Get names for models where label=cat exists in predictions
    f = Filter(predictions=create_label_filter(key="class", value="cat"))
    model_names = generate_select(
        models.Model.name, filters=f, label_source=models.Prediction
    ).distinct()
    model_names = db.query(model_names.subquery()).all()
    assert len(model_names) == 2
    assert (model_name1,) in model_names
    assert (model_name2,) in model_names

    # Q: Get names for models where label=tree exists in predictions
    f = Filter(predictions=create_label_filter(key="class", value="tree"))
    model_names = generate_select(
        models.Model.name, filters=f, label_source=models.Prediction
    ).distinct()
    model_names = db.query(model_names.subquery()).all()
    assert len(model_names) == 0

    # Q: Get models with metadatum with `numeric` > 0.5.
    f = Filter(
        predictions=Condition(
            lhs=Symbol.MODEL_META,
            lhs_key="some_numeric_attribute",
            rhs=Value.infer(0.5),
            op=FilterOperator.GT,
        )
    )
    model_names = generate_select(
        models.Model.name, filters=f, label_source=models.Prediction
    ).distinct()
    model_names = db.query(model_names.subquery()).all()
    assert len(model_names) == 1
    assert (model_name2,) in model_names

    # Q: Get models with metadatum with `numeric` < 0.5.
    f = Filter(
        predictions=Condition(
            lhs=Symbol.MODEL_META,
            lhs_key="some_numeric_attribute",
            rhs=Value.infer(0.5),
            op=FilterOperator.LT,
        )
    )
    model_names = generate_select(
        models.Model.name, filters=f, label_source=models.Prediction
    ).distinct()
    model_names = db.query(model_names.subquery()).all()
    assert len(model_names) == 1
    assert (model_name1,) in model_names


def test_query_by_metadata(
    db: Session,
    model_sim,
):
    # Q: Get datums with metadatum with `numeric` < 0.5, `str` == 'abc', and `bool` == True.
    f = Filter(
        datums=LogicalFunction(
            args=[
                Condition(
                    lhs=Symbol.DATUM_META,
                    lhs_key="some_numeric_attribute",
                    rhs=Value.infer(0.5),
                    op=FilterOperator.LT,
                ),
                Condition(
                    lhs=Symbol.DATUM_META,
                    lhs_key="some_str_attribute",
                    rhs=Value.infer("abc"),
                    op=FilterOperator.EQ,
                ),
                Condition(
                    lhs=Symbol.DATUM_META,
                    lhs_key="some_bool_attribute",
                    rhs=Value.infer(True),
                    op=FilterOperator.EQ,
                ),
            ],
            op=LogicalOperator.AND,
        )
    )
    datum_uids = generate_select(models.Datum.uid, filters=f).distinct()
    datum_uids = db.query(datum_uids.subquery()).all()
    assert len(datum_uids) == 1
    assert (datum_uid1,) in datum_uids

    # repeat with `bool` == False or != `True` and check we get nothing
    negative1 = Condition(
        lhs=Symbol.DATUM_META,
        lhs_key="some_bool_attribute",
        rhs=Value.infer(False),
        op=FilterOperator.EQ,
    )
    negative2 = Condition(
        lhs=Symbol.DATUM_META,
        lhs_key="some_bool_attribute",
        rhs=Value.infer(True),
        op=FilterOperator.NE,
    )
    for bool_filter in [negative1, negative2]:
        f = Filter(
            groundtruths=LogicalFunction(
                args=[
                    Condition(
                        lhs=Symbol.DATUM_META,
                        lhs_key="some_numeric_attribute",
                        rhs=Value.infer(0.5),
                        op=FilterOperator.LT,
                    ),
                    Condition(
                        lhs=Symbol.DATUM_META,
                        lhs_key="some_str_attribute",
                        rhs=Value.infer("abc"),
                        op=FilterOperator.EQ,
                    ),
                    bool_filter,
                ],
                op=LogicalOperator.AND,
            )
        )
        datum_uids = generate_select(models.Datum.uid, filters=f).distinct()
        datum_uids = db.query(datum_uids.subquery()).all()
        assert len(datum_uids) == 0

    # Q: Get datums with metadatum with `numeric` > 0.5 and `str` == 'abc'.
    f = Filter(
        datums=LogicalFunction(
            args=[
                Condition(
                    lhs=Symbol.DATUM_META,
                    lhs_key="some_numeric_attribute",
                    rhs=Value.infer(0.5),
                    op=FilterOperator.GT,
                ),
                Condition(
                    lhs=Symbol.DATUM_META,
                    lhs_key="some_str_attribute",
                    rhs=Value.infer("abc"),
                    op=FilterOperator.EQ,
                ),
            ],
            op=LogicalOperator.AND,
        )
    )
    datum_uids = generate_select(models.Datum.uid, filters=f).distinct()
    datum_uids = db.query(datum_uids.subquery()).all()
    assert len(datum_uids) == 1
    assert (datum_uid2,) in datum_uids

    # Q: Get datums with metadatum with `numeric` < 0.5 and `str` == 'xyz'.
    f = Filter(
        datums=LogicalFunction(
            args=[
                Condition(
                    lhs=Symbol.DATUM_META,
                    lhs_key="some_numeric_attribute",
                    rhs=Value.infer(0.5),
                    op=FilterOperator.LT,
                ),
                Condition(
                    lhs=Symbol.DATUM_META,
                    lhs_key="some_str_attribute",
                    rhs=Value.infer("xyz"),
                    op=FilterOperator.EQ,
                ),
            ],
            op=LogicalOperator.AND,
        )
    )
    datum_uids = generate_select(models.Datum.uid, filters=f).distinct()
    datum_uids = db.query(datum_uids.subquery()).all()
    assert len(datum_uids) == 1
    assert (datum_uid3,) in datum_uids

    # Q: Get models with metadatum with `numeric` > 0.5 and `str` == 'xyz'.
    f = Filter(
        datums=LogicalFunction(
            args=[
                Condition(
                    lhs=Symbol.DATUM_META,
                    lhs_key="some_numeric_attribute",
                    rhs=Value.infer(0.5),
                    op=FilterOperator.GT,
                ),
                Condition(
                    lhs=Symbol.DATUM_META,
                    lhs_key="some_str_attribute",
                    rhs=Value.infer("xyz"),
                    op=FilterOperator.EQ,
                ),
            ],
            op=LogicalOperator.AND,
        )
    )
    datum_uids = generate_select(models.Datum.uid, filters=f).distinct()
    datum_uids = db.query(datum_uids.subquery()).all()
    assert len(datum_uids) == 1
    assert (datum_uid4,) in datum_uids


def test_query_datums(
    db: Session,
    model_sim,
):
    # Q: Get datums with groundtruth labels of "cat"
    f = Filter(groundtruths=create_label_filter(key="class", value="cat"))
    datum_uids = generate_select(
        models.Datum.uid, filters=f, label_source=models.GroundTruth
    ).distinct()
    datum_uids = db.query(datum_uids.subquery()).all()
    assert len(datum_uids) == 2
    assert (datum_uid1,) in datum_uids
    assert (datum_uid2,) in datum_uids

    # Q: Get datums with groundtruth labels of "dog"
    f = Filter(groundtruths=create_label_filter(key="class", value="dog"))
    datum_uids = generate_select(
        models.Datum.uid, filters=f, label_source=models.GroundTruth
    ).distinct()
    datum_uids = db.query(datum_uids.subquery()).all()
    assert len(datum_uids) == 2
    assert (datum_uid3,) in datum_uids
    assert (datum_uid4,) in datum_uids

    # Q: Get datums with prediction labels of "cat"
    f = Filter(predictions=create_label_filter(key="class", value="cat"))
    datum_uids = generate_select(
        models.Datum.uid,
        filters=f,
        label_source=models.Prediction,
    ).distinct()
    datum_uids = db.query(datum_uids.subquery()).all()
    assert len(datum_uids) == 4
    assert (datum_uid1,) in datum_uids
    assert (datum_uid2,) in datum_uids
    assert (datum_uid3,) in datum_uids
    assert (datum_uid4,) in datum_uids


def test_complex_queries(
    db: Session,
    model_sim,
):
    # Q: Get datums that `model1` has annotations for with label `dog` and prediction score > 0.9.
    f = Filter(
        predictions=LogicalFunction(
            op=LogicalOperator.AND,
            args=[
                create_model_filter(model_name1),
                create_label_filter(key="class", value="dog"),
                Condition(
                    lhs=Symbol.SCORE,
                    rhs=Value.infer(0.9),
                    op=FilterOperator.GT,
                )
            ]
        )
    )
    datum_uids = generate_select(
        models.Datum.uid,
        filters=f,
        label_source=models.Prediction,
    ).distinct()
    datum_uids = db.query(datum_uids.subquery()).all()
    assert len(datum_uids) == 2
    assert (datum_uid3,) in datum_uids
    assert (datum_uid4,) in datum_uids

    # Q: Get datums that `model1` has `bounding_box` annotations for with label `dog` and prediction score > 0.75.
    f = Filter(
        predictions=LogicalFunction(
            op=LogicalOperator.AND,
            args=[
                create_model_filter(model_name1),
                create_label_filter(key="class", value="dog"),
                Condition(
                    lhs=Symbol.SCORE,
                    rhs=Value.infer(0.75),
                    op=FilterOperator.GT,
                ),
                Condition(
                    lhs=Symbol.BOX,
                    op=FilterOperator.ISNOTNULL,
                ),
            ]
        )
    )
    datum_uids = generate_select(
        models.Datum.uid,
        filters=f,
        label_source=models.Prediction,
    ).distinct()
    datum_uids = db.query(datum_uids.subquery()).all()
    assert len(datum_uids) == 2
    assert (datum_uid3,) in datum_uids
    assert (datum_uid4,) in datum_uids


def test_query_by_annotation_geometry(
    db: Session,
    model_sim,
):
    bounding_box_filter = Condition(
        lhs=Symbol.BOX_AREA,
        rhs=Value.infer(75.0),
        op=FilterOperator.GT,
    )

    # Q: Get `bounding_box` annotations that have an area > 75.
    f = Filter(
        annotations=bounding_box_filter,
    )
    annotations = generate_select(
        models.Annotation, filters=f, label_source=models.Prediction
    )
    annotations = db.query(annotations.subquery()).all()
    assert len(annotations) == 12

    # Q: Get `bounding_box` annotations from `model1` that have an area > 75.
    f = Filter(
        predictions=LogicalFunction(
            op=LogicalOperator.AND,
            args=[
                create_model_filter(model_name1),
                bounding_box_filter,
            ]
        )
    )
    annotations = generate_select(
        models.Annotation, filters=f, label_source=models.Prediction
    ).distinct()
    annotations = db.query(annotations.subquery()).all()
    assert len(annotations) == 4


def test_multiple_tables_in_args(
    db: Session,
    model_sim,
):
    f = Filter(
        groundtruths=create_datum_filter(datum_uid1),
        predictions=create_datum_filter(datum_uid1),
    )

    # Q: Get model + dataset name pairings for a datum with `uid1` using the full tables
    pairings = generate_select(
        models.Model, models.Dataset, filters=f
    ).distinct()
    pairings = db.query(pairings.subquery()).all()
    assert len(pairings) == 2
    name_pairings = [(pair[1], pair[6]) for pair in pairings]
    assert (
        model_name1,
        dset_name,
    ) in name_pairings
    assert (
        model_name2,
        dset_name,
    ) in name_pairings

    # Q: Get model + dataset name pairings for a datum with `uid1` using the table attributes directly
    name_pairings = generate_select(
        models.Model.name, models.Dataset.name, filters=f
    ).distinct()
    name_pairings = db.query(name_pairings.subquery()).all()
    assert len(name_pairings) == 2
    assert (
        model_name1,
        dset_name,
    ) in name_pairings
    assert (
        model_name2,
        dset_name,
    ) in name_pairings

    # Q: Get model + dataset name pairings for a datum with `uid1` using a mix of full tables and attributes
    pairings = generate_select(
        models.Model.name, models.Dataset, filters=f
    ).distinct()
    pairings = db.query(pairings.subquery()).all()
    name_pairings = [(pair[0], pair[2]) for pair in pairings]
    assert len(name_pairings) == 2
    assert (
        model_name1,
        dset_name,
    ) in name_pairings
    assert (
        model_name2,
        dset_name,
    ) in name_pairings


def create_geospatial_inside_filter(
    symbol: Symbol,
    value: Value,
) -> Condition:
    return Condition(
        lhs=symbol,
        lhs_key="some_geo_attribute",
        rhs=value,
        op=FilterOperator.INSIDE,
    )


def create_geospatial_outside_filter(
    symbol: Symbol,
    value: Value,
) -> Condition:
    return Condition(
        lhs=symbol,
        lhs_key="some_geo_attribute",
        rhs=value,
        op=FilterOperator.OUTSIDE,
    )


def create_geospatial_intersects_filter(
    symbol: Symbol,
    value: Value,
) -> Condition:
    return Condition(
        lhs=symbol,
        lhs_key="some_geo_attribute",
        rhs=value,
        op=FilterOperator.INTERSECTS,
    )


def _get_geospatial_names_from_filter(
    db: Session,
    value: Value,
    operator: FilterOperator,
    model_object: models.Datum | InstrumentedAttribute,
    symbol: Symbol,
    label_source: LabelSourceAlias = models.Annotation,
):
    match operator:
        case FilterOperator.INSIDE:
            geofilter = create_geospatial_inside_filter(
                symbol=symbol, value=value
            )
        case FilterOperator.OUTSIDE:
            geofilter = create_geospatial_outside_filter(
                symbol=symbol, value=value
            )
        case FilterOperator.INTERSECTS:
            geofilter = create_geospatial_intersects_filter(
                symbol=symbol, value=value
            )
        case _:
            raise NotImplementedError

    f = Filter(
        annotations=geofilter,
    )
    return (
        generate_query(
            model_object, db=db, filters=f, label_source=label_source
        )
        .distinct()
        .all()
    )


def test_datum_geospatial_filters(
    db: Session,
    model_sim,
    model_object=models.Datum.uid,
):
    symbol = Symbol.DATUM_META

    # test inside filters
    names = _get_geospatial_names_from_filter(
        db=db,
        value=Value(
            type=SupportedType.POLYGON,
            value=[
                [
                    [-20, -20],
                    [60, -20],
                    [60, 60],
                    [-20, 60],
                    [-20, -20],
                ]
            ],
        ),
        operator=FilterOperator.INSIDE,
        model_object=model_object,
        symbol=symbol,
    )
    assert len(names) == 2
    assert ("uid1",) in names
    assert ("uid3",) in names

    # test intersections
    names = _get_geospatial_names_from_filter(
        db=db,
        value=Value(
            type=SupportedType.POLYGON,
            value=[
                [
                    [60, 60],
                    [110, 60],
                    [110, 110],
                    [60, 110],
                    [60, 60],
                ]
            ],
        ),
        operator=FilterOperator.INTERSECTS,
        model_object=model_object,
        symbol=symbol,
    )
    assert len(names) == 2
    assert ("uid2",) in names
    assert ("uid4",) in names

    # test point
    names = _get_geospatial_names_from_filter(
        db=db,
        value=Value(
            type=SupportedType.POINT,
            value=[81, 80],
        ),
        operator=FilterOperator.INTERSECTS,
        model_object=model_object,
        symbol=symbol,
    )
    assert len(names) == 1
    assert ("uid4",) in names

    # test multipolygon
    names = _get_geospatial_names_from_filter(
        db=db,
        value=Value(
            type=SupportedType.MULTIPOLYGON,
            value=[
                [
                    [
                        [-20, -20],
                        [20, -20],
                        [20, 20],
                        [-20, 20],
                        [-20, -20],
                    ]
                ],
                [
                    [
                        [15, 15],
                        [15, 35],
                        [35, 35],
                        [35, 15],
                        [15, 15],
                    ]
                ],
            ],
        ),
        operator=FilterOperator.INTERSECTS,
        model_object=model_object,
        symbol=symbol,
    )
    assert len(names) == 3
    assert ("uid1",) in names
    assert ("uid2",) in names
    assert ("uid3",) in names

    # test WHERE miss
    names = _get_geospatial_names_from_filter(
        db=db,
        value=Value(
            type=SupportedType.POINT,
            value=[-11, -11],
        ),
        operator=FilterOperator.INTERSECTS,
        model_object=model_object,
        symbol=symbol,
    )
    assert len(names) == 0

    # test outside
    names = _get_geospatial_names_from_filter(
        db=db,
        value=Value(
            type=SupportedType.POINT,
            value=[-11, -11],
        ),
        operator=FilterOperator.OUTSIDE,
        model_object=model_object,
        symbol=symbol,
    )
    assert len(names) == 4
    assert ("uid1",) in names
    assert ("uid2",) in names
    assert ("uid3",) in names
    assert ("uid4",) in names

    names = _get_geospatial_names_from_filter(
        db=db,
        value=Value(
            type=SupportedType.POLYGON,
            value=[
                [
                    [-20, -20],
                    [60, -20],
                    [60, 60],
                    [-20, 60],
                    [-20, -20],
                ]
            ],
        ),
        operator=FilterOperator.OUTSIDE,
        model_object=model_object,
        symbol=symbol,
    )
    assert len(names) == 2
    assert ("uid2",) in names
    assert ("uid4",) in names


def test_dataset_geospatial_filters(
    db: Session,
    model_sim,
    model_object=models.Dataset.name,
):
    symbol = Symbol.DATASET_META

    # test inside filters
    names = _get_geospatial_names_from_filter(
        db=db,
        value=Value(
            type=SupportedType.POLYGON,
            value=[
                [
                    [-20, -20],
                    [60, -20],
                    [60, 60],
                    [-20, 60],
                    [-20, -20],
                ]
            ],
        ),
        operator=FilterOperator.INSIDE,
        model_object=model_object,
        symbol=symbol,
    )
    assert len(names) == 1
    assert ("dataset1",) in names

    # test point
    names = _get_geospatial_names_from_filter(
        db=db,
        value=Value(
            type=SupportedType.POINT,
            value=[1, 1],
        ),
        operator=FilterOperator.INTERSECTS,
        model_object=model_object,
        symbol=symbol,
    )
    assert len(names) == 1
    assert ("dataset1",) in names

    # test multipolygon
    names = _get_geospatial_names_from_filter(
        db=db,
        value=Value(
            type=SupportedType.MULTIPOLYGON,
            value=[
                [
                    [
                        [-20, -20],
                        [20, -20],
                        [20, 20],
                        [-20, 20],
                        [-20, -20],
                    ]
                ],
                [
                    [
                        [15, 15],
                        [15, 35],
                        [35, 35],
                        [35, 15],
                        [15, 15],
                    ]
                ],
            ],
        ),
        operator=FilterOperator.INTERSECTS,
        model_object=model_object,
        symbol=symbol,
    )
    assert len(names) == 1
    assert ("dataset1",) in names

    # test WHERE miss
    names = _get_geospatial_names_from_filter(
        db=db,
        value=Value(
            type=SupportedType.POINT,
            value=[-11, -11],
        ),
        operator=FilterOperator.INTERSECTS,
        model_object=model_object,
        symbol=symbol,
    )
    assert len(names) == 0

    # test outside
    names = _get_geospatial_names_from_filter(
        db=db,
        value=Value(
            type=SupportedType.POINT,
            value=[-11, -11],
        ),
        operator=FilterOperator.OUTSIDE,
        model_object=model_object,
        symbol=symbol,
    )
    assert len(names) == 1
    assert ("dataset1",) in names


def test_model_geospatial_filters(
    db: Session,
    model_sim,
    model_object=models.Model.name,
):
    symbol = Symbol.MODEL_META

    # test inside filters
    names = _get_geospatial_names_from_filter(
        db=db,
        value=Value(
            type=SupportedType.POLYGON,
            value=[
                [
                    [-20, -20],
                    [60, -20],
                    [60, 60],
                    [-20, 60],
                    [-20, -20],
                ]
            ],
        ),
        operator=FilterOperator.INSIDE,
        model_object=model_object,
        symbol=symbol,
        label_source=models.Prediction,
    )
    assert len(names) == 1
    assert ("model1",) in names

    # test point
    names = _get_geospatial_names_from_filter(
        db=db,
        value=Value(
            type=SupportedType.POINT,
            value=[1, 1],
        ),
        operator=FilterOperator.INTERSECTS,
        model_object=model_object,
        symbol=symbol,
        label_source=models.Prediction,
    )
    assert len(names) == 1
    assert ("model1",) in names

    # test multipolygon
    names = _get_geospatial_names_from_filter(
        db=db,
        value=Value(
            type=SupportedType.MULTIPOLYGON,
            value=[
                [
                    [
                        [-20, -20],
                        [20, -20],
                        [20, 20],
                        [-20, 20],
                        [-20, -20],
                    ]
                ],
                [
                    [
                        [15, 15],
                        [15, 35],
                        [35, 35],
                        [35, 15],
                        [15, 15],
                    ]
                ],
            ],
        ),
        operator=FilterOperator.INTERSECTS,
        model_object=model_object,
        symbol=symbol,
        label_source=models.Prediction,
    )
    assert len(names) == 1
    assert ("model1",) in names

    # test WHERE miss
    names = _get_geospatial_names_from_filter(
        db=db,
        value=Value(
            type=SupportedType.POINT,
            value=[-11, -11],
        ),
        operator=FilterOperator.INTERSECTS,
        model_object=model_object,
        symbol=symbol,
    )
    assert len(names) == 0

    # test outside
    names = _get_geospatial_names_from_filter(
        db=db,
        value=Value(
            type=SupportedType.POINT,
            value=[-11, -11],
        ),
        operator=FilterOperator.OUTSIDE,
        model_object=model_object,
        symbol=symbol,
        label_source=models.Prediction,
    )
    assert len(names) == 2
    assert ("model1",) in names
    assert ("model2",) in names


@pytest.fixture
def datetime_metadata() -> list[schemas.DateTime]:
    """List of datetimes using different formats."""
    return [
        schemas.DateTime(value="2022-01-01"),
        schemas.DateTime(
            value="2023-04-07T16:34:56",
        ),
        schemas.DateTime(value="2023-04-07T16:35:56"),
        schemas.DateTime(value="2023-11-12"),
        schemas.DateTime(value="2023-12-04T00:05:23+04:00"),
    ]


@pytest.fixture
def date_metadata() -> list[schemas.Date]:
    """List of dates using different formats."""
    return [
        schemas.Date(
            value="2022-01-01",
        ),
        schemas.Date(
            value="2023-04-07",
        ),
        schemas.Date(value="2023-04-08"),
        schemas.Date(
            value="2023-11-12",
        ),
        schemas.Date(
            value="2023-12-04",
        ),
    ]


@pytest.fixture
def time_metadata() -> list[schemas.Time]:
    """List of times using different formats."""
    return [
        schemas.Time(
            value="00:05:23",
        ),
        schemas.Time(
            value="16:34:56",
        ),
        schemas.Time(value="16:35:56.000283"),
        schemas.Time(
            value="18:02:23",
        ),
        schemas.Time(
            value="22:05:23",
        ),
    ]


@pytest.fixture
def duration_metadata() -> list[schemas.Duration]:
    """List of time durations using different formats."""
    return [
        schemas.Duration(
            value=0.0001,
        ),
        schemas.Duration(
            value=324.01,
        ),
        schemas.Duration(value=324.02),
        schemas.Duration(
            value=180223.0,
        ),
        schemas.Duration(
            value=220523.0,
        ),
    ]


def time_filter(
    db: Session,
    symbol: Symbol,
    type_: SupportedType,
    key: str,
    value: str | float,
    op: str,
):
    match op:
        case "==":
            op = FilterOperator.EQ
        case "!=":
            op = FilterOperator.NE
        case ">":
            op = FilterOperator.GT
        case ">=":
            op = FilterOperator.GTE
        case "<":
            op = FilterOperator.LT
        case "<=":
            op = FilterOperator.LTE
        case _:
            raise NotImplementedError
    
    f = Condition(
        lhs=symbol,
        lhs_key=key,
        rhs=Value(type=type_, value=value),
        op=op,
    )

    match symbol:
        case Symbol.DATASET_META:
            f = Filter(datasets=f)
            return generate_query(
                models.Dataset,
                db=db,
                filters=f,
                label_source=models.GroundTruth,
            ).all()
        case Symbol.MODEL_META:
            f = Filter(models=f)
            return generate_query(
                models.Model,
                db=db,
                filters=f,
                label_source=models.Prediction,
            ).all()
        case Symbol.DATUM_META:
            f = Filter(datums=f)
            return generate_query(
                models.Datum,
                db=db,
                filters=f,
                label_source=models.GroundTruth,
            ).all()
        case Symbol.ANNOTATION_META:
            f = Filter(annotations=f)
            return generate_query(
                models.Annotation,
                db=db,
                filters=f,
                label_source=models.GroundTruth,
            ).all()
        case _:
            raise NotImplementedError(symbol)


def _test_datetime_query(
    db: Session,
    symbol: Symbol,
    type_: SupportedType,
    key: str,
    metadata_: Sequence[
        schemas.DateTime | schemas.Date | schemas.Time | schemas.Duration
    ],
):
    """
    The metadata_ param is a pytest fixture containing sequential timestamps.
    """

    # Check equality operator
    op = "=="

    results = time_filter(
        db=db,
        symbol=symbol,
        type_=type_,
        key=key,
        value=metadata_[0].value,
        op=op,
    )
    assert len(results) == 0

    results = time_filter(
        db=db,
        symbol=symbol,
        type_=type_,
        key=key,
        value=metadata_[1].value,
        op=op,
    )
    assert len(results) == 1

    results = time_filter(
        db=db,
        symbol=symbol,
        type_=type_,
        key=key,
        value=metadata_[2].value,
        op=op,
    )
    assert len(results) == 0

    results = time_filter(
        db=db,
        symbol=symbol,
        type_=type_,
        key=key,
        value=metadata_[3].value,
        op=op,
    )
    assert len(results) == 1

    results = time_filter(
        db=db,
        symbol=symbol,
        type_=type_,
        key=key,
        value=metadata_[4].value,
        op=op,
    )
    assert len(results) == 0

    # Check inequality operator
    op = "!="

    results = time_filter(
        db=db,
        symbol=symbol,
        type_=type_,
        key=key,
        value=metadata_[0].value,
        op=op,
    )
    assert len(results) == 2

    results = time_filter(
        db=db,
        symbol=symbol,
        type_=type_,
        key=key,
        value=metadata_[1].value,
        op=op,
    )
    assert len(results) == 1

    results = time_filter(
        db=db,
        symbol=symbol,
        type_=type_,
        key=key,
        value=metadata_[2].value,
        op=op,
    )
    assert len(results) == 2

    results = time_filter(
        db=db,
        symbol=symbol,
        type_=type_,
        key=key,
        value=metadata_[3].value,
        op=op,
    )
    assert len(results) == 1

    results = time_filter(
        db=db,
        symbol=symbol,
        type_=type_,
        key=key,
        value=metadata_[4].value,
        op=op,
    )
    assert len(results) == 2

    # Check less-than operator
    op = "<"

    results = time_filter(
        db=db,
        symbol=symbol,
        type_=type_,
        key=key,
        value=metadata_[0].value,
        op=op,
    )
    assert len(results) == 0

    results = time_filter(
        db=db,
        symbol=symbol,
        type_=type_,
        key=key,
        value=metadata_[1].value,
        op=op,
    )
    assert len(results) == 0

    results = time_filter(
        db=db,
        symbol=symbol,
        type_=type_,
        key=key,
        value=metadata_[2].value,
        op=op,
    )
    assert len(results) == 1

    results = time_filter(
        db=db,
        symbol=symbol,
        type_=type_,
        key=key,
        value=metadata_[3].value,
        op=op,
    )
    assert len(results) == 1

    results = time_filter(
        db=db,
        symbol=symbol,
        type_=type_,
        key=key,
        value=metadata_[4].value,
        op=op,
    )
    assert len(results) == 2

    # Check greater-than operator
    op = ">"

    results = time_filter(
        db=db,
        symbol=symbol,
        type_=type_,
        key=key,
        value=metadata_[0].value,
        op=op,
    )
    assert len(results) == 2

    results = time_filter(
        db=db,
        symbol=symbol,
        type_=type_,
        key=key,
        value=metadata_[1].value,
        op=op,
    )
    assert len(results) == 1

    results = time_filter(
        db=db,
        symbol=symbol,
        type_=type_,
        key=key,
        value=metadata_[2].value,
        op=op,
    )
    assert len(results) == 1

    results = time_filter(
        db=db,
        symbol=symbol,
        type_=type_,
        key=key,
        value=metadata_[3].value,
        op=op,
    )
    assert len(results) == 0

    results = time_filter(
        db=db,
        symbol=symbol,
        type_=type_,
        key=key,
        value=metadata_[4].value,
        op=op,
    )
    assert len(results) == 0

    # Check less-than or equal operator
    op = "<="

    results = time_filter(
        db=db,
        symbol=symbol,
        type_=type_,
        key=key,
        value=metadata_[0].value,
        op=op,
    )
    assert len(results) == 0

    results = time_filter(
        db=db,
        symbol=symbol,
        type_=type_,
        key=key,
        value=metadata_[1].value,
        op=op,
    )
    assert len(results) == 1

    results = time_filter(
        db=db,
        symbol=symbol,
        type_=type_,
        key=key,
        value=metadata_[2].value,
        op=op,
    )
    assert len(results) == 1

    results = time_filter(
        db=db,
        symbol=symbol,
        type_=type_,
        key=key,
        value=metadata_[3].value,
        op=op,
    )
    assert len(results) == 2

    results = time_filter(
        db=db,
        symbol=symbol,
        type_=type_,
        key=key,
        value=metadata_[4].value,
        op=op,
    )
    assert len(results) == 2

    # Check greater-than or equal operator
    op = ">="

    results = time_filter(
        db=db,
        symbol=symbol,
        type_=type_,
        key=key,
        value=metadata_[0].value,
        op=op,
    )
    assert len(results) == 2

    results = time_filter(
        db=db,
        symbol=symbol,
        type_=type_,
        key=key,
        value=metadata_[1].value,
        op=op,
    )
    assert len(results) == 2

    results = time_filter(
        db=db,
        symbol=symbol,
        type_=type_,
        key=key,
        value=metadata_[2].value,
        op=op,
    )
    assert len(results) == 1

    results = time_filter(
        db=db,
        symbol=symbol,
        type_=type_,
        key=key,
        value=metadata_[3].value,
        op=op,
    )
    assert len(results) == 1

    results = time_filter(
        db=db,
        symbol=symbol,
        type_=type_,
        key=key,
        value=metadata_[4].value,
        op=op,
    )
    assert len(results) == 0


def test_dataset_datetime_queries(
    db: Session,
    datetime_metadata: list[schemas.DateTime],
    date_metadata: list[schemas.Date],
    time_metadata: list[schemas.Time],
    duration_metadata: list[schemas.Duration],
):
    datetime_key = "maybe_i_was_created_at_this_time"
    date_key = "idk_some_other_date"
    time_key = "a_third_key"
    duration_key = "some_duration"

    crud.create_dataset(
        db=db,
        dataset=schemas.Dataset(
            name="dataset1",
            metadata={
                datetime_key: {
                    "type": "datetime",
                    "value": datetime_metadata[1].value,
                },
                date_key: {"type": "date", "value": date_metadata[1].value},
                time_key: {"type": "time", "value": time_metadata[1].value},
                duration_key: {
                    "type": "duration",
                    "value": duration_metadata[1].value,
                },
            },
        ),
    )
    crud.create_dataset(
        db=db,
        dataset=schemas.Dataset(
            name="dataset2",
            metadata={
                datetime_key: {
                    "type": "datetime",
                    "value": datetime_metadata[3].value,
                },
                date_key: {"type": "date", "value": date_metadata[3].value},
                time_key: {"type": "time", "value": time_metadata[3].value},
                duration_key: {
                    "type": "duration",
                    "value": duration_metadata[3].value,
                },
            },
        ),
    )

    _test_datetime_query(
        db, Symbol.DATASET_META, SupportedType.DATETIME, datetime_key, datetime_metadata
    )
    _test_datetime_query(
        db, Symbol.DATASET_META, SupportedType.DATE, date_key, date_metadata
    )
    _test_datetime_query(
        db, Symbol.DATASET_META, SupportedType.TIME, time_key, time_metadata
    )
    _test_datetime_query(
        db, Symbol.DATASET_META, SupportedType.DURATION, duration_key, duration_metadata
    )


def test_model_datetime_queries(
    db: Session,
    datetime_metadata: list[schemas.DateTime],
    date_metadata: list[schemas.Date],
    time_metadata: list[schemas.Time],
    duration_metadata: list[schemas.Duration],
):
    datetime_key = "maybe_i_was_created_at_this_time"
    date_key = "idk_some_other_date"
    time_key = "a_third_key"
    duration_key = "some_duration"

    crud.create_model(
        db=db,
        model=schemas.Model(
            name="model1",
            metadata={
                datetime_key: {
                    "type": "datetime",
                    "value": datetime_metadata[1].value,
                },
                date_key: {"type": "date", "value": date_metadata[1].value},
                time_key: {"type": "time", "value": time_metadata[1].value},
                duration_key: {
                    "type": "duration",
                    "value": duration_metadata[1].value,
                },
            },
        ),
    )
    crud.create_model(
        db=db,
        model=schemas.Model(
            name="model2",
            metadata={
                datetime_key: {
                    "type": "datetime",
                    "value": datetime_metadata[3].value,
                },
                date_key: {"type": "date", "value": date_metadata[3].value},
                time_key: {"type": "time", "value": time_metadata[3].value},
                duration_key: {
                    "type": "duration",
                    "value": duration_metadata[3].value,
                },
            },
        ),
    )

    _test_datetime_query(
        db, Symbol.MODEL_META, SupportedType.DATETIME, datetime_key, datetime_metadata
    )
    _test_datetime_query(db, Symbol.MODEL_META, SupportedType.DATE, date_key, date_metadata)
    _test_datetime_query(db, Symbol.MODEL_META, SupportedType.TIME, time_key, time_metadata)
    _test_datetime_query(
        db, Symbol.MODEL_META, SupportedType.DURATION, duration_key, duration_metadata
    )


def test_datum_datetime_queries(
    db: Session,
    datum_1,
    datum_2,
    datum_3,
    datum_4,
    datetime_metadata: list[schemas.DateTime],
    date_metadata: list[schemas.Date],
    time_metadata: list[schemas.Time],
    duration_metadata: list[schemas.Duration],
):
    datetime_key = "maybe_i_was_created_at_this_time"
    date_key = "idk_some_other_date"
    time_key = "a_third_key"
    duration_key = "some_duration"

    def add_metadata_typing(value):
        return {"type": type(value).__name__.lower(), "value": value.value}

    datum_1.metadata[datetime_key] = add_metadata_typing(datetime_metadata[1])
    datum_4.metadata[datetime_key] = add_metadata_typing(datetime_metadata[3])

    datum_1.metadata[date_key] = add_metadata_typing(date_metadata[1])
    datum_4.metadata[date_key] = add_metadata_typing(date_metadata[3])

    datum_1.metadata[time_key] = add_metadata_typing(time_metadata[1])
    datum_4.metadata[time_key] = add_metadata_typing(time_metadata[3])

    datum_1.metadata[duration_key] = add_metadata_typing(duration_metadata[1])
    datum_4.metadata[duration_key] = add_metadata_typing(duration_metadata[3])

    annotation = schemas.Annotation(
        labels=[schemas.Label(key="k1", value="v1")],
    )

    crud.create_dataset(
        db=db,
        dataset=schemas.Dataset(
            name=dset_name,
        ),
    )

    crud.create_groundtruths(
        db=db,
        groundtruths=[
            schemas.GroundTruth(
                dataset_name=dset_name, datum=datum_1, annotations=[annotation]
            ),
            schemas.GroundTruth(
                dataset_name=dset_name, datum=datum_4, annotations=[annotation]
            ),
        ],
    )

    crud.create_model(
        db=db,
        model=schemas.Model(
            name="model1",
            metadata={
                datetime_key: {
                    "type": "datetime",
                    "value": datetime_metadata[1].value,
                },
                date_key: {"type": "date", "value": date_metadata[1].value},
                time_key: {"type": "time", "value": time_metadata[1].value},
            },
        ),
    )
    crud.create_model(
        db=db,
        model=schemas.Model(
            name="model2",
            metadata={
                datetime_key: {
                    "type": "datetime",
                    "value": datetime_metadata[3].value,
                },
                date_key: {"type": "date", "value": date_metadata[3].value},
                time_key: {"type": "time", "value": time_metadata[3].value},
            },
        ),
    )

    _test_datetime_query(
        db, Symbol.DATUM_META, SupportedType.DATETIME, datetime_key, datetime_metadata
    )
    _test_datetime_query(db, Symbol.DATUM_META, SupportedType.DATE, date_key, date_metadata)
    _test_datetime_query(db, Symbol.DATUM_META, SupportedType.TIME, time_key, time_metadata)
    _test_datetime_query(
        db, Symbol.DATUM_META, SupportedType.DURATION, duration_key, duration_metadata
    )


def test_annotation_datetime_queries(
    db: Session,
    datum_1,
    datetime_metadata: list[schemas.DateTime],
    date_metadata: list[schemas.Date],
    time_metadata: list[schemas.Time],
    duration_metadata: list[schemas.Duration],
):
    datetime_key = "maybe_i_was_created_at_this_time"
    date_key = "idk_some_other_date"
    time_key = "a_third_key"
    duration_key = "some_duration"

    annotation_1 = schemas.Annotation(
        labels=[schemas.Label(key="k1", value="v1")],
        metadata={
            datetime_key: {
                "type": "datetime",
                "value": datetime_metadata[1].value,
            },
            date_key: {"type": "date", "value": date_metadata[1].value},
            time_key: {"type": "time", "value": time_metadata[1].value},
            duration_key: {
                "type": "duration",
                "value": duration_metadata[1].value,
            },
        },
    )
    annotation_4 = schemas.Annotation(
        labels=[schemas.Label(key="k4", value="v4")],
        metadata={
            datetime_key: {
                "type": "datetime",
                "value": datetime_metadata[3].value,
            },
            date_key: {"type": "date", "value": date_metadata[3].value},
            time_key: {"type": "time", "value": time_metadata[3].value},
            duration_key: {
                "type": "duration",
                "value": duration_metadata[3].value,
            },
        },
    )

    crud.create_dataset(
        db=db,
        dataset=schemas.Dataset(
            name=dset_name,
        ),
    )

    crud.create_groundtruths(
        db=db,
        groundtruths=[
            schemas.GroundTruth(
                dataset_name=dset_name,
                datum=datum_1,
                annotations=[
                    annotation_1,
                    annotation_4,
                ],
            )
        ],
    )

    _test_datetime_query(
        db, Symbol.ANNOTATION_META, SupportedType.DATETIME, datetime_key, datetime_metadata
    )
    _test_datetime_query(
        db, Symbol.ANNOTATION_META, SupportedType.DATE, date_key, date_metadata
    )
    _test_datetime_query(
        db, Symbol.ANNOTATION_META, SupportedType.TIME, time_key, time_metadata
    )
    _test_datetime_query(
        db, Symbol.ANNOTATION_META, SupportedType.DURATION, duration_key, duration_metadata
    )


def test_query_expression_types(
    db: Session,
    model_sim,
):
    cat_filter = Filter(
        groundtruths=create_label_filter(key="class", value="cat")
    )

    # Test `distinct`
    dataset_names = (
        generate_query(
            models.Dataset.name,
            db=db,
            filters=cat_filter,
            label_source=models.GroundTruth,
        )
        .distinct()
        .all()
    )
    assert len(dataset_names) == 1
    assert (dset_name,) in dataset_names

    # Test `func.count`, note this returns 10 b/c of joins.
    count = (
        generate_query(
            func.count(models.Dataset.name),
            db=db,
            filters=cat_filter,
            label_source=models.GroundTruth,
        )
        .distinct()
        .scalar()
    )
    assert count == 10

    # Test `func.count` with nested distinct.
    count = (
        generate_query(
            func.count(distinct(models.Dataset.name)),
            db=db,
            filters=cat_filter,
            label_source=models.GroundTruth,
        )
        .distinct()
        .scalar()
    )
    assert count == 1

    # Test nested functions
    max_area = generate_query(
        func.max(func.ST_Area(models.Annotation.box)),
        db=db,
        filters=cat_filter,
        label_source=models.GroundTruth,
    ).scalar()
    assert max_area == 100.0
