import numpy
import pytest
from sqlalchemy.orm import Session

from velour_api import crud, enums, schemas
from velour_api.backend import Query, models
from velour_api.enums import TaskType

dset_name = "dataset1"
model_name1 = "model1"
model_name2 = "model2"
datum_uid1 = "uid1"
datum_uid2 = "uid2"
datum_uid3 = "uid3"
datum_uid4 = "uid4"


@pytest.fixture
def metadata_1() -> dict[str, int | float | str]:
    return {
        "some_numeric_attribute": 0.4,
        "some_str_attribute": "abc",
        "height": 10,
        "width": 10,
        "some_bool_attribute": True,
    }


@pytest.fixture
def metadata_2() -> dict[str, int | float | str]:
    return {
        "some_numeric_attribute": 0.6,
        "some_str_attribute": "abc",
        "height": 10,
        "width": 10,
        "some_bool_attribute": False,
    }


@pytest.fixture
def metadata_3() -> dict[str, int | float | str]:
    return {
        "some_numeric_attribute": 0.4,
        "some_str_attribute": "xyz",
        "height": 10,
        "width": 10,
        "some_bool_attribute": True,
    }


@pytest.fixture
def metadata_4() -> dict[str, int | float | str]:
    return {
        "some_numeric_attribute": 0.6,
        "some_str_attribute": "xyz",
        "height": 10,
        "width": 10,
        "some_bool_attribute": False,
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
def geospatial_coordinates() -> (
    dict[
        str,
        schemas.GeoJSONPoint
        | schemas.GeoJSONPolygon
        | schemas.GeoJSONMultiPolygon,
    ]
):
    return {
        "point": {"type": "Point", "coordinates": [125.2750725, 38.760525]},
        "polygon1": {
            "type": "Polygon",
            "coordinates": [
                [
                    [-10, -10],
                    [10, -10],
                    [10, 10],
                    [-10, 10],
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
                ]
            ],
        },
        "multipolygon": {
            "type": "MultiPolygon",
            "coordinates": [
                [
                    [
                        [50, 50],
                        [70, 50],
                        [70, 70],
                        [50, 70],
                    ],
                    [
                        [30, 30],
                        [35, 30],
                        [35, 35],
                        [30, 35],
                    ],
                ],
                [
                    [
                        [10, 10],
                        [20, 10],
                        [20, 20],
                        [10, 20],
                    ],
                ],
            ],
        },
    }


@pytest.fixture
def datum_1(metadata_1, geospatial_coordinates) -> schemas.Datum:
    return schemas.Datum(
        uid=datum_uid1,
        dataset=dset_name,
        metadata=metadata_1,
        geospatial=geospatial_coordinates["polygon1"],
    )


@pytest.fixture
def datum_2(metadata_2, geospatial_coordinates) -> schemas.Datum:
    return schemas.Datum(
        uid=datum_uid2,
        dataset=dset_name,
        metadata=metadata_2,
        geospatial=geospatial_coordinates["multipolygon"],
    )


@pytest.fixture
def datum_3(metadata_3, geospatial_coordinates) -> schemas.Datum:
    return schemas.Datum(
        uid=datum_uid3,
        dataset=dset_name,
        metadata=metadata_3,
        geospatial=geospatial_coordinates["polygon2"],
    )


@pytest.fixture
def datum_4(metadata_4, geospatial_coordinates) -> schemas.Datum:
    return schemas.Datum(
        uid=datum_uid4,
        dataset=dset_name,
        metadata=metadata_4,
        geospatial=geospatial_coordinates["polygon3"],
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
            task_type=TaskType.CLASSIFICATION,
            labels=[label_cat],
        ),
        schemas.Annotation(
            task_type=TaskType.DETECTION,
            labels=[label_cat],
            bounding_box=schemas.BoundingBox.from_extrema(0, 0, 10, 10),
            metadata=metadata_1,
        ),
        schemas.Annotation(
            task_type=TaskType.DETECTION,
            labels=[label_cat],
            bounding_box=schemas.BoundingBox.from_extrema(0, 0, 1, 50),
            metadata=metadata_2,
        ),
        schemas.Annotation(
            task_type=TaskType.DETECTION,
            labels=[label_cat],
            raster=raster_1,
            metadata=metadata_1,
        ),
        schemas.Annotation(
            task_type=TaskType.DETECTION,
            labels=[label_cat],
            raster=raster_2,
            metadata=metadata_2,
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
            task_type=TaskType.CLASSIFICATION,
            labels=[label_dog],
        ),
        schemas.Annotation(
            task_type=TaskType.DETECTION,
            labels=[label_dog],
            bounding_box=schemas.BoundingBox.from_extrema(0, 0, 10, 10),
            metadata=metadata_3,
        ),
        schemas.Annotation(
            task_type=TaskType.DETECTION,
            labels=[label_dog],
            bounding_box=schemas.BoundingBox.from_extrema(0, 0, 1, 50),
            metadata=metadata_4,
        ),
        schemas.Annotation(
            task_type=TaskType.DETECTION,
            labels=[label_dog],
            raster=raster_1,
            metadata=metadata_3,
        ),
        schemas.Annotation(
            task_type=TaskType.DETECTION,
            labels=[label_dog],
            raster=raster_2,
            metadata=metadata_4,
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
            task_type=TaskType.CLASSIFICATION,
            labels=[
                schemas.Label(key="class", value="cat", score=0.9),
                schemas.Label(key="class", value="dog", score=0.1),
            ],
        ),
        schemas.Annotation(
            task_type=TaskType.DETECTION,
            labels=[
                schemas.Label(key="class", value="cat", score=0.8),
                schemas.Label(key="class", value="dog", score=0.2),
            ],
            bounding_box=schemas.BoundingBox.from_extrema(0, 0, 10, 10),
            metadata=metadata_1,
        ),
        schemas.Annotation(
            task_type=TaskType.DETECTION,
            labels=[
                schemas.Label(key="class", value="cat", score=0.7),
                schemas.Label(key="class", value="dog", score=0.3),
            ],
            bounding_box=schemas.BoundingBox.from_extrema(0, 0, 1, 50),
            metadata=metadata_2,
        ),
        schemas.Annotation(
            task_type=TaskType.DETECTION,
            labels=[
                schemas.Label(key="class", value="cat", score=0.75),
                schemas.Label(key="class", value="dog", score=0.25),
            ],
            raster=raster_1,
            metadata=metadata_1,
        ),
        schemas.Annotation(
            task_type=TaskType.DETECTION,
            labels=[
                schemas.Label(key="class", value="cat", score=0.95),
                schemas.Label(key="class", value="dog", score=0.05),
            ],
            raster=raster_2,
            metadata=metadata_2,
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
            task_type=TaskType.CLASSIFICATION,
            labels=[
                schemas.Label(key="class", value="cat", score=0.1),
                schemas.Label(key="class", value="dog", score=0.9),
            ],
        ),
        schemas.Annotation(
            task_type=TaskType.DETECTION,
            labels=[
                schemas.Label(key="class", value="cat", score=0.2),
                schemas.Label(key="class", value="dog", score=0.8),
            ],
            bounding_box=schemas.BoundingBox.from_extrema(0, 0, 10, 10),
            metadata=metadata_3,
        ),
        schemas.Annotation(
            task_type=TaskType.DETECTION,
            labels=[
                schemas.Label(key="class", value="cat", score=0.3),
                schemas.Label(key="class", value="dog", score=0.7),
            ],
            bounding_box=schemas.BoundingBox.from_extrema(0, 0, 1, 50),
            metadata=metadata_4,
        ),
        schemas.Annotation(
            task_type=TaskType.DETECTION,
            labels=[
                schemas.Label(key="class", value="cat", score=0.25),
                schemas.Label(key="class", value="dog", score=0.75),
            ],
            raster=raster_1,
            metadata=metadata_3,
        ),
        schemas.Annotation(
            task_type=TaskType.DETECTION,
            labels=[
                schemas.Label(key="class", value="cat", score=0.05),
                schemas.Label(key="class", value="dog", score=0.95),
            ],
            raster=raster_2,
            metadata=metadata_4,
        ),
    ]


@pytest.fixture
def groundtruth_cat_datum_1(
    datum_1,
    groundtruth_annotations_cat,
) -> schemas.GroundTruth:
    return schemas.GroundTruth(
        datum=datum_1,
        annotations=groundtruth_annotations_cat,
    )


@pytest.fixture
def groundtruth_cat_datum_2(
    datum_2,
    groundtruth_annotations_cat,
) -> schemas.GroundTruth:
    return schemas.GroundTruth(
        datum=datum_2,
        annotations=groundtruth_annotations_cat,
    )


@pytest.fixture
def groundtruth_dog_datum_3(
    datum_3,
    groundtruth_annotations_dog,
) -> schemas.GroundTruth:
    return schemas.GroundTruth(
        datum=datum_3,
        annotations=groundtruth_annotations_dog,
    )


@pytest.fixture
def groundtruth_dog_datum_4(
    datum_4,
    groundtruth_annotations_dog,
) -> schemas.GroundTruth:
    return schemas.GroundTruth(
        datum=datum_4,
        annotations=groundtruth_annotations_dog,
    )


@pytest.fixture
def prediction_cat_datum1_model1(
    datum_1,
    prediction_annotations_cat,
) -> schemas.Prediction:
    return schemas.Prediction(
        model=model_name1,
        datum=datum_1,
        annotations=prediction_annotations_cat,
    )


@pytest.fixture
def prediction_cat_datum2_model1(
    datum_2,
    prediction_annotations_cat,
) -> schemas.Prediction:
    return schemas.Prediction(
        model=model_name1,
        datum=datum_2,
        annotations=prediction_annotations_cat,
    )


@pytest.fixture
def prediction_dog_datum3_model1(
    datum_3,
    prediction_annotations_dog,
) -> schemas.Prediction:
    return schemas.Prediction(
        model=model_name1,
        datum=datum_3,
        annotations=prediction_annotations_dog,
    )


@pytest.fixture
def prediction_dog_datum4_model1(
    datum_4,
    prediction_annotations_dog,
) -> schemas.Prediction:
    return schemas.Prediction(
        model=model_name1,
        datum=datum_4,
        annotations=prediction_annotations_dog,
    )


@pytest.fixture
def prediction_dog_datum1_model2(
    datum_1,
    prediction_annotations_dog,
) -> schemas.Prediction:
    return schemas.Prediction(
        model=model_name2,
        datum=datum_1,
        annotations=prediction_annotations_dog,
    )


@pytest.fixture
def prediction_dog_datum2_model2(
    datum_2,
    prediction_annotations_dog,
) -> schemas.Prediction:
    return schemas.Prediction(
        model=model_name2,
        datum=datum_2,
        annotations=prediction_annotations_dog,
    )


@pytest.fixture
def prediction_cat_datum3_model2(
    datum_3,
    prediction_annotations_cat,
) -> schemas.Prediction:
    return schemas.Prediction(
        model=model_name2,
        datum=datum_3,
        annotations=prediction_annotations_cat,
    )


@pytest.fixture
def prediction_cat_datum4_model2(
    datum_4,
    prediction_annotations_cat,
) -> schemas.Prediction:
    return schemas.Prediction(
        model=model_name2,
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
    geospatial_coordinates,
):
    crud.create_dataset(
        db=db,
        dataset=schemas.Dataset(
            name=dset_name,
            metadata=metadata_1,
            geospatial=geospatial_coordinates["polygon1"],
        ),
    )
    crud.create_groundtruth(db=db, groundtruth=groundtruth_cat_datum_1)
    crud.create_groundtruth(db=db, groundtruth=groundtruth_cat_datum_2)
    crud.create_groundtruth(db=db, groundtruth=groundtruth_dog_datum_3)
    crud.create_groundtruth(db=db, groundtruth=groundtruth_dog_datum_4)
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
    geospatial_coordinates,
):
    crud.create_model(
        db=db,
        model=schemas.Model(
            name=model_name1,
            metadata=metadata_1,
            geospatial=geospatial_coordinates["polygon1"],
        ),
    )
    crud.create_prediction(db=db, prediction=prediction_cat_datum1_model1)
    crud.create_prediction(db=db, prediction=prediction_cat_datum2_model1)
    crud.create_prediction(db=db, prediction=prediction_dog_datum3_model1)
    crud.create_prediction(db=db, prediction=prediction_dog_datum4_model1)
    crud.finalize(db=db, dataset_name=dset_name, model_name=model_name1)

    crud.create_model(
        db=db,
        model=schemas.Model(
            name=model_name2,
            metadata=metadata_4,
        ),
    )
    crud.create_prediction(db=db, prediction=prediction_dog_datum1_model2)
    crud.create_prediction(db=db, prediction=prediction_dog_datum2_model2)
    crud.create_prediction(db=db, prediction=prediction_cat_datum3_model2)
    crud.create_prediction(db=db, prediction=prediction_cat_datum4_model2)
    crud.finalize(db=db, dataset_name=dset_name, model_name=model_name2)


def test_query_datasets(
    db: Session,
    model_sim,
):
    # Check that passing a non-InstrumentedAttribute returns None
    q = Query("not_a_valid_attribute")
    assert len(q._selected) == 0

    # Q: Get names for datasets where label class=cat exists in groundtruths.
    f = schemas.Filter(labels=[{"class": "cat"}])
    query_obj = Query(models.Dataset.name)
    assert len(query_obj._selected) == 1

    q = query_obj.filter(f).groundtruths()
    dataset_names = db.query(q).distinct().all()
    assert len(dataset_names) == 1
    assert (dset_name,) in dataset_names

    # Q: Get names for datasets where label=tree exists in groundtruths
    f = schemas.Filter(labels=[{"class": "tree"}])
    q = Query(models.Dataset.name).filter(f).any()
    dataset_names = db.query(q).distinct().all()
    assert len(dataset_names) == 0


def test_query_models(
    db: Session,
    model_sim,
):
    # Q: Get names for all models that operate over a dataset that meets the name equality
    f = schemas.Filter(
        dataset_names=[dset_name],
    )
    q = Query(models.Model.name).filter(f).any()
    model_names = db.query(q).distinct().all()
    assert len(model_names) == 2
    assert (model_name1,) in model_names
    assert (model_name2,) in model_names

    # Q: Get names for models where label=cat exists in predictions
    f = schemas.Filter(labels=[{"class": "cat"}])
    q = Query(models.Model.name).filter(f).any()
    model_names = db.query(q).distinct().all()
    assert len(model_names) == 2
    assert (model_name1,) in model_names
    assert (model_name2,) in model_names

    # Q: Get names for models where label=tree exists in predictions
    f = schemas.Filter(labels=[{"class": "tree"}])
    q = Query(models.Model.name).filter(f).any()
    model_names = db.query(q).distinct().all()
    assert len(model_names) == 0

    # Q: Get names for models that operate over dataset.
    f = schemas.Filter(dataset_names=[dset_name])
    q = Query(models.Model.name).filter(f).any()
    model_names = db.query(q).distinct().all()
    assert len(model_names) == 2
    assert (model_name1,) in model_names
    assert (model_name2,) in model_names

    # Q: Get names for models that operate over dataset that doesn't exist.
    f = schemas.Filter(dataset_names=["invalid"])
    q = Query(models.Model.name).filter(f).any()
    model_names = db.query(q).distinct().all()
    assert len(model_names) == 0

    # Q: Get models with metadatum with `numeric` > 0.5.
    f = schemas.Filter(
        model_metadata={
            "some_numeric_attribute": [
                schemas.NumericFilter(
                    value=0.5,
                    operator=">",
                ),
            ]
        }
    )
    q = Query(models.Model.name).filter(f).any()
    model_names = db.query(q).distinct().all()
    assert len(model_names) == 1
    assert (model_name2,) in model_names

    # Q: Get models with metadatum with `numeric` < 0.5.
    f = schemas.Filter(
        model_metadata={
            "some_numeric_attribute": [
                schemas.NumericFilter(
                    value=0.5,
                    operator="<",
                ),
            ]
        }
    )
    q = Query(models.Model.name).filter(f).any()
    model_names = db.query(q).distinct().all()
    assert len(model_names) == 1
    assert (model_name1,) in model_names


def test_query_by_metadata(
    db: Session,
    model_sim,
):
    # Q: Get datums with metadatum with `numeric` < 0.5, `str` == 'abc', and `bool` == True.
    f = schemas.Filter(
        datum_metadata={
            "some_numeric_attribute": [
                schemas.NumericFilter(
                    value=0.5,
                    operator="<",
                ),
            ],
            "some_str_attribute": [
                schemas.StringFilter(
                    value="abc",
                    operator="==",
                ),
            ],
            "some_bool_attribute": [
                schemas.BooleanFilter(
                    value=True,
                    operator="==",
                )
            ],
        }
    )
    q = Query(models.Datum.uid).filter(f).any()
    datum_uids = db.query(q).distinct().all()
    assert len(datum_uids) == 1
    assert (datum_uid1,) in datum_uids

    # repeat with `bool` == False or != `True` and check we get nothing
    for val, op in ([False, "=="], [True, "!="]):
        f = schemas.Filter(
            datum_metadata={
                "some_numeric_attribute": [
                    schemas.NumericFilter(
                        value=0.5,
                        operator="<",
                    ),
                ],
                "some_str_attribute": [
                    schemas.StringFilter(
                        value="abc",
                        operator="==",
                    ),
                ],
                "some_bool_attribute": [
                    schemas.BooleanFilter(
                        value=val,
                        operator=op,
                    )
                ],
            }
        )
        q = Query(models.Datum.uid).filter(f).any()
        datum_uids = db.query(q).distinct().all()
        assert len(datum_uids) == 0

    # Q: Get datums with metadatum with `numeric` > 0.5 and `str` == 'abc'.
    f = schemas.Filter(
        datum_metadata={
            "some_numeric_attribute": [
                schemas.NumericFilter(
                    value=0.5,
                    operator=">",
                ),
            ],
            "some_str_attribute": [
                schemas.StringFilter(
                    value="abc",
                    operator="==",
                ),
            ],
        }
    )
    q = Query(models.Datum.uid).filter(f).any()
    datum_uids = db.query(q).distinct().all()
    assert len(datum_uids) == 1
    assert (datum_uid2,) in datum_uids

    # Q: Get datums with metadatum with `numeric` < 0.5 and `str` == 'xyz'.
    f = schemas.Filter(
        datum_metadata={
            "some_numeric_attribute": [
                schemas.NumericFilter(
                    value=0.5,
                    operator="<",
                ),
            ],
            "some_str_attribute": [
                schemas.StringFilter(
                    value="xyz",
                    operator="==",
                ),
            ],
        }
    )
    q = Query(models.Datum.uid).filter(f).any()
    datum_uids = db.query(q).distinct().all()
    assert len(datum_uids) == 1
    assert (datum_uid3,) in datum_uids

    # Q: Get models with metadatum with `numeric` > 0.5 and `str` == 'xyz'.
    f = schemas.Filter(
        datum_metadata={
            "some_numeric_attribute": [
                schemas.NumericFilter(
                    value=0.5,
                    operator=">",
                ),
            ],
            "some_str_attribute": [
                schemas.StringFilter(
                    value="xyz",
                    operator="==",
                ),
            ],
        }
    )
    q = Query(models.Datum.uid).filter(f).any()
    datum_uids = db.query(q).distinct().all()
    assert len(datum_uids) == 1
    assert (datum_uid4,) in datum_uids


def test_query_datums(
    db: Session,
    model_sim,
):
    # Q: Get datums with groundtruth labels of "cat"
    f = schemas.Filter(labels=[{"class": "cat"}])
    q = Query(models.Datum.uid).filter(f).groundtruths()
    datum_uids = db.query(q).distinct().all()
    assert len(datum_uids) == 2
    assert (datum_uid1,) in datum_uids
    assert (datum_uid2,) in datum_uids

    # Q: Get datums with groundtruth labels of "dog"
    f = schemas.Filter(labels=[{"class": "dog"}])
    q = Query(models.Datum.uid).filter(f).groundtruths()
    datum_uids = db.query(q).distinct().all()
    assert len(datum_uids) == 2
    assert (datum_uid3,) in datum_uids
    assert (datum_uid4,) in datum_uids

    # Q: Get datums with prediction labels of "cat"
    f = schemas.Filter(labels=[{"class": "cat"}])
    q = Query(models.Datum.uid).filter(f).predictions()
    datum_uids = db.query(q).distinct().all()
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
    f = schemas.Filter(
        model_names=[model_name1],
        labels=[{"class": "dog"}],
        prediction_scores=[
            schemas.NumericFilter(
                value=0.9,
                operator=">",
            ),
        ],
    )
    q = Query(models.Datum.uid).filter(f).predictions()
    datum_uids = db.query(q).distinct().all()
    assert len(datum_uids) == 2
    assert (datum_uid3,) in datum_uids
    assert (datum_uid4,) in datum_uids

    # Q: Get datums that `model1` has `bounding_box` annotations for with label `dog` and prediction score > 0.75.
    f = schemas.Filter(
        model_names=[model_name1],
        labels=[{"class": "dog"}],
        prediction_scores=[
            schemas.NumericFilter(
                value=0.75,
                operator=">",
            )
        ],
        annotation_types=[enums.AnnotationType.BOX],
    )
    q = Query(models.Datum.uid).filter(f).predictions()
    datum_uids = db.query(q).distinct().all()
    assert len(datum_uids) == 2
    assert (datum_uid3,) in datum_uids
    assert (datum_uid4,) in datum_uids


def test_query_by_annotation_geometry(
    db: Session,
    model_sim,
):
    f = schemas.Filter(
        annotation_geometric_area=[
            schemas.NumericFilter(
                value=75,
                operator=">",
            ),
        ],
    )

    # Q: Get `bounding_box` annotations that have an area > 75.
    q = Query(models.Annotation).filter(f).any()
    annotations = db.query(q).all()
    assert len(annotations) == 12

    # Q: Get `bounding_box` annotations from `model1` that have an area > 75.
    f.model_names = [model_name1]
    q = Query(models.Annotation).filter(f).any()
    annotations = db.query(q).all()
    assert len(annotations) == 4


def test_multiple_tables_in_args(
    db: Session,
    model_sim,
):
    f = schemas.Filter(
        datum_uids=[datum_uid1],
    )

    # Q: Get model + dataset name pairings for a datum with `uid1` using the full tables
    name_pairings = (
        db.query(Query(models.Model.name, models.Dataset.name).filter(f).any())
        .distinct()
        .all()
    )
    assert len(name_pairings) == 2
    assert (
        model_name1,
        dset_name,
    ) in name_pairings
    assert (
        model_name2,
        dset_name,
    ) in name_pairings

    # Q: Get model + dataset name pairings for a datum with `uid1` using the table attributes directly
    q = Query(models.Model.name, models.Dataset.name).filter(f).any()
    name_pairings = db.query(q).distinct().all()
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
    q = Query(models.Model.name, models.Dataset).filter(f).any()
    name_pairings = [
        (
            pair[0],
            pair[2],
        )
        for pair in db.query(q).distinct().all()
    ]
    assert len(name_pairings) == 2
    assert (
        model_name1,
        dset_name,
    ) in name_pairings
    assert (
        model_name2,
        dset_name,
    ) in name_pairings


def test_datum_geospatial_filters(
    db: Session,
    model_sim,
    model_object=models.Datum.uid,
    arg_name: str = "datum_geospatial",
):
    def _get_geospatial_names_from_filter(
        db: Session,
        geodict: dict[
            str,
            list[list[list[list[float | int]]]]
            | list[list[list[float | int]]]
            | list[float | int]
            | str,
        ],
        operator: str,
        model_object: models.Datum,
        arg_name: str,
    ):
        f = schemas.Filter(
            **{
                arg_name: [
                    schemas.GeospatialFilter(
                        value=schemas.geojson.from_dict(geodict),
                        operator=operator,
                    ),
                ]
            }
        )

        q = Query(model_object).filter(f).any()
        names = db.query(q).distinct().all()
        return names

    # test inside filters
    names = _get_geospatial_names_from_filter(
        db=db,
        geodict={
            "type": "Polygon",
            "coordinates": [
                [
                    [-20, -20],
                    [60, -20],
                    [60, 60],
                    [-20, 60],
                ]
            ],
        },
        operator="inside",
        model_object=model_object,
        arg_name=arg_name,
    )

    assert len(names) == 2
    assert ("uid1",) in names
    assert ("uid3",) in names

    # test intersections
    names = _get_geospatial_names_from_filter(
        db=db,
        geodict={
            "type": "Polygon",
            "coordinates": [
                [
                    [60, 60],
                    [110, 60],
                    [110, 110],
                    [60, 110],
                ]
            ],
        },
        operator="intersect",
        model_object=model_object,
        arg_name=arg_name,
    )

    assert len(names) == 2
    assert ("uid2",) in names
    assert ("uid4",) in names

    # test point
    names = _get_geospatial_names_from_filter(
        db=db,
        geodict={
            "type": "Point",
            "coordinates": [81, 80],
        },
        operator="intersect",
        model_object=model_object,
        arg_name=arg_name,
    )

    assert len(names) == 1
    assert ("uid4",) in names

    # test multipolygon
    names = _get_geospatial_names_from_filter(
        db=db,
        geodict={
            "type": "MultiPolygon",
            "coordinates": [
                [
                    [
                        [-20, -20],
                        [20, -20],
                        [20, 20],
                        [-20, 20],
                    ]
                ],
                [
                    [
                        [15, 15],
                        [15, 35],
                        [35, 35],
                        [35, 15],
                    ]
                ],
            ],
        },
        operator="intersect",
        model_object=model_object,
        arg_name=arg_name,
    )

    assert len(names) == 3
    assert ("uid1",) in names
    assert ("uid2",) in names
    assert ("uid3",) in names

    # test WHERE miss
    names = _get_geospatial_names_from_filter(
        db=db,
        geodict={
            "type": "Point",
            "coordinates": [-11, -11],
        },
        operator="intersect",
        model_object=model_object,
        arg_name=arg_name,
    )

    assert len(names) == 0

    # test outside
    names = _get_geospatial_names_from_filter(
        db=db,
        geodict={
            "type": "Point",
            "coordinates": [-11, -11],
        },
        operator="outside",
        model_object=model_object,
        arg_name=arg_name,
    )

    assert len(names) == 4
    assert ("uid1",) in names
    assert ("uid2",) in names
    assert ("uid3",) in names
    assert ("uid4",) in names

    names = _get_geospatial_names_from_filter(
        db=db,
        geodict={
            "type": "Polygon",
            "coordinates": [
                [
                    [-20, -20],
                    [60, -20],
                    [60, 60],
                    [-20, 60],
                ]
            ],
        },
        operator="outside",
        model_object=model_object,
        arg_name=arg_name,
    )

    assert len(names) == 2
    assert ("uid2",) in names
    assert ("uid4",) in names


def test_dataset_geospatial_filters(
    db: Session,
    model_sim,
    model_object=models.Dataset.name,
    arg_name: str = "dataset_geospatial",
):
    def _get_geospatial_names_from_filter(
        db: Session,
        geodict: dict[
            str,
            list[list[list[list[float | int]]]]
            | list[list[list[float | int]]]
            | list[float | int]
            | str,
        ],
        operator: str,
        model_object: models.Datum,
        arg_name: str,
    ):
        f = schemas.Filter(
            **{
                arg_name: [
                    schemas.GeospatialFilter(
                        value=schemas.geojson.from_dict(geodict),
                        operator=operator,
                    ),
                ]
            }
        )

        q = Query(model_object).filter(f).any()
        names = db.query(q).distinct().all()
        return names

    # test inside filters
    names = _get_geospatial_names_from_filter(
        db=db,
        geodict={
            "type": "Polygon",
            "coordinates": [
                [
                    [-20, -20],
                    [60, -20],
                    [60, 60],
                    [-20, 60],
                ]
            ],
        },
        operator="inside",
        model_object=model_object,
        arg_name=arg_name,
    )

    assert len(names) == 1
    assert ("dataset1",) in names

    # test point
    names = _get_geospatial_names_from_filter(
        db=db,
        geodict={
            "type": "Point",
            "coordinates": [1, 1],
        },
        operator="intersect",
        model_object=model_object,
        arg_name=arg_name,
    )

    assert len(names) == 1
    assert ("dataset1",) in names

    # test multipolygon
    names = _get_geospatial_names_from_filter(
        db=db,
        geodict={
            "type": "MultiPolygon",
            "coordinates": [
                [
                    [
                        [-20, -20],
                        [20, -20],
                        [20, 20],
                        [-20, 20],
                    ]
                ],
                [
                    [
                        [15, 15],
                        [15, 35],
                        [35, 35],
                        [35, 15],
                    ]
                ],
            ],
        },
        operator="intersect",
        model_object=model_object,
        arg_name=arg_name,
    )

    assert len(names) == 1
    assert ("dataset1",) in names

    # test WHERE miss
    names = _get_geospatial_names_from_filter(
        db=db,
        geodict={
            "type": "Point",
            "coordinates": [-11, -11],
        },
        operator="intersect",
        model_object=model_object,
        arg_name=arg_name,
    )

    assert len(names) == 0

    # test outside
    names = _get_geospatial_names_from_filter(
        db=db,
        geodict={
            "type": "Point",
            "coordinates": [-11, -11],
        },
        operator="outside",
        model_object=model_object,
        arg_name=arg_name,
    )

    assert len(names) == 1
    assert ("dataset1",) in names


def test_model_geospatial_filters(
    db: Session,
    model_sim,
    model_object=models.Model.name,
    arg_name: str = "model_geospatial",
):
    def _get_geospatial_names_from_filter(
        db: Session,
        geodict: dict[
            str,
            list[list[list[list[float | int]]]]
            | list[list[list[float | int]]]
            | list[float | int]
            | str,
        ],
        operator: str,
        model_object: models.Datum,
        arg_name: str,
    ):
        f = schemas.Filter(
            **{
                arg_name: [
                    schemas.GeospatialFilter(
                        value=schemas.geojson.from_dict(geodict),
                        operator=operator,
                    ),
                ]
            }
        )

        q = Query(model_object).filter(f).any()
        names = db.query(q).distinct().all()
        return names

    # test inside filters
    names = _get_geospatial_names_from_filter(
        db=db,
        geodict={
            "type": "Polygon",
            "coordinates": [
                [
                    [-20, -20],
                    [60, -20],
                    [60, 60],
                    [-20, 60],
                ]
            ],
        },
        operator="inside",
        model_object=model_object,
        arg_name=arg_name,
    )

    assert len(names) == 1
    assert ("model1",) in names

    # test point
    names = _get_geospatial_names_from_filter(
        db=db,
        geodict={
            "type": "Point",
            "coordinates": [1, 1],
        },
        operator="intersect",
        model_object=model_object,
        arg_name=arg_name,
    )

    assert len(names) == 1
    assert ("model1",) in names

    # test multipolygon
    names = _get_geospatial_names_from_filter(
        db=db,
        geodict={
            "type": "MultiPolygon",
            "coordinates": [
                [
                    [
                        [-20, -20],
                        [20, -20],
                        [20, 20],
                        [-20, 20],
                    ]
                ],
                [
                    [
                        [15, 15],
                        [15, 35],
                        [35, 35],
                        [35, 15],
                    ]
                ],
            ],
        },
        operator="intersect",
        model_object=model_object,
        arg_name=arg_name,
    )

    assert len(names) == 1
    assert ("model1",) in names

    # test WHERE miss
    names = _get_geospatial_names_from_filter(
        db=db,
        geodict={
            "type": "Point",
            "coordinates": [-11, -11],
        },
        operator="intersect",
        model_object=model_object,
        arg_name=arg_name,
    )

    assert len(names) == 0

    # test outside
    names = _get_geospatial_names_from_filter(
        db=db,
        geodict={
            "type": "Point",
            "coordinates": [-11, -11],
        },
        operator="outside",
        model_object=model_object,
        arg_name=arg_name,
    )

    assert len(names) == 1
    assert ("model1",) in names


@pytest.fixture
def datetime_metadata() -> list[schemas.DateTime]:
    """List of datetimes using different formats."""
    return [
        schemas.DateTime(
            datetime="2022-01-01",
        ),
        schemas.DateTime(
            datetime="Apr 07 2023 16:34:56",
        ),
        schemas.DateTime(datetime="Apr 07 2023 4:35:56 PM"),
        schemas.DateTime(
            datetime="November 12, 2023",
        ),
        schemas.DateTime(
            datetime="2023-12-04T00:05:23+04:00",
        ),
    ]


@pytest.fixture
def date_metadata() -> list[schemas.Date]:
    """List of dates using different formats."""
    return [
        schemas.Date(
            date="2022-01-01",
        ),
        schemas.Date(
            date="Apr 07 2023",
        ),
        schemas.Date(date="Apr 08 2023"),
        schemas.Date(
            date="November 12, 2023",
        ),
        schemas.Date(
            date="2023-12-04",
        ),
    ]


@pytest.fixture
def time_metadata() -> list[schemas.Time]:
    """List of times using different formats."""
    return [
        schemas.Time(
            time="00:05:23",
        ),
        schemas.Time(
            time="16:34:56",
        ),
        schemas.Time(time="16:35:56.000283"),
        schemas.Time(
            time="18:02:23",
        ),
        schemas.Time(
            time="22:05:23",
        ),
    ]


@pytest.fixture
def duration_metadata() -> list[schemas.Duration]:
    """List of time durations using different formats."""
    return [
        schemas.Duration(
            duration="0.0001",
        ),
        schemas.Duration(
            duration="324.01",
        ),
        schemas.Duration(duration="324.02"),
        schemas.Duration(
            duration="180223.0",
        ),
        schemas.Duration(
            duration="220523.0",
        ),
    ]


def _test_dataset_datetime_query(
    db: Session,
    key: str,
    metadata_: list[
        schemas.DateTime | schemas.Date | schemas.Time | schemas.Duration
    ],
):
    """
    The metadata_ param is a pytest fixture containing sequential timestamps.
    """

    time_filter = lambda idx, op: (  # noqa: E731
        Query(models.Dataset)
        .filter(
            schemas.Filter(
                dataset_metadata={
                    key: [
                        schemas.DateTimeFilter(
                            value=metadata_[idx], operator=op
                        )
                    ]
                }
            )
        )
        .any()
    )

    # Check equality operator
    op = "=="

    results = db.query(time_filter(0, op)).all()
    assert len(results) == 0

    results = db.query(time_filter(1, op)).all()
    assert len(results) == 1
    assert results[0].name == "dataset1"

    results = db.query(time_filter(2, op)).all()
    assert len(results) == 0

    results = db.query(time_filter(3, op)).all()
    assert len(results) == 1
    assert results[0].name == "dataset2"

    results = db.query(time_filter(4, op)).all()
    assert len(results) == 0

    # Check inequality operator
    op = "!="

    results = db.query(time_filter(0, op)).all()
    assert len(results) == 2
    assert "dataset1" in [result.name for result in results]
    assert "dataset2" in [result.name for result in results]

    results = db.query(time_filter(1, op)).all()
    assert len(results) == 1
    assert results[0].name == "dataset2"

    results = db.query(time_filter(2, op)).all()
    assert len(results) == 2
    assert "dataset1" in [result.name for result in results]
    assert "dataset2" in [result.name for result in results]

    results = db.query(time_filter(3, op)).all()
    assert len(results) == 1
    assert results[0].name == "dataset1"

    results = db.query(time_filter(4, op)).all()
    assert len(results) == 2
    assert "dataset1" in [result.name for result in results]
    assert "dataset2" in [result.name for result in results]

    # Check less-than operator
    op = "<"

    results = db.query(time_filter(0, op)).all()
    assert len(results) == 0

    results = db.query(time_filter(1, op)).all()
    assert len(results) == 0

    results = db.query(time_filter(2, op)).all()
    assert len(results) == 1
    assert results[0].name == "dataset1"

    results = db.query(time_filter(3, op)).all()
    assert len(results) == 1
    assert results[0].name == "dataset1"

    results = db.query(time_filter(4, op)).all()
    assert len(results) == 2
    assert "dataset1" in [result.name for result in results]
    assert "dataset2" in [result.name for result in results]

    # Check greater-than operator
    op = ">"

    results = db.query(time_filter(0, op)).all()
    assert len(results) == 2
    assert "dataset1" in [result.name for result in results]
    assert "dataset2" in [result.name for result in results]

    results = db.query(time_filter(1, op)).all()
    assert len(results) == 1
    assert results[0].name == "dataset2"

    results = db.query(time_filter(2, op)).all()
    assert len(results) == 1
    assert results[0].name == "dataset2"

    results = db.query(time_filter(3, op)).all()
    assert len(results) == 0

    results = db.query(time_filter(4, op)).all()
    assert len(results) == 0

    # Check less-than or equal operator
    op = "<="

    results = db.query(time_filter(0, op)).all()
    assert len(results) == 0

    results = db.query(time_filter(1, op)).all()
    assert len(results) == 1
    assert results[0].name == "dataset1"

    results = db.query(time_filter(2, op)).all()
    assert len(results) == 1
    assert results[0].name == "dataset1"

    results = db.query(time_filter(3, op)).all()
    assert len(results) == 2
    assert "dataset1" in [result.name for result in results]
    assert "dataset2" in [result.name for result in results]

    results = db.query(time_filter(4, op)).all()
    assert len(results) == 2
    assert "dataset1" in [result.name for result in results]
    assert "dataset2" in [result.name for result in results]

    # Check greater-than or equal operator
    op = ">="

    results = db.query(time_filter(0, op)).all()
    assert len(results) == 2
    assert "dataset1" in [result.name for result in results]
    assert "dataset2" in [result.name for result in results]

    results = db.query(time_filter(1, op)).all()
    assert len(results) == 2
    assert "dataset1" in [result.name for result in results]
    assert "dataset2" in [result.name for result in results]

    results = db.query(time_filter(2, op)).all()
    assert len(results) == 1
    assert results[0].name == "dataset2"

    results = db.query(time_filter(3, op)).all()
    assert len(results) == 1
    assert results[0].name == "dataset2"

    results = db.query(time_filter(4, op)).all()
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
                datetime_key: datetime_metadata[1].model_dump(),
                date_key: date_metadata[1].model_dump(),
                time_key: time_metadata[1].model_dump(),
                duration_key: duration_metadata[1].model_dump(),
            },
        ),
    )
    crud.create_dataset(
        db=db,
        dataset=schemas.Dataset(
            name="dataset2",
            metadata={
                datetime_key: datetime_metadata[3].model_dump(),
                date_key: date_metadata[3].model_dump(),
                time_key: time_metadata[3].model_dump(),
                duration_key: duration_metadata[3].model_dump(),
            },
        ),
    )

    _test_dataset_datetime_query(db, datetime_key, datetime_metadata)
    _test_dataset_datetime_query(db, date_key, date_metadata)
    _test_dataset_datetime_query(db, time_key, time_metadata)
    _test_dataset_datetime_query(db, duration_key, duration_metadata)


def _test_model_datetime_query(
    db: Session,
    key: str,
    metadata_: list[
        schemas.DateTime | schemas.Date | schemas.Time | schemas.Duration
    ],
):
    """
    The metadata_ param is a pytest fixture containing sequential timestamps.
    """

    time_filter = lambda idx, op: (  # noqa: E731
        Query(models.Model)
        .filter(
            schemas.Filter(
                model_metadata={
                    key: [
                        schemas.DateTimeFilter(
                            value=metadata_[idx], operator=op
                        )
                    ]
                }
            )
        )
        .any()
    )

    # Check equality operator
    op = "=="

    results = db.query(time_filter(0, op)).all()
    assert len(results) == 0

    results = db.query(time_filter(1, op)).all()
    assert len(results) == 1
    assert results[0].name == "model1"

    results = db.query(time_filter(2, op)).all()
    assert len(results) == 0

    results = db.query(time_filter(3, op)).all()
    assert len(results) == 1
    assert results[0].name == "model2"

    results = db.query(time_filter(4, op)).all()
    assert len(results) == 0

    # Check inequality operator
    op = "!="

    results = db.query(time_filter(0, op)).all()
    assert len(results) == 2
    assert "model1" in [result.name for result in results]
    assert "model2" in [result.name for result in results]

    results = db.query(time_filter(1, op)).all()
    assert len(results) == 1
    assert results[0].name == "model2"

    results = db.query(time_filter(2, op)).all()
    assert len(results) == 2
    assert "model1" in [result.name for result in results]
    assert "model2" in [result.name for result in results]

    results = db.query(time_filter(3, op)).all()
    assert len(results) == 1
    assert results[0].name == "model1"

    results = db.query(time_filter(4, op)).all()
    assert len(results) == 2
    assert "model1" in [result.name for result in results]
    assert "model2" in [result.name for result in results]

    # Check less-than operator
    op = "<"

    results = db.query(time_filter(0, op)).all()
    assert len(results) == 0

    results = db.query(time_filter(1, op)).all()
    assert len(results) == 0

    results = db.query(time_filter(2, op)).all()
    assert len(results) == 1
    assert results[0].name == "model1"

    results = db.query(time_filter(3, op)).all()
    assert len(results) == 1
    assert results[0].name == "model1"

    results = db.query(time_filter(4, op)).all()
    assert len(results) == 2
    assert "model1" in [result.name for result in results]
    assert "model2" in [result.name for result in results]

    # Check greater-than operator
    op = ">"

    results = db.query(time_filter(0, op)).all()
    assert len(results) == 2
    assert "model1" in [result.name for result in results]
    assert "model2" in [result.name for result in results]

    results = db.query(time_filter(1, op)).all()
    assert len(results) == 1
    assert results[0].name == "model2"

    results = db.query(time_filter(2, op)).all()
    assert len(results) == 1
    assert results[0].name == "model2"

    results = db.query(time_filter(3, op)).all()
    assert len(results) == 0

    results = db.query(time_filter(4, op)).all()
    assert len(results) == 0

    # Check less-than or equal operator
    op = "<="

    results = db.query(time_filter(0, op)).all()
    assert len(results) == 0

    results = db.query(time_filter(1, op)).all()
    assert len(results) == 1
    assert results[0].name == "model1"

    results = db.query(time_filter(2, op)).all()
    assert len(results) == 1
    assert results[0].name == "model1"

    results = db.query(time_filter(3, op)).all()
    assert len(results) == 2
    assert "model1" in [result.name for result in results]
    assert "model2" in [result.name for result in results]

    results = db.query(time_filter(4, op)).all()
    assert len(results) == 2
    assert "model1" in [result.name for result in results]
    assert "model2" in [result.name for result in results]

    # Check greater-than or equal operator
    op = ">="

    results = db.query(time_filter(0, op)).all()
    assert len(results) == 2
    assert "model1" in [result.name for result in results]
    assert "model2" in [result.name for result in results]

    results = db.query(time_filter(1, op)).all()
    assert len(results) == 2
    assert "model1" in [result.name for result in results]
    assert "model2" in [result.name for result in results]

    results = db.query(time_filter(2, op)).all()
    assert len(results) == 1
    assert results[0].name == "model2"

    results = db.query(time_filter(3, op)).all()
    assert len(results) == 1
    assert results[0].name == "model2"

    results = db.query(time_filter(4, op)).all()
    assert len(results) == 0


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
                datetime_key: datetime_metadata[1].model_dump(),
                date_key: date_metadata[1].model_dump(),
                time_key: time_metadata[1].model_dump(),
                duration_key: duration_metadata[1].model_dump(),
            },
        ),
    )
    crud.create_model(
        db=db,
        model=schemas.Model(
            name="model2",
            metadata={
                datetime_key: datetime_metadata[3].model_dump(),
                date_key: date_metadata[3].model_dump(),
                time_key: time_metadata[3].model_dump(),
                duration_key: duration_metadata[3].model_dump(),
            },
        ),
    )

    _test_model_datetime_query(db, datetime_key, datetime_metadata)
    _test_model_datetime_query(db, date_key, date_metadata)
    _test_model_datetime_query(db, time_key, time_metadata)
    _test_model_datetime_query(db, duration_key, duration_metadata)


def _test_datum_datetime_query(
    db: Session,
    key: str,
    metadata_: list[
        schemas.DateTime | schemas.Date | schemas.Time | schemas.Duration
    ],
):
    """
    The metadata_ param is a pytest fixture containing sequential timestamps.
    """

    time_filter = lambda idx, op: (  # noqa: E731
        Query(models.Datum)
        .filter(
            schemas.Filter(
                datum_metadata={
                    key: [
                        schemas.DateTimeFilter(
                            value=metadata_[idx], operator=op
                        )
                    ]
                }
            )
        )
        .any()
    )

    assert len(db.query(models.Datum).all()) == 4

    # Check equality operator
    op = "=="

    results = db.query(time_filter(0, op)).all()
    assert len(results) == 0

    results = db.query(time_filter(1, op)).all()
    assert len(results) == 1
    assert results[0].uid == datum_uid1

    results = db.query(time_filter(2, op)).all()
    assert len(results) == 2
    assert datum_uid2 in [result.uid for result in results]
    assert datum_uid3 in [result.uid for result in results]

    results = db.query(time_filter(3, op)).all()
    assert len(results) == 1
    assert results[0].uid == datum_uid4

    results = db.query(time_filter(4, op)).all()
    assert len(results) == 0

    # Check inequality operator
    op = "!="

    results = db.query(time_filter(0, op)).all()
    assert len(results) == 4
    assert datum_uid1 in [result.uid for result in results]
    assert datum_uid2 in [result.uid for result in results]
    assert datum_uid3 in [result.uid for result in results]
    assert datum_uid4 in [result.uid for result in results]

    results = db.query(time_filter(1, op)).all()
    assert len(results) == 3
    assert datum_uid2 in [result.uid for result in results]
    assert datum_uid3 in [result.uid for result in results]
    assert datum_uid4 in [result.uid for result in results]

    results = db.query(time_filter(2, op)).all()
    assert len(results) == 2
    assert datum_uid1 in [result.uid for result in results]
    assert datum_uid4 in [result.uid for result in results]

    results = db.query(time_filter(3, op)).all()
    assert len(results) == 3
    assert datum_uid1 in [result.uid for result in results]
    assert datum_uid2 in [result.uid for result in results]
    assert datum_uid3 in [result.uid for result in results]

    results = db.query(time_filter(4, op)).all()
    assert len(results) == 4
    assert datum_uid1 in [result.uid for result in results]
    assert datum_uid2 in [result.uid for result in results]
    assert datum_uid3 in [result.uid for result in results]
    assert datum_uid4 in [result.uid for result in results]

    # Check less-than operator
    op = "<"

    results = db.query(time_filter(0, op)).all()
    assert len(results) == 0

    results = db.query(time_filter(1, op)).all()
    assert len(results) == 0

    results = db.query(time_filter(2, op)).all()
    assert len(results) == 1
    assert results[0].uid == datum_uid1

    results = db.query(time_filter(3, op)).all()
    assert len(results) == 3
    assert datum_uid1 in [result.uid for result in results]
    assert datum_uid2 in [result.uid for result in results]
    assert datum_uid3 in [result.uid for result in results]

    results = db.query(time_filter(4, op)).all()
    assert len(results) == 4
    assert datum_uid1 in [result.uid for result in results]
    assert datum_uid2 in [result.uid for result in results]
    assert datum_uid3 in [result.uid for result in results]
    assert datum_uid4 in [result.uid for result in results]

    # Check greater-than operator
    op = ">"

    results = db.query(time_filter(0, op)).all()
    assert len(results) == 4
    assert datum_uid1 in [result.uid for result in results]
    assert datum_uid2 in [result.uid for result in results]
    assert datum_uid3 in [result.uid for result in results]
    assert datum_uid4 in [result.uid for result in results]

    results = db.query(time_filter(1, op)).all()
    assert len(results) == 3
    assert datum_uid2 in [result.uid for result in results]
    assert datum_uid3 in [result.uid for result in results]
    assert datum_uid4 in [result.uid for result in results]

    results = db.query(time_filter(2, op)).all()
    assert len(results) == 1
    assert results[0].uid == datum_uid4

    results = db.query(time_filter(3, op)).all()
    assert len(results) == 0

    results = db.query(time_filter(4, op)).all()
    assert len(results) == 0

    # Check less-than or equal operator
    op = "<="

    results = db.query(time_filter(0, op)).all()
    assert len(results) == 0

    results = db.query(time_filter(1, op)).all()
    assert len(results) == 1
    assert results[0].uid == datum_uid1

    results = db.query(time_filter(2, op)).all()
    assert len(results) == 3
    assert datum_uid1 in [result.uid for result in results]
    assert datum_uid2 in [result.uid for result in results]
    assert datum_uid3 in [result.uid for result in results]

    results = db.query(time_filter(3, op)).all()
    assert len(results) == 4
    assert datum_uid1 in [result.uid for result in results]
    assert datum_uid2 in [result.uid for result in results]
    assert datum_uid3 in [result.uid for result in results]
    assert datum_uid4 in [result.uid for result in results]

    results = db.query(time_filter(4, op)).all()
    assert len(results) == 4
    assert datum_uid1 in [result.uid for result in results]
    assert datum_uid2 in [result.uid for result in results]
    assert datum_uid3 in [result.uid for result in results]
    assert datum_uid4 in [result.uid for result in results]

    # Check greater-than or equal operator
    op = ">="

    results = db.query(time_filter(0, op)).all()
    assert len(results) == 4
    assert datum_uid1 in [result.uid for result in results]
    assert datum_uid2 in [result.uid for result in results]
    assert datum_uid3 in [result.uid for result in results]
    assert datum_uid4 in [result.uid for result in results]

    results = db.query(time_filter(1, op)).all()
    assert len(results) == 4
    assert datum_uid1 in [result.uid for result in results]
    assert datum_uid2 in [result.uid for result in results]
    assert datum_uid3 in [result.uid for result in results]
    assert datum_uid4 in [result.uid for result in results]

    results = db.query(time_filter(2, op)).all()
    assert len(results) == 3
    assert datum_uid2 in [result.uid for result in results]
    assert datum_uid3 in [result.uid for result in results]
    assert datum_uid4 in [result.uid for result in results]

    results = db.query(time_filter(3, op)).all()
    assert len(results) == 1
    assert results[0].uid == datum_uid4

    results = db.query(time_filter(4, op)).all()
    assert len(results) == 0


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

    datum_1.metadata[datetime_key] = datetime_metadata[1].model_dump()
    datum_2.metadata[datetime_key] = datetime_metadata[2].model_dump()
    datum_3.metadata[datetime_key] = datetime_metadata[2].model_dump()
    datum_4.metadata[datetime_key] = datetime_metadata[3].model_dump()

    datum_1.metadata[date_key] = date_metadata[1].model_dump()
    datum_2.metadata[date_key] = date_metadata[2].model_dump()
    datum_3.metadata[date_key] = date_metadata[2].model_dump()
    datum_4.metadata[date_key] = date_metadata[3].model_dump()

    datum_1.metadata[time_key] = time_metadata[1].model_dump()
    datum_2.metadata[time_key] = time_metadata[2].model_dump()
    datum_3.metadata[time_key] = time_metadata[2].model_dump()
    datum_4.metadata[time_key] = time_metadata[3].model_dump()

    datum_1.metadata[duration_key] = duration_metadata[1].model_dump()
    datum_2.metadata[duration_key] = duration_metadata[2].model_dump()
    datum_3.metadata[duration_key] = duration_metadata[2].model_dump()
    datum_4.metadata[duration_key] = duration_metadata[3].model_dump()

    annotation = schemas.Annotation(
        task_type=enums.TaskType.CLASSIFICATION,
        labels=[schemas.Label(key="k1", value="v1")],
    )

    crud.create_dataset(
        db=db,
        dataset=schemas.Dataset(
            name=dset_name,
        ),
    )

    crud.create_groundtruth(
        db=db,
        groundtruth=schemas.GroundTruth(
            datum=datum_1, annotations=[annotation]
        ),
    )
    crud.create_groundtruth(
        db=db,
        groundtruth=schemas.GroundTruth(
            datum=datum_2, annotations=[annotation]
        ),
    )
    crud.create_groundtruth(
        db=db,
        groundtruth=schemas.GroundTruth(
            datum=datum_3, annotations=[annotation]
        ),
    )
    crud.create_groundtruth(
        db=db,
        groundtruth=schemas.GroundTruth(
            datum=datum_4, annotations=[annotation]
        ),
    )

    crud.create_model(
        db=db,
        model=schemas.Model(
            name="model1",
            metadata={
                datetime_key: datetime_metadata[1].model_dump(),
                date_key: date_metadata[1].model_dump(),
                time_key: time_metadata[1].model_dump(),
            },
        ),
    )
    crud.create_model(
        db=db,
        model=schemas.Model(
            name="model2",
            metadata={
                datetime_key: datetime_metadata[3].model_dump(),
                date_key: date_metadata[3].model_dump(),
                time_key: time_metadata[3].model_dump(),
            },
        ),
    )

    _test_datum_datetime_query(db, datetime_key, datetime_metadata)
    _test_datum_datetime_query(db, date_key, date_metadata)
    _test_datum_datetime_query(db, time_key, time_metadata)
    _test_datum_datetime_query(db, duration_key, duration_metadata)


def _test_annotation_datetime_query(
    db: Session,
    key: str,
    metadata_: list[
        schemas.DateTime | schemas.Date | schemas.Time | schemas.Duration
    ],
):
    """
    The metadata_ param is a pytest fixture containing sequential timestamps.
    """

    assert len(db.query(models.Annotation).all()) == 4

    time_filter = lambda idx, op: (  # noqa: E731
        Query(models.Annotation)
        .filter(
            schemas.Filter(
                annotation_metadata={
                    key: [
                        schemas.DateTimeFilter(
                            value=metadata_[idx], operator=op
                        )
                    ]
                }
            )
        )
        .any()
    )

    # Check equality operator
    op = "=="

    results = db.query(time_filter(0, op)).all()
    assert len(results) == 0

    results = db.query(time_filter(1, op)).all()
    assert len(results) == 1

    results = db.query(time_filter(2, op)).all()
    assert len(results) == 2

    results = db.query(time_filter(3, op)).all()
    assert len(results) == 1

    results = db.query(time_filter(4, op)).all()
    assert len(results) == 0

    # Check inequality operator
    op = "!="

    results = db.query(time_filter(0, op)).all()
    assert len(results) == 4

    results = db.query(time_filter(1, op)).all()
    assert len(results) == 3

    results = db.query(time_filter(2, op)).all()
    assert len(results) == 2

    results = db.query(time_filter(3, op)).all()
    assert len(results) == 3

    results = db.query(time_filter(4, op)).all()
    assert len(results) == 4

    # Check less-than operator
    op = "<"

    results = db.query(time_filter(0, op)).all()
    assert len(results) == 0

    results = db.query(time_filter(1, op)).all()
    assert len(results) == 0

    results = db.query(time_filter(2, op)).all()
    assert len(results) == 1

    results = db.query(time_filter(3, op)).all()
    assert len(results) == 3

    results = db.query(time_filter(4, op)).all()
    assert len(results) == 4

    # Check greater-than operator
    op = ">"

    results = db.query(time_filter(0, op)).all()
    assert len(results) == 4

    results = db.query(time_filter(1, op)).all()
    assert len(results) == 3

    results = db.query(time_filter(2, op)).all()
    assert len(results) == 1

    results = db.query(time_filter(3, op)).all()
    assert len(results) == 0

    results = db.query(time_filter(4, op)).all()
    assert len(results) == 0

    # Check less-than or equal operator
    op = "<="

    results = db.query(time_filter(0, op)).all()
    assert len(results) == 0

    results = db.query(time_filter(1, op)).all()
    assert len(results) == 1

    results = db.query(time_filter(2, op)).all()
    assert len(results) == 3

    results = db.query(time_filter(3, op)).all()
    assert len(results) == 4

    results = db.query(time_filter(4, op)).all()
    assert len(results) == 4

    # Check greater-than or equal operator
    op = ">="

    results = db.query(time_filter(0, op)).all()
    assert len(results) == 4

    results = db.query(time_filter(1, op)).all()
    assert len(results) == 4

    results = db.query(time_filter(2, op)).all()
    assert len(results) == 3

    results = db.query(time_filter(3, op)).all()
    assert len(results) == 1

    results = db.query(time_filter(4, op)).all()
    assert len(results) == 0


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
        task_type=enums.TaskType.CLASSIFICATION,
        labels=[schemas.Label(key="k1", value="v1")],
        metadata={
            datetime_key: datetime_metadata[1].model_dump(),
            date_key: date_metadata[1].model_dump(),
            time_key: time_metadata[1].model_dump(),
            duration_key: duration_metadata[1].model_dump(),
        },
    )
    annotation_2 = schemas.Annotation(
        task_type=enums.TaskType.CLASSIFICATION,
        labels=[schemas.Label(key="k2", value="v2")],
        metadata={
            datetime_key: datetime_metadata[2].model_dump(),
            date_key: date_metadata[2].model_dump(),
            time_key: time_metadata[2].model_dump(),
            duration_key: duration_metadata[2].model_dump(),
        },
    )
    annotation_3 = schemas.Annotation(
        task_type=enums.TaskType.CLASSIFICATION,
        labels=[schemas.Label(key="k3", value="v3")],
        metadata={
            datetime_key: datetime_metadata[2].model_dump(),
            date_key: date_metadata[2].model_dump(),
            time_key: time_metadata[2].model_dump(),
            duration_key: duration_metadata[2].model_dump(),
        },
    )
    annotation_4 = schemas.Annotation(
        task_type=enums.TaskType.CLASSIFICATION,
        labels=[schemas.Label(key="k4", value="v4")],
        metadata={
            datetime_key: datetime_metadata[3].model_dump(),
            date_key: date_metadata[3].model_dump(),
            time_key: time_metadata[3].model_dump(),
            duration_key: duration_metadata[3].model_dump(),
        },
    )

    crud.create_dataset(
        db=db,
        dataset=schemas.Dataset(
            name=dset_name,
        ),
    )

    crud.create_groundtruth(
        db=db,
        groundtruth=schemas.GroundTruth(
            datum=datum_1,
            annotations=[
                annotation_1,
                annotation_2,
                annotation_3,
                annotation_4,
            ],
        ),
    )

    _test_annotation_datetime_query(db, datetime_key, datetime_metadata)
    _test_annotation_datetime_query(db, date_key, date_metadata)
    _test_annotation_datetime_query(db, time_key, time_metadata)
    _test_annotation_datetime_query(db, duration_key, duration_metadata)
