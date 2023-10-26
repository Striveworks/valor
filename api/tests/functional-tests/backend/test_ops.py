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
def metadata_1() -> list[schemas.Metadatum]:
    return [
        schemas.Metadatum(key="numeric", value=0.4),
        schemas.Metadatum(key="str", value="abc"),
        schemas.Metadatum(key="height", value=10),
        schemas.Metadatum(key="width", value=10),
    ]


@pytest.fixture
def metadata_2() -> list[schemas.Metadatum]:
    return [
        schemas.Metadatum(key="numeric", value=0.6),
        schemas.Metadatum(key="str", value="abc"),
        schemas.Metadatum(key="height", value=10),
        schemas.Metadatum(key="width", value=10),
    ]


@pytest.fixture
def metadata_3() -> list[schemas.Metadatum]:
    return [
        schemas.Metadatum(key="numeric", value=0.4),
        schemas.Metadatum(key="str", value="xyz"),
        schemas.Metadatum(key="height", value=10),
        schemas.Metadatum(key="width", value=10),
    ]


@pytest.fixture
def metadata_4() -> list[schemas.Metadatum]:
    return [
        schemas.Metadatum(key="numeric", value=0.6),
        schemas.Metadatum(key="str", value="xyz"),
        schemas.Metadatum(key="height", value=10),
        schemas.Metadatum(key="width", value=10),
    ]


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
        dataset=dset_name,
        metadata=metadata_1,
    )


@pytest.fixture
def datum_2(metadata_2) -> schemas.Datum:
    return schemas.Datum(
        uid=datum_uid2,
        dataset=dset_name,
        metadata=metadata_2,
    )


@pytest.fixture
def datum_3(metadata_3) -> schemas.Datum:
    return schemas.Datum(
        uid=datum_uid3,
        dataset=dset_name,
        metadata=metadata_3,
    )


@pytest.fixture
def datum_4(metadata_4) -> schemas.Datum:
    return schemas.Datum(
        uid=datum_uid4,
        dataset=dset_name,
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
):
    crud.create_dataset(
        db=db,
        dataset=schemas.Dataset(
            name=dset_name,
            metadata=metadata_1,
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
):
    crud.create_model(
        db=db,
        model=schemas.Model(
            name=model_name1,
            metadata=metadata_1,
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
    # Q: Get names for datasets where label class=cat exists in groundtruths.
    f = schemas.Filter(
        labels=schemas.LabelFilter(
            labels=[schemas.Label(key="class", value="cat")]
        )
    )
    q = Query(models.Dataset.name).filter(f).groundtruths()
    dataset_names = db.query(q).distinct().all()
    assert len(dataset_names) == 1
    assert (dset_name,) in dataset_names

    # Q: Get names for datasets where label=tree exists in groundtruths
    f = schemas.Filter(
        labels=schemas.LabelFilter(
            labels=[schemas.Label(key="class", value="tree")]
        )
    )
    q = Query(models.Dataset.name).filter(f).any()
    dataset_names = db.query(q).distinct().all()
    assert len(dataset_names) == 0


def test_query_models(
    db: Session,
    model_sim,
):
    # Q: Get names for all models that operate over a dataset that meets the name equality
    f = schemas.Filter(
        datasets=schemas.DatasetFilter(
            names=[dset_name],
        )
    )
    q = Query(models.Model.name).filter(f).any()
    model_names = db.query(q).distinct().all()
    assert len(model_names) == 2
    assert (model_name1,) in model_names
    assert (model_name2,) in model_names

    # Q: Get names for models where label=cat exists in predictions
    f = schemas.Filter(
        labels=schemas.LabelFilter(
            labels=[schemas.Label(key="class", value="cat")]
        )
    )
    q = Query(models.Model.name).filter(f).any()
    model_names = db.query(q).distinct().all()
    assert len(model_names) == 2
    assert (model_name1,) in model_names
    assert (model_name2,) in model_names

    # Q: Get names for models where label=tree exists in predictions
    f = schemas.Filter(
        labels=schemas.LabelFilter(
            labels=[schemas.Label(key="class", value="tree")]
        )
    )
    q = Query(models.Model.name).filter(f).any()
    model_names = db.query(q).distinct().all()
    assert len(model_names) == 0

    # Q: Get names for models that operate over dataset.
    f = schemas.Filter(datasets=schemas.DatasetFilter(names=[dset_name]))
    q = Query(models.Model.name).filter(f).any()
    model_names = db.query(q).distinct().all()
    assert len(model_names) == 2
    assert (model_name1,) in model_names
    assert (model_name2,) in model_names

    # Q: Get names for models that operate over dataset that doesn't exist.
    f = schemas.Filter(datasets=schemas.DatasetFilter(names=["invalid"]))
    q = Query(models.Model.name).filter(f).any()
    model_names = db.query(q).distinct().all()
    assert len(model_names) == 0

    # Q: Get models with metadatum with `numeric` > 0.5.
    f = schemas.Filter(
        models=schemas.ModelFilter(
            metadata=[
                schemas.MetadatumFilter(
                    key="numeric",
                    comparison=schemas.NumericFilter(
                        value=0.5,
                        operator=">",
                    ),
                )
            ]
        )
    )
    q = Query(models.Model.name).filter(f).any()
    model_names = db.query(q).distinct().all()
    assert len(model_names) == 1
    assert (model_name2,) in model_names

    # Q: Get models with metadatum with `numeric` < 0.5.
    f = schemas.Filter(
        models=schemas.ModelFilter(
            metadata=[
                schemas.MetadatumFilter(
                    key="numeric",
                    comparison=schemas.NumericFilter(
                        value=0.5,
                        operator="<",
                    ),
                )
            ]
        )
    )
    q = Query(models.Model.name).filter(f).any()
    model_names = db.query(q).distinct().all()
    assert len(model_names) == 1
    assert (model_name1,) in model_names


def test_query_by_metadata(
    db: Session,
    model_sim,
):
    # Q: Get models with metadatum with `numeric` < 0.5 and `str` == 'abc'.
    f = schemas.Filter(
        datums=schemas.DatumFilter(
            metadata=[
                schemas.MetadatumFilter(
                    key="numeric",
                    comparison=schemas.NumericFilter(
                        value=0.5,
                        operator="<",
                    ),
                ),
                schemas.MetadatumFilter(
                    key="str",
                    comparison=schemas.StringFilter(
                        value="abc",
                        operator="==",
                    ),
                ),
            ]
        )
    )
    q = Query(models.Datum.uid).filter(f).any()
    datum_uids = db.query(q).distinct().all()
    assert len(datum_uids) == 1
    assert (datum_uid1,) in datum_uids

    # Q: Get models with metadatum with `numeric` > 0.5 and `str` == 'abc'.
    f = schemas.Filter(
        datums=schemas.DatumFilter(
            metadata=[
                schemas.MetadatumFilter(
                    key="numeric",
                    comparison=schemas.NumericFilter(
                        value=0.5,
                        operator=">",
                    ),
                ),
                schemas.MetadatumFilter(
                    key="str",
                    comparison=schemas.StringFilter(
                        value="abc",
                        operator="==",
                    ),
                ),
            ]
        )
    )
    q = Query(models.Datum.uid).filter(f).any()
    datum_uids = db.query(q).distinct().all()
    assert len(datum_uids) == 1
    assert (datum_uid2,) in datum_uids

    # Q: Get models with metadatum with `numeric` < 0.5 and `str` == 'xyz'.
    f = schemas.Filter(
        datums=schemas.DatumFilter(
            metadata=[
                schemas.MetadatumFilter(
                    key="numeric",
                    comparison=schemas.NumericFilter(
                        value=0.5,
                        operator="<",
                    ),
                ),
                schemas.MetadatumFilter(
                    key="str",
                    comparison=schemas.StringFilter(
                        value="xyz",
                        operator="==",
                    ),
                ),
            ]
        )
    )
    q = Query(models.Datum.uid).filter(f).any()
    datum_uids = db.query(q).distinct().all()
    assert len(datum_uids) == 1
    assert (datum_uid3,) in datum_uids

    # Q: Get models with metadatum with `numeric` > 0.5 and `str` == 'xyz'.
    f = schemas.Filter(
        datums=schemas.DatumFilter(
            metadata=[
                schemas.MetadatumFilter(
                    key="numeric",
                    comparison=schemas.NumericFilter(
                        value=0.5,
                        operator=">",
                    ),
                ),
                schemas.MetadatumFilter(
                    key="str",
                    comparison=schemas.StringFilter(
                        value="xyz",
                        operator="==",
                    ),
                ),
            ]
        )
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
    f = schemas.Filter(
        labels=schemas.LabelFilter(
            labels=[schemas.Label(key="class", value="cat")]
        )
    )
    q = Query(models.Datum.uid).filter(f).groundtruths()
    datum_uids = db.query(q).distinct().all()
    assert len(datum_uids) == 2
    assert (datum_uid1,) in datum_uids
    assert (datum_uid2,) in datum_uids

    # Q: Get datums with groundtruth labels of "dog"
    f = schemas.Filter(
        labels=schemas.LabelFilter(
            labels=[schemas.Label(key="class", value="dog")]
        )
    )
    q = Query(models.Datum.uid).filter(f).groundtruths()
    datum_uids = db.query(q).distinct().all()
    assert len(datum_uids) == 2
    assert (datum_uid3,) in datum_uids
    assert (datum_uid4,) in datum_uids

    # Q: Get datums with prediction labels of "cat"
    f = schemas.Filter(
        labels=schemas.LabelFilter(
            labels=[schemas.Label(key="class", value="cat")]
        )
    )
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
        models=schemas.ModelFilter(
            names=[model_name1],
        ),
        labels=schemas.LabelFilter(
            labels=[schemas.Label(key="class", value="dog")]
        ),
        predictions=schemas.PredictionFilter(
            score=schemas.NumericFilter(
                value=0.9,
                operator=">",
            )
        ),
    )
    q = Query(models.Datum.uid).filter(f).predictions()
    datum_uids = db.query(q).distinct().all()
    assert len(datum_uids) == 2
    assert (datum_uid3,) in datum_uids
    assert (datum_uid4,) in datum_uids

    # Q: Get datums that `model1` has `bounding_box` annotations for with label `dog` and prediction score > 0.75.
    f = schemas.Filter(
        models=schemas.ModelFilter(
            names=[model_name1],
        ),
        labels=schemas.LabelFilter(
            labels=[schemas.Label(key="class", value="dog")]
        ),
        predictions=schemas.PredictionFilter(
            score=schemas.NumericFilter(
                value=0.75,
                operator=">",
            )
        ),
        annotations=schemas.AnnotationFilter(
            annotation_types=[enums.AnnotationType.BOX]
        ),
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
        annotations=schemas.AnnotationFilter(
            geometry=schemas.GeometricFilter(
                type=enums.AnnotationType.BOX,
                area=schemas.NumericFilter(
                    value=75,
                    operator=">",
                ),
            )
        )
    )

    # Q: Get `bounding_box` annotations that have an area > 75.
    q = Query(models.Annotation).filter(f).any()
    annotations = db.query(q).all()
    assert len(annotations) == 12

    # Q: Get `bounding_box` annotations from `model1` that have an area > 75.
    f.models = schemas.ModelFilter(names=[model_name1])
    q = Query(models.Annotation).filter(f).any()
    annotations = db.query(q).all()
    assert len(annotations) == 4


def test_multiple_tables_in_args(
    db: Session,
    model_sim,
):
    f = schemas.Filter(
        datums=schemas.DatumFilter(uids=[datum_uid1]),
    )

    # Q: Get model + dataset name pairings for a datum with `uid1` using the full tables
    q = Query(models.Model, models.Dataset).filter(f).any()
    name_pairings = [
        (
            pair[1],
            pair[5],
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
