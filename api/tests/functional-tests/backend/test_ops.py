import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from velour_api import crud, enums, schemas
from velour_api.backend import Query, models
from velour_api.enums import TaskType

dset_name = "dataset1"
model_name1 = "model1"
model_name2 = "model2"
datum_uid = "uid1"
label_key = "class"
gt_label = "dog"
pd_label = "cat"


@pytest.fixture
def dataset_sim(db: Session):
    crud.create_dataset(
        db=db,
        dataset=schemas.Dataset(name=dset_name),
    )
    crud.create_groundtruth(
        db=db,
        groundtruth=schemas.GroundTruth(
            datum=schemas.Datum(
                uid=datum_uid,
                dataset=dset_name,
                metadata=[
                    schemas.Metadatum(key="type", value="image"),
                    schemas.Metadatum(key="number", value=5),
                ],
            ),
            annotations=[
                schemas.Annotation(
                    task_type=TaskType.CLASSIFICATION,
                    labels=[schemas.Label(key=label_key, value=gt_label)],
                ),
                schemas.Annotation(
                    task_type=TaskType.DETECTION,
                    labels=[
                        schemas.Label(
                            key=label_key,
                            value=pd_label,
                        ),
                        schemas.Label(
                            key=label_key,
                            value=gt_label,
                        ),
                    ],
                    bounding_box=schemas.BoundingBox.from_extrema(
                        0, 0, 10, 10
                    ),
                    metadata=[
                        schemas.Metadatum(key="angle", value="45.55"),
                        schemas.Metadatum(key="height", value="100000.5"),
                    ],
                ),
                schemas.Annotation(
                    task_type=TaskType.DETECTION,
                    labels=[
                        schemas.Label(
                            key=label_key,
                            value=pd_label,
                        ),
                        schemas.Label(
                            key=label_key,
                            value=gt_label,
                        ),
                    ],
                    bounding_box=schemas.BoundingBox.from_extrema(0, 0, 5, 10),
                    metadata=[
                        schemas.Metadatum(key="angle", value="85.55"),
                        schemas.Metadatum(key="height", value="500000.5"),
                    ],
                ),
            ],
        ),
    )
    crud.finalize(db=db, dataset_name=dset_name)


@pytest.fixture
def model_sim(
    db: Session,
    dataset_sim,
):
    crud.create_model(
        db=db,
        model=schemas.Model(name=model_name1),
    )
    crud.create_prediction(
        db=db,
        prediction=schemas.Prediction(
            model=model_name1,
            datum=schemas.Datum(
                uid=datum_uid,
                dataset=dset_name,
                metadata=[
                    schemas.Metadatum(key="type", value="image"),
                    schemas.Metadatum(key="angle", value="45.55"),
                    schemas.Metadatum(key="height", value="100000.5"),
                ],
            ),
            annotations=[
                schemas.Annotation(
                    task_type=TaskType.CLASSIFICATION,
                    labels=[
                        schemas.Label(
                            key=label_key, value=pd_label, score=0.9
                        ),
                        schemas.Label(
                            key=label_key, value=gt_label, score=0.1
                        ),
                    ],
                    metadata=[schemas.Metadatum(key="number", value=0.1)],
                ),
                schemas.Annotation(
                    task_type=TaskType.DETECTION,
                    labels=[
                        schemas.Label(
                            key=label_key, value=pd_label, score=0.6
                        ),
                        schemas.Label(
                            key=label_key, value=gt_label, score=0.4
                        ),
                    ],
                    bounding_box=schemas.BoundingBox.from_extrema(
                        0, 0, 10, 10
                    ),
                    metadata=[schemas.Metadatum(key="number", value=0.5)],
                ),
                schemas.Annotation(
                    task_type=TaskType.DETECTION,
                    labels=[
                        schemas.Label(
                            key=label_key, value=pd_label, score=0.55
                        ),
                        schemas.Label(
                            key=label_key, value=gt_label, score=0.45
                        ),
                    ],
                    bounding_box=schemas.BoundingBox.from_extrema(0, 0, 5, 10),
                    metadata=[schemas.Metadatum(key="number", value=0.9)],
                ),
            ],
        ),
    )
    crud.finalize(db=db, dataset_name=dset_name, model_name=model_name1)

    crud.create_model(
        db=db,
        model=schemas.Model(name=model_name2),
    )
    crud.create_prediction(
        db=db,
        prediction=schemas.Prediction(
            model=model_name2,
            datum=schemas.Datum(
                uid=datum_uid,
                dataset=dset_name,
                metadata=[
                    schemas.Metadatum(key="type", value="image"),
                    schemas.Metadatum(key="number", value=5),
                ],
            ),
            annotations=[
                schemas.Annotation(
                    task_type=TaskType.CLASSIFICATION,
                    labels=[
                        schemas.Label(
                            key=label_key, value=pd_label, score=0.2
                        ),
                        schemas.Label(
                            key=label_key, value=gt_label, score=0.8
                        ),
                    ],
                )
            ],
        ),
    )
    crud.finalize(db=db, dataset_name=dset_name, model_name=model_name2)


def test_basic(
    db: Session,
    model_sim,
):
    # Q: Get model ids for all models that operate over a dataset that meets the name equality
    f = schemas.Filter(
        datasets=schemas.DatasetFilter(
            names=[dset_name],
        )
    )
    q = Query(models.Model.name).filter(f).query()
    model_names = db.query(q).distinct().all()
    assert len(model_names) == 2
    assert (model_name1,) in model_names
    assert (model_name2,) in model_names


def test_annotation_query(
    db: Session,
    model_sim,
):
    # Q: Get annotations with bounding box area >= 50
    f = schemas.Filter(
        annotations=schemas.AnnotationFilter(
            geometry=schemas.GeometricFilter(
                type=enums.AnnotationType.BOX,
                area=schemas.NumericFilter(value=50, operator=">="),
            )
        )
    )
    q = Query(models.Annotation).filter(f).query()
    annotations = db.query(q).all()
    assert len(annotations) == 2
    areas = [
        db.scalar(
            select(func.ST_Area(models.Annotation.box)).where(
                models.Annotation.id == annotation.id
            )
        )
        for annotation in annotations
    ]
    assert set(areas) == {50, 100}

    # Q: Get annotations with bounding box area > 50
    f = schemas.Filter(
        annotations=schemas.AnnotationFilter(
            geometry=schemas.GeometricFilter(
                type=enums.AnnotationType.BOX,
                area=schemas.NumericFilter(value=50, operator=">"),
            )
        )
    )
    q = Query(models.Annotation).filter(f).query()
    annotations = db.query(q).all()
    assert len(annotations) == 1
    areas = [
        db.scalar(
            select(func.ST_Area(models.Annotation.box)).where(
                models.Annotation.id == annotation.id
            )
        )
        for annotation in annotations
    ]
    assert set(areas) == {100}


def test_numeric_metadata_query(
    db: Session,
    model_sim,
):
    # Q: Get annotations with metadatum key `number` > 0.5
    f = schemas.Filter(
        annotations=schemas.AnnotationFilter(
            metadata=[
                schemas.MetadatumFilter(
                    key="number",
                    comparison=schemas.NumericFilter(value=0.5, operator=">"),
                )
            ]
        )
    )
    q = Query(models.Annotation).filter(f).query()
    numbers = [annotation.meta["number"] for annotation in db.query(q).all()]
    assert len(numbers) == 1
    assert numbers == [0.9]

    # Q: Get annotations with metadatum key `number` >= 0.5
    f = schemas.Filter(
        annotations=schemas.AnnotationFilter(
            metadata=[
                schemas.MetadatumFilter(
                    key="number",
                    comparison=schemas.NumericFilter(value=0.5, operator=">="),
                )
            ]
        )
    )
    q = Query(models.Annotation).filter(f).query()
    numbers = [annotation.meta["number"] for annotation in db.query(q).all()]
    assert len(numbers) == 2
    assert set(numbers) == {0.5, 0.9}

    # Q: Get annotations with metadatum key `number` < 0.5
    f = schemas.Filter(
        annotations=schemas.AnnotationFilter(
            metadata=[
                schemas.MetadatumFilter(
                    key="number",
                    comparison=schemas.NumericFilter(value=0.5, operator="<"),
                )
            ]
        )
    )
    q = Query(models.Annotation).filter(f).query()
    numbers = [annotation.meta["number"] for annotation in db.query(q).all()]
    assert len(numbers) == 1
    assert set(numbers) == {0.1}


# TODO - Need to implement automatic subquerying
def test_Query_extremities(
    db: Session,
    model_sim,
):
    # checking that this is how the data was initialized
    assert gt_label == "dog"
    assert pd_label == "cat"

    # Q: Get prediction score(s) where the groundtruth has label of "dog" and prediction has label of "cat"
    #       constrain by dataset_name and model_name.
    gt_filter = schemas.Filter(
        datasets=schemas.DatasetFilter(names=[dset_name]),
        labels=schemas.LabelFilter(
            labels=[schemas.Label(key=label_key, value="dog")]
        ),
    )
    subq = Query(models.Datum).filter(gt_filter).query()
    ids = [datum.id for datum in db.query(subq).all()]

    pd_filter = schemas.Filter(
        datasets=schemas.DatasetFilter(names=[dset_name]),
        models=schemas.ModelFilter(names=[model_name1]),
        datums=schemas.DatumFilter(ids=ids),
        labels=schemas.LabelFilter(
            labels=[schemas.Label(key=label_key, value="cat")]
        ),
    )
    q = Query(models.Prediction).filter(pd_filter).query()
    scores = [prediction.score for prediction in db.query(q).all()]
    assert len(scores) == 3
    assert set(scores) == set([0.9, 0.6, 0.55])

    # Q: Get prediction score(s) where the groundtruth has label of "dog" and prediction has label of "dog"
    #       constrain by dataset_name and model_name.
    pd_filter = schemas.Filter(
        datasets=schemas.DatasetFilter(names=[dset_name]),
        models=schemas.ModelFilter(names=[model_name1]),
        datums=schemas.DatumFilter(ids=ids),
        labels=schemas.LabelFilter(
            labels=[schemas.Label(key=label_key, value="dog")]
        ),
    )
    q = Query(models.Prediction).filter(pd_filter).query()
    scores = [prediction.score for prediction in db.query(q).all()]
    assert len(scores) == 3
    assert set(scores) == set([0.1, 0.4, 0.45])

    # Q: Get prediction score(s) where both groundtruth and prediction labels are of "dog" and constrain by dataset_name.
    pd_filter = schemas.Filter(
        datasets=schemas.DatasetFilter(names=[dset_name]),
        datums=schemas.DatumFilter(ids=ids),
        labels=schemas.LabelFilter(
            labels=[schemas.Label(key=label_key, value="dog")]
        ),
    )
    q = Query(models.Prediction).filter(pd_filter).query()
    scores = [prediction.score for prediction in db.query(q).all()]
    assert len(scores) == 4
    assert set(scores) == set([0.1, 0.4, 0.45, 0.8])
