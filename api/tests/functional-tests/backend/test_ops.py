import pytest
from sqlalchemy.orm import Session

from velour_api import crud, schemas
from velour_api.backend import models
from velour_api.backend.ops import Query
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
            ),
            annotations=[
                schemas.Annotation(
                    task_type=TaskType.CLASSIFICATION,
                    labels=[schemas.Label(key=label_key, value=gt_label)],
                )
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
                )
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


def test_Query(
    db: Session,
    model_sim,
):
    """Same underlying query as functional-tests/backend/test_graph.py::test_generate_query"""

    # Q: Get model ids for all models that operate over a dataset that meets the name equality
    f = schemas.Filter(
        datasets=schemas.DatasetFilter(
            names=[dset_name],
        )
    )
    q = Query(models.Model).filter(f).query()
    model_names = [model.name for model in db.query(q).all()]
    assert len(model_names) == 2
    assert model_name1 in model_names
    assert model_name2 in model_names


def test_Query_extremities(
    db: Session,
    model_sim,
):
    """Same underlying query as functional-tests/backend/test_graph.py::test_generate_query_extremities"""

    # checking that this is how the data was initialized
    assert gt_label == "dog"
    assert pd_label == "cat"

    # Q: Get prediction score(s) where the groundtruth has label of "dog" and prediction has label of "cat"
    #       constrain by dataset_name and model_name.
    f = schemas.Filter(
        datasets=schemas.DatasetFilter(names=[dset_name]),
        models=schemas.ModelFilter(names=[model_name1]),
        groundtruth_labels=schemas.LabelFilter(
            labels=[schemas.Label(key=label_key, value="dog")]
        ),
        prediction_labels=schemas.LabelFilter(
            labels=[schemas.Label(key=label_key, value="cat")]
        ),
    )
    q = Query(models.Prediction).filter(f).query()
    scores = [prediction.score for prediction in db.query(q).all()]
    assert len(scores) == 1
    assert scores == [0.9]

    # Q: Get prediction score(s) where the groundtruth has label of "dog" and prediction has label of "dog"
    #       constrain by dataset_name.
    f = schemas.Filter(
        datasets=schemas.DatasetFilter(names=[dset_name]),
        groundtruth_labels=schemas.LabelFilter(
            labels=[schemas.Label(key=label_key, value="dog")]
        ),
        prediction_labels=schemas.LabelFilter(
            labels=[schemas.Label(key=label_key, value="dog")]
        ),
    )
    q = Query(models.Prediction).filter(f).query()
    scores = [prediction.score for prediction in db.query(q).all()]
    assert len(scores) == 2
    assert 0.1 in scores
    assert 0.8 in scores
