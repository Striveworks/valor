import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from velour_api import crud, schemas
from velour_api.backend import models
from velour_api.backend.graph import Graph
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
    """Uploads a dataset, do not include in args if using model_sim"""
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
    """Uploads a dataset and model"""
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


def test_generate_query(
    db: Session,
    model_sim,
):
    # instantiate graph model
    graph = Graph()

    # Q: Get model ids for all models that operate over a dataset that meets the name equality
    target = graph.model
    filters = {graph.dataset: [models.Dataset.name == dset_name]}
    generated_query = graph.generate_query(target, filters)
    model_ids = db.scalars(generated_query).all()
    assert len(model_ids) == 2
    model_names = [
        db.scalar(select(models.Model).where(models.Model.id == id)).name
        for id in model_ids
    ]
    assert model_name1 in model_names
    assert model_name2 in model_names


def test_generate_query_extremities(
    db: Session,
    model_sim,
):
    # instantiate graph model
    graph = Graph()

    # checking that this is how the data was initialized
    assert gt_label == "dog"
    assert pd_label == "cat"

    # Q: Get prediction score(s) where the groundtruth has label of "dog" and prediction has label of "cat"
    #       constrain by dataset_name and model_name.
    filters = {
        graph.dataset: [models.Dataset.name == dset_name],
        graph.model: [models.Model.name == model_name1],
        graph.groundtruth_label: [models.Label.value == "dog"],
        graph.prediction_label: [models.Label.value == "cat"],
    }
    generated_query = graph.generate_query(graph.prediction, filters)
    prediction_ids = db.scalars(generated_query).all()
    scores = [
        db.scalar(
            select(models.Prediction).where(models.Prediction.id == id)
        ).score
        for id in prediction_ids
    ]
    assert len(scores) == 1
    assert scores == [0.9]

    # Q: Get prediction score(s) where the groundtruth has label of "dog" and prediction has label of "dog"
    #       constrain by dataset_name.
    filters = {
        graph.dataset: [models.Dataset.name == dset_name],
        graph.groundtruth_label: [models.Label.value == "dog"],
        graph.prediction_label: [models.Label.value == "dog"],
    }
    generated_query = graph.generate_query(graph.prediction, filters)
    prediction_ids = db.scalars(generated_query).all()
    scores = [
        db.scalar(
            select(models.Prediction).where(models.Prediction.id == id)
        ).score
        for id in prediction_ids
    ]
    assert len(scores) == 2
    assert 0.1 in scores
    assert 0.8 in scores
