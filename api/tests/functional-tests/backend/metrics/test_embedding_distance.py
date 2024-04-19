import pytest
from sqlalchemy.orm import Session

from valor_api import enums, schemas
from valor_api.backend import core
from valor_api.backend.metrics.embed_distance import (
    _compute_embedding_distance,
)


@pytest.fixture
def create_dataset(db: Session):
    core.create_dataset(db=db, dataset=schemas.Dataset(name="testdataset"))
    core.create_groundtruth(
        db=db,
        groundtruth=schemas.GroundTruth(
            dataset_name="testdataset",
            datum=schemas.Datum(uid="1"),
            annotations=[],
        ),
    )
    core.create_groundtruth(
        db=db,
        groundtruth=schemas.GroundTruth(
            dataset_name="testdataset",
            datum=schemas.Datum(uid="2"),
            annotations=[],
        ),
    )
    core.create_groundtruth(
        db=db,
        groundtruth=schemas.GroundTruth(
            dataset_name="testdataset",
            datum=schemas.Datum(uid="3"),
            annotations=[],
        ),
    )


@pytest.fixture
def create_model(db: Session):
    core.create_model(db=db, model=schemas.Model(name="testmodel"))
    core.create_prediction(
        db=db,
        prediction=schemas.Prediction(
            dataset_name="testdataset",
            model_name="testmodel",
            datum=schemas.Datum(uid="1"),
            annotations=[
                schemas.Annotation(
                    task_type=enums.TaskType.EMBEDDING_DISTANCE,
                    labels=[schemas.Label(key="k1", value="v1")],
                    embedding=[1, 0],
                )
            ],
        ),
    )
    core.create_prediction(
        db=db,
        prediction=schemas.Prediction(
            dataset_name="testdataset",
            model_name="testmodel",
            datum=schemas.Datum(uid="2"),
            annotations=[
                schemas.Annotation(
                    task_type=enums.TaskType.EMBEDDING_DISTANCE,
                    labels=[schemas.Label(key="k1", value="v1")],
                    embedding=[0.9, -0.1],
                )
            ],
        ),
    )
    core.create_prediction(
        db=db,
        prediction=schemas.Prediction(
            dataset_name="testdataset",
            model_name="testmodel",
            datum=schemas.Datum(uid="3"),
            annotations=[
                schemas.Annotation(
                    task_type=enums.TaskType.EMBEDDING_DISTANCE,
                    labels=[schemas.Label(key="k1", value="v1")],
                    embedding=[1, 0],
                )
            ],
        ),
    )


def test__(db: Session, create_dataset, create_model):
    x = _compute_embedding_distance(
        db, schemas.Filter(), schemas.Filter(model_names=["testmodel"])
    )
    print(x)
