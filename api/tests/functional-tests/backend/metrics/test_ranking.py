import pytest
from sqlalchemy.orm import Session

from valor_api import crud, enums, schemas
from valor_api.backend import models
from valor_api.backend.core import create_or_get_evaluations
from valor_api.backend.metrics.ranking import compute_ranking_metrics


@pytest.fixture
def ranking_test_data(
    db: Session,
    dataset_name: str,
    model_name: str,
    groundtruth_ranking: list[schemas.GroundTruth],
    prediction_ranking: list[schemas.Prediction],
):
    crud.create_dataset(
        db=db,
        dataset=schemas.Dataset(
            name=dataset_name,
            metadata={"type": "image"},
        ),
    )
    for gt in groundtruth_ranking:
        crud.create_groundtruth(db=db, groundtruth=gt)
    crud.finalize(db=db, dataset_name=dataset_name)

    crud.create_model(
        db=db,
        model=schemas.Model(
            name=model_name,
            metadata={"type": "image"},
        ),
    )
    for pd in prediction_ranking:
        crud.create_prediction(db=db, prediction=pd)
    crud.finalize(db=db, dataset_name=dataset_name, model_name=model_name)

    assert len(db.query(models.Datum).all()) == 2
    assert len(db.query(models.Annotation).all()) == 9
    assert len(db.query(models.Label).all()) == 4
    assert len(db.query(models.GroundTruth).all()) == 2
    assert len(db.query(models.Prediction).all()) == 6


def test_ranking(
    db: Session,
    dataset_name: str,
    model_name: str,
    ranking_test_data,
):
    # default request
    job_request = schemas.EvaluationRequest(
        model_names=[model_name],
        datum_filter=schemas.Filter(dataset_names=[dataset_name]),
        parameters=schemas.EvaluationParameters(
            task_type=enums.TaskType.RANKING,
        ),
        meta={},
    )

    # creates evaluation job
    evaluations = create_or_get_evaluations(db=db, job_request=job_request)
    assert len(evaluations) == 1
    assert evaluations[0].status == enums.EvaluationStatus.PENDING

    # computation, normally run as background task
    _ = compute_ranking_metrics(
        db=db,
        evaluation_id=evaluations[0].id,
    )

    # get evaluations
    evaluations = create_or_get_evaluations(db=db, job_request=job_request)
    assert len(evaluations) == 1
    assert evaluations[0].status in {
        enums.EvaluationStatus.RUNNING,
        enums.EvaluationStatus.DONE,
    }

    assert evaluations[0].metrics is not None

    metrics = {}
    for metric in evaluations[0].metrics:
        assert metric.parameters  # handles type error
        metrics[metric.parameters["label_key"]] = metric.value

    expected_metrics = {
        "k1": 0.5,  # (1 + .33 + .166 + .5) / 4
        "k2": 0,  # no predictions for this key
    }

    for key, value in metrics.items():
        assert expected_metrics[key] == value

    for key, value in expected_metrics.items():
        assert metrics[key] == value


def test_ranking_with_label_map(
    db: Session,
    dataset_name: str,
    model_name: str,
    ranking_test_data,
):
    # test label map
    label_map = [[["k2", "v2"], ["k1", "v1"]]]

    # default request
    job_request = schemas.EvaluationRequest(
        model_names=[model_name],
        datum_filter=schemas.Filter(dataset_names=[dataset_name]),
        parameters=schemas.EvaluationParameters(
            task_type=enums.TaskType.RANKING, label_map=label_map
        ),
        meta={},
    )

    # creates evaluation job
    evaluations = create_or_get_evaluations(db=db, job_request=job_request)
    assert len(evaluations) == 1
    assert evaluations[0].status == enums.EvaluationStatus.PENDING

    # computation, normally run as background task
    _ = compute_ranking_metrics(db=db, evaluation_id=evaluations[0].id)

    # get evaluations
    evaluations = create_or_get_evaluations(db=db, job_request=job_request)
    assert len(evaluations) == 1
    assert evaluations[0].status in {
        enums.EvaluationStatus.RUNNING,
        enums.EvaluationStatus.DONE,
    }

    assert evaluations[0].metrics is not None

    metrics = {}
    for metric in evaluations[0].metrics:
        assert metric.parameters  # handles type error
        metrics[metric.parameters["label_key"]] = metric.value

    expected_metrics = {
        "k1": 0.4,  # (1 + .33 + .166 + .5 + 0) / 5,
        "k2": 0,  # still no predictions for this key
    }

    for key, value in metrics.items():
        assert expected_metrics[key] == value

    for key, value in expected_metrics.items():
        assert metrics[key] == value
