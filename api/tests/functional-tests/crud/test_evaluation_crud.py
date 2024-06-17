import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from valor_api import crud, enums, exceptions, schemas
from valor_api.backend import core, models


def test_evaluation_creation_exceptions(db: Session):
    crud.create_dataset(db=db, dataset=schemas.Dataset(name="mydataset"))
    crud.create_groundtruths(
        db=db,
        groundtruths=[
            schemas.GroundTruth(
                dataset_name="mydataset",
                datum=schemas.Datum(uid="123"),
                annotations=[
                    schemas.Annotation(
                        labels=[schemas.Label(key="class", value="dog")],
                    )
                ],
            )
        ],
    )
    crud.create_model(db=db, model=schemas.Model(name="mymodel"))

    # test no dataset exists
    with pytest.raises(exceptions.EvaluationRequestError) as e:
        core.create_or_get_evaluations(
            db=db,
            job_request=schemas.EvaluationRequest(
                dataset_names=["does_not_exist"],
                model_names=["mymodel"],
                parameters=schemas.EvaluationParameters(
                    task_type=enums.TaskType.CLASSIFICATION
                ),
            ),
            allow_retries=False,
        )
    assert "DatasetDoesNotExist" in str(e)

    # test dataset not finalized
    with pytest.raises(exceptions.EvaluationRequestError) as e:
        core.create_or_get_evaluations(
            db=db,
            job_request=schemas.EvaluationRequest(
                dataset_names=["mydataset"],
                model_names=["mymodel"],
                parameters=schemas.EvaluationParameters(
                    task_type=enums.TaskType.CLASSIFICATION
                ),
            ),
            allow_retries=False,
        )
    assert "mydataset" in str(e)

    crud.finalize(db=db, dataset_name="mydataset")

    # test no model exists
    with pytest.raises(exceptions.EvaluationRequestError) as e:
        core.create_or_get_evaluations(
            db=db,
            job_request=schemas.EvaluationRequest(
                dataset_names=["mydataset"],
                model_names=["does_not_exist"],
                parameters=schemas.EvaluationParameters(
                    task_type=enums.TaskType.CLASSIFICATION
                ),
            ),
            allow_retries=False,
        )
    assert "ModelDoesNotExist" in str(e)

    # test model not finalized
    with pytest.raises(exceptions.EvaluationRequestError) as e:
        core.create_or_get_evaluations(
            db=db,
            job_request=schemas.EvaluationRequest(
                dataset_names=["mydataset"],
                model_names=["mymodel"],
                parameters=schemas.EvaluationParameters(
                    task_type=enums.TaskType.CLASSIFICATION
                ),
            ),
            allow_retries=False,
        )
    assert "ModelNotFinalized" in str(e)

    crud.create_predictions(
        db=db,
        predictions=[
            schemas.Prediction(
                dataset_name="mydataset",
                model_name="mymodel",
                datum=schemas.Datum(uid="123"),
                annotations=[
                    schemas.Annotation(
                        labels=[
                            schemas.Label(key="class", value="dog", score=1.0)
                        ],
                    )
                ],
            )
        ],
    )

    evaluations = core.create_or_get_evaluations(
        db=db,
        job_request=schemas.EvaluationRequest(
            dataset_names=["mydataset"],
            model_names=["mymodel"],
            parameters=schemas.EvaluationParameters(
                task_type=enums.TaskType.CLASSIFICATION
            ),
        ),
        allow_retries=False,
    )
    assert len(evaluations) == 1
    assert evaluations[0].status == enums.EvaluationStatus.PENDING


def test_restart_failed_evaluation(db: Session):
    crud.create_dataset(db=db, dataset=schemas.Dataset(name="dataset"))
    crud.create_groundtruths(
        db=db,
        groundtruths=[
            schemas.GroundTruth(
                dataset_name="dataset",
                datum=schemas.Datum(uid="123"),
                annotations=[
                    schemas.Annotation(
                        labels=[schemas.Label(key="class", value="dog")],
                    )
                ],
            )
        ],
    )
    crud.create_model(db=db, model=schemas.Model(name="model"))
    crud.create_predictions(
        db=db,
        predictions=[
            schemas.Prediction(
                dataset_name="dataset",
                model_name="model",
                datum=schemas.Datum(uid="123"),
                annotations=[
                    schemas.Annotation(
                        labels=[
                            schemas.Label(key="class", value="dog", score=1.0)
                        ],
                    )
                ],
            )
        ],
    )
    crud.finalize(db=db, dataset_name="dataset")

    # create evaluation and overwrite status to failed
    evaluations1 = core.create_or_get_evaluations(
        db=db,
        job_request=schemas.EvaluationRequest(
            dataset_names=["dataset"],
            model_names=["model"],
            parameters=schemas.EvaluationParameters(
                task_type=enums.TaskType.CLASSIFICATION,
            ),
        ),
        allow_retries=False,
    )
    assert len(evaluations1) == 1
    assert evaluations1[0].status == enums.EvaluationStatus.PENDING
    try:
        evaluation = core.fetch_evaluation_from_id(
            db=db, evaluation_id=evaluations1[0].id
        )
        evaluation.status = enums.EvaluationStatus.FAILED
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise e

    # get evaluation and verify it is failed
    evaluations2 = crud.create_or_get_evaluations(
        db=db,
        job_request=schemas.EvaluationRequest(
            dataset_names=["dataset"],
            model_names=["model"],
            parameters=schemas.EvaluationParameters(
                task_type=enums.TaskType.CLASSIFICATION,
            ),
        ),
        allow_retries=False,
    )
    assert len(evaluations2) == 1
    assert evaluations2[0].status == enums.EvaluationStatus.FAILED
    assert evaluations2[0].id == evaluations1[0].id

    # get evaluation and allow retries, this should result in a finished eval
    evaluations3 = crud.create_or_get_evaluations(
        db=db,
        job_request=schemas.EvaluationRequest(
            dataset_names=["dataset"],
            model_names=["model"],
            parameters=schemas.EvaluationParameters(
                task_type=enums.TaskType.CLASSIFICATION,
            ),
        ),
        allow_retries=True,
    )
    assert len(evaluations3) == 1
    assert evaluations3[0].status == enums.EvaluationStatus.PENDING
    assert evaluations3[0].id == evaluations1[0].id

    # check that evaluation has completed
    evaluations4 = crud.create_or_get_evaluations(
        db=db,
        job_request=schemas.EvaluationRequest(
            dataset_names=["dataset"],
            model_names=["model"],
            parameters=schemas.EvaluationParameters(
                task_type=enums.TaskType.CLASSIFICATION
            ),
        ),
        allow_retries=False,
    )
    assert len(evaluations4) == 1
    assert evaluations4[0].status == enums.EvaluationStatus.DONE
    assert evaluations4[0].id == evaluations1[0].id


@pytest.fixture
def create_evaluations(db: Session):

    rows = [
        models.Evaluation(
            id=idx,
            dataset_names=["1", "2"],
            model_name=str(idx),
            parameters=schemas.EvaluationParameters(
                task_type=enums.TaskType.CLASSIFICATION
            ).model_dump(),
            filters=schemas.Filter().model_dump(),
            status=status,
        )
        for idx, status in enumerate(enums.EvaluationStatus)
    ]

    try:
        db.add_all(rows)
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise e

    yield [(row.id, row.status) for row in rows]

    for row in rows:
        try:
            db.delete(row)
        except IntegrityError:
            db.rollback()


def test_delete_evaluation(db: Session, create_evaluations):

    for idx, status in create_evaluations:
        assert (
            db.scalar(
                select(func.count(models.Evaluation.id)).where(
                    models.Evaluation.id == idx
                )
            )
            == 1
        )
        if status in {
            enums.EvaluationStatus.PENDING,
            enums.EvaluationStatus.RUNNING,
        }:
            with pytest.raises(exceptions.EvaluationRunningError):
                crud.delete_evaluation(db=db, evaluation_id=idx)
            assert (
                db.scalar(
                    select(func.count(models.Evaluation.id)).where(
                        models.Evaluation.id == idx
                    )
                )
                == 1
            )
        elif status == enums.EvaluationStatus.DELETING:
            with pytest.raises(exceptions.EvaluationDoesNotExistError):
                crud.delete_evaluation(db=db, evaluation_id=idx)
            assert (
                db.scalar(
                    select(func.count(models.Evaluation.id)).where(
                        models.Evaluation.id == idx
                    )
                )
                == 1
            )
        else:
            crud.delete_evaluation(db=db, evaluation_id=idx)
            assert (
                db.scalar(
                    select(func.count(models.Evaluation.id)).where(
                        models.Evaluation.id == idx
                    )
                )
                == 0
            )

    # check for id that doesnt exist
    with pytest.raises(exceptions.EvaluationDoesNotExistError):
        crud.delete_evaluation(db=db, evaluation_id=10000)


@pytest.fixture
def create_evaluation_with_metrics(db: Session):
    evaluation_id = 0
    number_of_metrics = 4

    evaluation = models.Evaluation(
        id=evaluation_id,
        dataset_names=["1", "2"],
        model_name="3",
        parameters=schemas.EvaluationParameters(
            task_type=enums.TaskType.CLASSIFICATION
        ).model_dump(),
        filters=schemas.Filter().model_dump(),
        status=enums.EvaluationStatus.DONE,
    )
    metrics = [
        models.Metric(
            evaluation_id=evaluation_id,
            label_id=None,
            type="Precision",
            value=float(i) / float(number_of_metrics),
            parameters=dict(),
        )
        for i in range(number_of_metrics)
    ]

    try:
        db.add(evaluation)
        db.add_all(metrics)
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise e

    yield (evaluation_id, number_of_metrics)

    for row in [evaluation, *metrics]:
        try:
            db.delete(row)
            db.commit()
        except IntegrityError:
            db.rollback()


def test_delete_evaluation_with_metrics(
    db: Session, create_evaluation_with_metrics
):
    row_id, num_metrics = create_evaluation_with_metrics

    assert num_metrics == 4
    assert (
        db.scalar(
            select(func.count(models.Evaluation.id)).where(
                models.Evaluation.id == row_id
            )
        )
        == 1
    )
    assert (
        db.scalar(
            select(func.count(models.Metric.id)).where(
                models.Metric.evaluation_id == row_id
            )
        )
        == num_metrics
    )

    crud.delete_evaluation(db=db, evaluation_id=row_id)

    assert (
        db.scalar(
            select(func.count(models.Evaluation.id)).where(
                models.Evaluation.id == row_id
            )
        )
        == 0
    )
    assert (
        db.scalar(
            select(func.count(models.Metric.id)).where(
                models.Metric.evaluation_id == row_id
            )
        )
        == 0
    )
