import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

# client
from valor import Client
from valor.exceptions import ClientException

# api
from valor_api import enums, schemas
from valor_api.backend import models


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


def test_delete_evaluation(db: Session, client: Client, create_evaluations):

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
            with pytest.raises(ClientException) as e:
                client.delete_evaluation(evaluation_id=idx)
            assert "EvaluationRunningError" in str(e)
            assert (
                db.scalar(
                    select(func.count(models.Evaluation.id)).where(
                        models.Evaluation.id == idx
                    )
                )
                == 1
            )
        elif status == enums.EvaluationStatus.DELETING:
            with pytest.raises(ClientException) as e:
                client.delete_evaluation(evaluation_id=idx)
            assert "EvaluationDoesNotExistError" in str(e)
            assert (
                db.scalar(
                    select(func.count(models.Evaluation.id)).where(
                        models.Evaluation.id == idx
                    )
                )
                == 1
            )
        else:
            client.delete_evaluation(evaluation_id=idx)
            assert (
                db.scalar(
                    select(func.count(models.Evaluation.id)).where(
                        models.Evaluation.id == idx
                    )
                )
                == 0
            )

    # check for id that doesnt exist
    with pytest.raises(ClientException) as e:
        client.delete_evaluation(evaluation_id=10000)
    assert "EvaluationDoesNotExistError" in str(e)
