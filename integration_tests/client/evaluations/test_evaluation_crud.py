import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

# client
from valor import (
    Client,
    Dataset,
    Filter,
    GroundTruth,
    Label,
    Model,
    Prediction,
)
from valor.enums import EvaluationStatus
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


def test_delete_evaluation_scope(
    client: Client,
    gt_clfs: list[GroundTruth],
    pred_clfs: list[Prediction],
    dataset_name: str,
    model_name: str,
):
    dataset = Dataset.create(dataset_name)
    for gt in gt_clfs:
        dataset.add_groundtruth(gt)
    dataset.finalize()

    model = Model.create(model_name)
    for pd in pred_clfs:
        model.add_prediction(dataset, pd)
    model.finalize_inferences(dataset)

    eval1 = model.evaluate_classification(dataset)
    assert eval1.wait_for_completion(timeout=30) == EvaluationStatus.DONE
    eval2 = model.evaluate_classification(
        dataset, filters=Filter(labels=Label.key == "k4")
    )
    assert eval2.wait_for_completion(timeout=30)

    assert eval1.id != eval2.id
    assert len(client.get_evaluations(evaluation_ids=[eval1.id])) == 1
    assert len(client.get_evaluations(evaluation_ids=[eval2.id])) == 1

    # delete eval 1
    client.delete_evaluation(eval1.id)

    assert len(client.get_evaluations(evaluation_ids=[eval1.id])) == 0
    assert len(client.get_evaluations(evaluation_ids=[eval2.id])) == 1

    # show that we can still make evaluations
    eval3 = model.evaluate_classification(dataset)
    assert eval3.wait_for_completion(timeout=30)

    assert eval1.id != eval2.id
    assert eval1.id != eval3.id
    assert eval2.id != eval3.id
    assert len(client.get_evaluations(evaluation_ids=[eval1.id])) == 0
    assert len(client.get_evaluations(evaluation_ids=[eval2.id])) == 1
    assert len(client.get_evaluations(evaluation_ids=[eval3.id])) == 1

    # show that eval1 was repreated in eval3
    assert eval1.id != eval3.id
    for metric in eval1.metrics:
        assert metric in eval3.metrics
    for metric in eval3.metrics:
        assert metric in eval1.metrics
