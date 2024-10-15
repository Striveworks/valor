import pytest
from sqlalchemy.orm import Session

from valor_api import enums, schemas
from valor_api.backend import core
from valor_api.backend.metrics.metric_utils import validate_computation


@validate_computation
def _test_successful_computation(db, evaluation_id, *args, **kwargs):
    pass


@validate_computation
def _test_failed_computation(db, evaluation_id, *args, **kwargs):
    raise RuntimeError("This is my test function.")


def test_validate_computation(
    db: Session,
    created_dataset: str,
    created_model: str,
):
    # create evaluation
    core.set_dataset_status(db, created_dataset, enums.TableStatus.FINALIZED)
    created = core.create_or_get_evaluations(
        db,
        schemas.EvaluationRequest(
            dataset_names=[created_dataset],
            model_names=[created_model],
            parameters=schemas.EvaluationParameters(
                task_type=enums.TaskType.CLASSIFICATION,
            ),
        ),
    )
    assert len(created) == 1
    evaluation_id = created[0].id
    assert (
        core.get_evaluation_status(db, evaluation_id)
        == enums.EvaluationStatus.PENDING
    )

    with pytest.raises(RuntimeError) as e:
        _test_successful_computation(db, evaluation_id)
    assert "db" in str(e)
    with pytest.raises(RuntimeError) as e:
        _test_successful_computation(1, evaluation_id=evaluation_id)
    assert "db" in str(e)
    with pytest.raises(RuntimeError) as e:
        _test_successful_computation(evaluation_id, db=db)
    assert "evaluation_id" in str(e)

    with pytest.raises(TypeError) as e:
        _test_successful_computation(db=1, evaluation_id=evaluation_id)
    assert "db" in str(e)
    with pytest.raises(TypeError) as e:
        _test_successful_computation(db=db, evaluation_id="12343")
    assert "evaluation_id" in str(e)

    with pytest.raises(RuntimeError) as e:
        _test_failed_computation(db=db, evaluation_id=evaluation_id)
    assert "This is my test function." in str(e)
    assert (
        core.get_evaluation_status(db, evaluation_id)
        == enums.EvaluationStatus.FAILED
    )

    _test_successful_computation(db=db, evaluation_id=evaluation_id)
    assert (
        core.get_evaluation_status(db, evaluation_id)
        == enums.EvaluationStatus.DONE
    )
