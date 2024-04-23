from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from valor_api import crud, enums, schemas
from valor_api.backend import core


def test_restart_failed_evaluation(db: Session):
    crud.create_dataset(db=db, dataset=schemas.Dataset(name="dataset"))
    crud.create_model(db=db, model=schemas.Model(name="model"))
    crud.finalize(db=db, dataset_name="dataset")

    # create evaluation and overwrite status to failed
    evaluations1 = core.create_or_get_evaluations(
        db=db,
        job_request=schemas.EvaluationRequest(
            model_names=["model"],
            datum_filter=schemas.Filter(dataset_names=["dataset"]),
            parameters=schemas.EvaluationParameters(
                task_type=enums.TaskType.CLASSIFICATION
            ),
            meta=None,
        ),
        allow_retries=False,
    )
    assert len(evaluations1) == 1
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
            model_names=["model"],
            datum_filter=schemas.Filter(dataset_names=["dataset"]),
            parameters=schemas.EvaluationParameters(
                task_type=enums.TaskType.CLASSIFICATION
            ),
            meta=None,
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
            model_names=["model"],
            datum_filter=schemas.Filter(dataset_names=["dataset"]),
            parameters=schemas.EvaluationParameters(
                task_type=enums.TaskType.CLASSIFICATION
            ),
            meta=None,
        ),
        allow_retries=True,
    )
    assert len(evaluations3) == 1
    assert evaluations3[0].status == enums.EvaluationStatus.DONE
    assert evaluations3[0].id == evaluations1[0].id
