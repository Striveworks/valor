import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from valor_api import crud, enums, exceptions, schemas
from valor_api.backend import core


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
