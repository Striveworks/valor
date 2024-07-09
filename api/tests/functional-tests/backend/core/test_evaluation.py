import datetime

import pytest
from pydantic import ValidationError
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from valor_api import crud, enums, exceptions, schemas
from valor_api.backend import core, models
from valor_api.backend.core.evaluation import (
    _fetch_evaluation_from_subrequest,
    validate_request,
)


@pytest.fixture
def gt_clfs_create(
    dataset_name: str,
    img1: schemas.Datum,
    img2: schemas.Datum,
) -> list[schemas.GroundTruth]:
    return [
        schemas.GroundTruth(
            dataset_name=dataset_name,
            datum=img1,
            annotations=[
                schemas.Annotation(
                    labels=[
                        schemas.Label(key="k1", value="v1"),
                        schemas.Label(key="k2", value="v2"),
                    ],
                ),
            ],
        ),
        schemas.GroundTruth(
            dataset_name=dataset_name,
            datum=img2,
            annotations=[
                schemas.Annotation(
                    labels=[schemas.Label(key="k2", value="v3")],
                ),
            ],
        ),
    ]


@pytest.fixture
def pred_clfs_create(
    dataset_name: str,
    model_name: str,
    img1: schemas.Datum,
    img2: schemas.Datum,
) -> list[schemas.Prediction]:
    return [
        schemas.Prediction(
            dataset_name=dataset_name,
            model_name=model_name,
            datum=img1,
            annotations=[
                schemas.Annotation(
                    labels=[
                        schemas.Label(key="k1", value="v1", score=0.2),
                        schemas.Label(key="k1", value="v2", score=0.8),
                        schemas.Label(key="k2", value="v4", score=1.0),
                    ],
                ),
            ],
        ),
        schemas.Prediction(
            dataset_name=dataset_name,
            model_name=model_name,
            datum=img2,
            annotations=[
                schemas.Annotation(
                    labels=[
                        schemas.Label(key="k2", value="v2", score=0.8),
                        schemas.Label(key="k2", value="v3", score=0.1),
                        schemas.Label(key="k2", value="v0", score=0.1),
                    ],
                ),
            ],
        ),
    ]


@pytest.fixture
def finalized_dataset(db: Session, created_dataset: str) -> str:
    core.set_dataset_status(
        db=db, name=created_dataset, status=enums.TableStatus.FINALIZED
    )
    return created_dataset


@pytest.fixture
def finalized_model(
    db: Session, created_dataset: str, created_model: str
) -> str:
    core.set_model_status(
        db=db,
        dataset_name=created_dataset,
        model_name=created_model,
        status=enums.TableStatus.FINALIZED,
    )
    return created_model


def test_validate_request(
    db: Session,
    dataset_name: str,
    model_name: str,
):
    # test empty dataset list
    with pytest.raises(ValidationError):
        validate_request(
            db=db,
            job_request=schemas.EvaluationRequest(
                dataset_names=[],
                model_names=[model_name],
                parameters=schemas.EvaluationParameters(
                    task_type=enums.TaskType.CLASSIFICATION
                ),
            ),
        )

    core.create_dataset(db, dataset=schemas.Dataset(name=dataset_name))

    # test empty model list
    with pytest.raises(ValidationError):
        validate_request(
            db=db,
            job_request=schemas.EvaluationRequest(
                dataset_names=[dataset_name],
                model_names=[],
                parameters=schemas.EvaluationParameters(
                    task_type=enums.TaskType.CLASSIFICATION
                ),
            ),
        )

    core.create_model(db=db, model=schemas.Model(name=model_name))

    # test dataset in state `enums.TableStatus.CREATING`
    with pytest.raises(exceptions.EvaluationRequestError) as e:
        validate_request(
            db=db,
            job_request=schemas.EvaluationRequest(
                dataset_names=[dataset_name],
                model_names=[model_name],
                parameters=schemas.EvaluationParameters(
                    task_type=enums.TaskType.CLASSIFICATION
                ),
            ),
        )
    assert "DatasetNotFinalized" in str(e)

    core.create_groundtruths(
        db=db,
        groundtruths=[
            schemas.GroundTruth(
                dataset_name=dataset_name,
                datum=schemas.Datum(uid="uid1"),
                annotations=[
                    schemas.Annotation(
                        labels=[schemas.Label(key="k1", value="v1")],
                    )
                ],
            )
        ],
    )
    core.set_dataset_status(db, dataset_name, enums.TableStatus.FINALIZED)

    # test model in state `enums.TableStatus.CREATING`
    with pytest.raises(exceptions.EvaluationRequestError) as e:
        validate_request(
            db=db,
            job_request=schemas.EvaluationRequest(
                dataset_names=[dataset_name],
                model_names=[model_name],
                parameters=schemas.EvaluationParameters(
                    task_type=enums.TaskType.CLASSIFICATION
                ),
            ),
        )
    assert "ModelNotFinalizedError" in str(e)

    # create a prediction
    # automatically finalizes over dataset
    core.create_predictions(
        db=db,
        predictions=[
            schemas.Prediction(
                dataset_name=dataset_name,
                model_name=model_name,
                datum=schemas.Datum(uid="uid1"),
                annotations=[
                    schemas.Annotation(
                        labels=[
                            schemas.Label(key="k1", value="v1", score=1.0)
                        ],
                    )
                ],
            )
        ],
    )

    # both dataset and model should be in valid finalized states
    validate_request(
        db=db,
        job_request=schemas.EvaluationRequest(
            dataset_names=[dataset_name],
            model_names=[model_name],
            parameters=schemas.EvaluationParameters(
                task_type=enums.TaskType.CLASSIFICATION
            ),
        ),
    )

    core.create_model(db=db, model=schemas.Model(name="second_model"))
    core.create_predictions(
        db=db,
        predictions=[
            schemas.Prediction(
                dataset_name=dataset_name,
                model_name="second_model",
                datum=schemas.Datum(uid="uid1"),
                annotations=[
                    schemas.Annotation(
                        labels=[
                            schemas.Label(key="k1", value="v1", score=1.0)
                        ],
                    )
                ],
            )
        ],
    )

    validate_request(
        db=db,
        job_request=schemas.EvaluationRequest(
            dataset_names=[dataset_name],
            model_names=[model_name, "second_model"],
            parameters=schemas.EvaluationParameters(
                task_type=enums.TaskType.CLASSIFICATION
            ),
        ),
    )

    core.set_model_status(
        db=db,
        dataset_name=dataset_name,
        model_name="second_model",
        status=enums.TableStatus.DELETING,
    )

    # test model in deleting state
    with pytest.raises(exceptions.EvaluationRequestError) as e:
        validate_request(
            db=db,
            job_request=schemas.EvaluationRequest(
                dataset_names=[dataset_name],
                model_names=[model_name, "second_model"],
                parameters=schemas.EvaluationParameters(
                    task_type=enums.TaskType.CLASSIFICATION
                ),
            ),
        )
    assert "second_model" in str(e)

    core.delete_model(db=db, name="second_model")
    core.set_dataset_status(
        db=db, name=dataset_name, status=enums.TableStatus.DELETING
    )

    # test dataset in deleting state
    with pytest.raises(exceptions.DatasetDoesNotExistError) as e:
        validate_request(
            db=db,
            job_request=schemas.EvaluationRequest(
                dataset_names=[dataset_name],
                model_names=[model_name, "second_model"],
                parameters=schemas.EvaluationParameters(
                    task_type=enums.TaskType.CLASSIFICATION
                ),
            ),
        )
    assert dataset_name in str(e)


def test__fetch_evaluation_from_subrequest(
    db: Session,
    finalized_dataset: str,
    finalized_model: str,
):
    # create evaluation 1
    job_request_1 = schemas.EvaluationRequest(
        dataset_names=[finalized_dataset],
        model_names=[finalized_model],
        parameters=schemas.EvaluationParameters(
            task_type=enums.TaskType.CLASSIFICATION,
        ),
    )
    created_1 = core.create_or_get_evaluations(db, job_request_1)
    assert len(created_1) == 1

    # create evaluation 2
    job_request_2 = schemas.EvaluationRequest(
        dataset_names=[finalized_dataset],
        model_names=[finalized_model],
        parameters=schemas.EvaluationParameters(
            task_type=enums.TaskType.SEMANTIC_SEGMENTATION,
        ),
    )
    created_2 = core.create_or_get_evaluations(db, job_request_2)
    assert len(created_2) == 1

    # test fetching a subrequest
    subrequest = schemas.EvaluationRequest(
        dataset_names=[finalized_dataset],
        model_names=[finalized_model],
        parameters=schemas.EvaluationParameters(
            task_type=enums.TaskType.CLASSIFICATION,
        ),
    )
    existing = _fetch_evaluation_from_subrequest(
        db=db,
        subrequest=subrequest,
    )
    assert existing is not None
    assert (
        schemas.EvaluationParameters(**existing.parameters).task_type
        == enums.TaskType.CLASSIFICATION
    )

    # test `request.model_names` is empty
    with pytest.raises(RuntimeError):
        subrequest.model_names = []
        _fetch_evaluation_from_subrequest(db=db, subrequest=subrequest)

    # test `request.model_names` has multiple entries
    with pytest.raises(RuntimeError):
        subrequest.model_names = [finalized_model, "some_other_model"]
        _fetch_evaluation_from_subrequest(db=db, subrequest=subrequest)


def test_create_evaluation(
    db: Session,
    finalized_dataset: str,
    finalized_model: str,
):
    job_request_1 = schemas.EvaluationRequest(
        dataset_names=[finalized_dataset],
        model_names=[finalized_model],
        parameters=schemas.EvaluationParameters(
            task_type=enums.TaskType.CLASSIFICATION,
        ),
    )
    created = core.create_or_get_evaluations(db, job_request_1)
    assert len(created) == 1
    assert created[0].status == enums.EvaluationStatus.PENDING
    evaluation_id = created[0].id

    assert (
        core.get_evaluation_status(db, evaluation_id)
        == enums.EvaluationStatus.PENDING
    )

    # test duplication check
    existing = core.create_or_get_evaluations(db, job_request_1)
    assert len(existing) == 1
    assert existing[0].status == enums.EvaluationStatus.PENDING
    assert existing[0].id == evaluation_id

    assert (
        core.get_evaluation_status(db, evaluation_id)
        == enums.EvaluationStatus.PENDING
    )

    rows = db.query(models.Evaluation).all()
    assert len(rows) == 1
    assert rows[0].id == evaluation_id
    assert rows[0].dataset_names == [finalized_dataset]
    assert rows[0].model_name == finalized_model
    assert rows[0].filters == schemas.Filter().model_dump()
    assert (
        rows[0].parameters
        == schemas.EvaluationParameters(
            task_type=enums.TaskType.CLASSIFICATION,
        ).model_dump()
    )

    # test - bad request
    with pytest.raises(exceptions.EvaluationRequestError) as e:
        job_request_1 = schemas.EvaluationRequest(
            dataset_names=["some_other_dataset"],
            model_names=[finalized_model],
            parameters=schemas.EvaluationParameters(
                task_type=enums.TaskType.CLASSIFICATION,
            ),
        )
        core.create_or_get_evaluations(db, job_request_1)
    assert "DatasetDoesNotExist" in str(e)
    with pytest.raises(exceptions.EvaluationRequestError) as e:
        job_request_1 = schemas.EvaluationRequest(
            dataset_names=[finalized_dataset],
            model_names=["some_other_model"],
            parameters=schemas.EvaluationParameters(
                task_type=enums.TaskType.CLASSIFICATION,
            ),
        )
        core.create_or_get_evaluations(db, job_request_1)
    assert "ModelDoesNotExist" in str(e)


def test_fetch_evaluation_from_id(
    db: Session,
    finalized_dataset: str,
    finalized_model: str,
):
    # create evaluation 1
    job_request_1 = schemas.EvaluationRequest(
        dataset_names=[finalized_dataset],
        model_names=[finalized_model],
        parameters=schemas.EvaluationParameters(
            task_type=enums.TaskType.CLASSIFICATION,
        ),
    )
    created_1 = core.create_or_get_evaluations(db, job_request_1)
    assert len(created_1) == 1
    assert created_1[0].status == enums.EvaluationStatus.PENDING
    evaluation_id_1 = created_1[0].id

    # create evaluation 2
    job_request_2 = schemas.EvaluationRequest(
        dataset_names=[finalized_dataset],
        model_names=[finalized_model],
        parameters=schemas.EvaluationParameters(
            task_type=enums.TaskType.SEMANTIC_SEGMENTATION,
        ),
    )
    created_2 = core.create_or_get_evaluations(db, job_request_2)
    assert len(created_2) == 1
    assert created_2[0].status == enums.EvaluationStatus.PENDING
    evaluation_id_2 = created_2[0].id

    fetched_evaluation = core.fetch_evaluation_from_id(db, evaluation_id_1)
    assert fetched_evaluation.id == evaluation_id_1
    assert (
        fetched_evaluation.parameters["task_type"]
        == enums.TaskType.CLASSIFICATION
    )

    fetched_evaluation = core.fetch_evaluation_from_id(db, evaluation_id_2)
    assert fetched_evaluation.id == evaluation_id_2
    assert (
        fetched_evaluation.parameters["task_type"]
        == enums.TaskType.SEMANTIC_SEGMENTATION
    )
    assert isinstance(fetched_evaluation.created_at, datetime.datetime)


def test_get_evaluations(
    db: Session,
    finalized_dataset: str,
    finalized_model: str,
):
    # create evaluation 1
    job_request_1 = schemas.EvaluationRequest(
        dataset_names=[finalized_dataset],
        model_names=[finalized_model],
        parameters=schemas.EvaluationParameters(
            task_type=enums.TaskType.CLASSIFICATION,
        ),
    )
    created_1 = core.create_or_get_evaluations(db, job_request_1)
    assert len(created_1) == 1
    assert created_1[0].status == enums.EvaluationStatus.PENDING

    # create evaluation 2
    job_request_2 = schemas.EvaluationRequest(
        dataset_names=[finalized_dataset],
        model_names=[finalized_model],
        parameters=schemas.EvaluationParameters(
            task_type=enums.TaskType.SEMANTIC_SEGMENTATION,
        ),
    )
    created_2 = core.create_or_get_evaluations(db, job_request_2)
    assert len(created_2) == 1
    assert created_2[0].status == enums.EvaluationStatus.PENDING

    # test get by dataset
    evaluations_by_dataset = core.get_paginated_evaluations(
        db=db,
        dataset_names=[finalized_dataset],
    )
    assert len(evaluations_by_dataset) == 2

    # test get by model
    evaluations_by_model = core.get_paginated_evaluations(
        db=db,
        model_names=[finalized_model],
    )
    assert len(evaluations_by_model) == 2

    # test get by id
    evaluations_by_id = core.get_paginated_evaluations(
        db=db,
        evaluation_ids=[created_1[0].id, created_2[0].id],
    )
    assert len(evaluations_by_id) == 2

    # make sure stratifying works by dataset and evaluation id
    evaluations_by_dataset_and_eval_id, _ = core.get_paginated_evaluations(
        db=db,
        evaluation_ids=[created_1[0].id],
        dataset_names=[finalized_dataset],
    )
    assert len(evaluations_by_dataset_and_eval_id) == 1
    assert evaluations_by_dataset_and_eval_id[0].id == created_1[0].id

    # make sure stratifying works by model and evaluation id
    evaluations_by_model_and_eval_id, _ = core.get_paginated_evaluations(
        db=db,
        evaluation_ids=[created_2[0].id],
        model_names=[finalized_model],
    )
    assert len(evaluations_by_model_and_eval_id) == 1
    assert evaluations_by_model_and_eval_id[0].id == created_2[0].id

    # make sure stratifying works by dataset, model and evaluation id
    evaluations_by_dataset_model_eval_id, _ = core.get_paginated_evaluations(
        db=db,
        evaluation_ids=[created_2[0].id],
        dataset_names=[finalized_dataset],
        model_names=[finalized_model],
    )
    assert len(evaluations_by_dataset_model_eval_id) == 1
    assert evaluations_by_dataset_model_eval_id[0].id == created_2[0].id

    # make sure stratifying works by dataset and model
    evaluations_by_dataset_model_eval_id, _ = core.get_paginated_evaluations(
        db=db,
        dataset_names=[finalized_dataset],
        model_names=[finalized_model],
    )
    assert len(evaluations_by_dataset_model_eval_id) == 2

    # test pagination
    with pytest.raises(ValueError):
        # offset is greater than the number of items returned in query
        evaluations, headers = core.get_paginated_evaluations(
            db=db,
            dataset_names=[finalized_dataset],
            model_names=[finalized_model],
            offset=6,
            limit=1,
        )

    evaluations, headers = core.get_paginated_evaluations(
        db=db,
        dataset_names=[finalized_dataset],
        model_names=[finalized_model],
        offset=1,
        limit=1,
    )

    assert len(evaluations) == 1
    assert headers == {"content-range": "items 1-1/2"}

    # check that having too high of a limit param doesn't throw an error
    evaluations, headers = core.get_paginated_evaluations(
        db=db,
        dataset_names=[finalized_dataset],
        model_names=[finalized_model],
        offset=0,
        limit=6,
    )

    assert len(evaluations) == 2
    assert headers == {"content-range": "items 0-1/2"}

    # test that we can reconstitute the full set using paginated calls
    first, header = core.get_paginated_evaluations(db, offset=1, limit=1)
    assert len(first) == 1
    assert header == {"content-range": "items 1-1/2"}

    second, header = core.get_paginated_evaluations(db, offset=0, limit=1)
    assert len(second) == 1
    assert header == {"content-range": "items 0-0/2"}

    combined = first + second
    assert len(combined)

    # test metrics_to_sort_by when there aren't any metrics to sort by
    evaluations, headers = core.get_paginated_evaluations(
        db=db,
        dataset_names=[finalized_dataset],
        model_names=[finalized_model],
        offset=0,
        limit=6,
        metrics_to_sort_by={"IOU": "k1"},
    )

    assert len(evaluations) == 2
    assert headers == {"content-range": "items 0-1/2"}

    # test that we can reconstitute the full set using paginated calls
    first, header = core.get_paginated_evaluations(db, offset=1, limit=1)
    assert len(first) == 1
    assert header == {"content-range": "items 1-1/2"}

    second, header = core.get_paginated_evaluations(db, offset=0, limit=1)
    assert len(second) == 1
    assert header == {"content-range": "items 0-0/2"}

    combined = first + second
    assert len(combined)

    evaluations, headers = core.get_paginated_evaluations(
        db=db,
        dataset_names=[finalized_dataset],
        model_names=[finalized_model],
        offset=0,
        limit=6,
        metrics_to_sort_by={"IOU": {"key": "k1", "value": "v1"}},
    )

    assert len(evaluations) == 2
    assert headers == {"content-range": "items 0-1/2"}

    # test that we can reconstitute the full set using paginated calls
    first, header = core.get_paginated_evaluations(db, offset=1, limit=1)
    assert len(first) == 1
    assert header == {"content-range": "items 1-1/2"}

    second, header = core.get_paginated_evaluations(db, offset=0, limit=1)
    assert len(second) == 1
    assert header == {"content-range": "items 0-0/2"}

    combined = first + second
    assert len(combined)


def test_get_evaluation_requests_from_model(
    db: Session, finalized_dataset: str, finalized_model: str
):
    # create evaluation 1
    job_request_1 = schemas.EvaluationRequest(
        dataset_names=[finalized_dataset],
        model_names=[finalized_model],
        parameters=schemas.EvaluationParameters(
            task_type=enums.TaskType.CLASSIFICATION,
        ),
    )
    core.create_or_get_evaluations(db, job_request_1)

    # create evaluation 2
    job_request_2 = schemas.EvaluationRequest(
        dataset_names=[finalized_dataset],
        model_names=[finalized_model],
        parameters=schemas.EvaluationParameters(
            task_type=enums.TaskType.SEMANTIC_SEGMENTATION,
        ),
    )
    core.create_or_get_evaluations(db, job_request_2)

    eval_requests = core.get_evaluation_requests_from_model(
        db, finalized_model
    )

    assert len(eval_requests) == 2

    for eval_request in eval_requests:
        assert eval_request.model_name == finalized_model
        assert eval_request.dataset_names == [finalized_dataset]

    assert {
        eval_request.parameters.task_type for eval_request in eval_requests
    } == {enums.TaskType.CLASSIFICATION, enums.TaskType.SEMANTIC_SEGMENTATION}


def test_evaluation_status(
    db: Session,
    finalized_dataset: str,
    finalized_model: str,
):
    # create evaluation 1
    job_request_1 = schemas.EvaluationRequest(
        dataset_names=[finalized_dataset],
        model_names=[finalized_model],
        parameters=schemas.EvaluationParameters(
            task_type=enums.TaskType.CLASSIFICATION,
        ),
    )
    evaluations = core.create_or_get_evaluations(db, job_request_1)
    assert len(evaluations) == 1
    assert evaluations[0].status == enums.EvaluationStatus.PENDING
    evaluation_id = evaluations[0].id

    # check that evaluation is created with PENDING status.
    assert (
        core.get_evaluation_status(db, evaluation_id)
        == enums.EvaluationStatus.PENDING
    )

    # test
    with pytest.raises(exceptions.EvaluationStateError):
        core.set_evaluation_status(
            db, evaluation_id, enums.EvaluationStatus.DONE
        )
    with pytest.raises(exceptions.EvaluationStateError):
        core.set_evaluation_status(
            db, evaluation_id, enums.EvaluationStatus.DELETING
        )

    # set evaluation to running
    core.set_evaluation_status(
        db, evaluation_id, enums.EvaluationStatus.RUNNING
    )

    # test
    assert (
        core.get_evaluation_status(db, evaluation_id)
        == enums.EvaluationStatus.RUNNING
    )
    with pytest.raises(exceptions.EvaluationStateError):
        core.set_evaluation_status(
            db, evaluation_id, enums.EvaluationStatus.PENDING
        )
    with pytest.raises(exceptions.EvaluationStateError):
        core.set_evaluation_status(
            db, evaluation_id, enums.EvaluationStatus.DELETING
        )

    # set evaluation to failed
    core.set_evaluation_status(
        db, evaluation_id, enums.EvaluationStatus.FAILED
    )

    # test
    assert (
        core.get_evaluation_status(db, evaluation_id)
        == enums.EvaluationStatus.FAILED
    )
    with pytest.raises(exceptions.EvaluationStateError):
        core.set_evaluation_status(
            db, evaluation_id, enums.EvaluationStatus.PENDING
        )
    with pytest.raises(exceptions.EvaluationStateError):
        core.set_evaluation_status(
            db, evaluation_id, enums.EvaluationStatus.DONE
        )

    # set evaluation to running
    core.set_evaluation_status(
        db, evaluation_id, enums.EvaluationStatus.RUNNING
    )

    # set evaluation to done
    core.set_evaluation_status(db, evaluation_id, enums.EvaluationStatus.DONE)

    # test
    assert (
        core.get_evaluation_status(db, evaluation_id)
        == enums.EvaluationStatus.DONE
    )
    with pytest.raises(exceptions.EvaluationStateError):
        core.set_evaluation_status(
            db, evaluation_id, enums.EvaluationStatus.PENDING
        )
    with pytest.raises(exceptions.EvaluationStateError):
        core.set_evaluation_status(
            db, evaluation_id, enums.EvaluationStatus.RUNNING
        )
    with pytest.raises(exceptions.EvaluationStateError):
        core.set_evaluation_status(
            db, evaluation_id, enums.EvaluationStatus.FAILED
        )

    # set evaluation to deleting
    core.set_evaluation_status(
        db, evaluation_id, enums.EvaluationStatus.DELETING
    )

    # test an evaluation marked as DELETING is basically non-existent
    with pytest.raises(exceptions.EvaluationDoesNotExistError):
        core.get_evaluation_status(db, evaluation_id)

    with pytest.raises(exceptions.EvaluationDoesNotExistError):
        core.set_evaluation_status(
            db, evaluation_id, enums.EvaluationStatus.PENDING
        )


def test_count_active_evaluations(
    db: Session,
    finalized_dataset: str,
    finalized_model: str,
):
    # create evaluation 1
    job_request_1 = schemas.EvaluationRequest(
        dataset_names=[finalized_dataset],
        model_names=[finalized_model],
        parameters=schemas.EvaluationParameters(
            task_type=enums.TaskType.CLASSIFICATION,
        ),
    )
    created = core.create_or_get_evaluations(db, job_request_1)
    assert len(created) == 1
    evaluation_1 = created[0].id

    # create evaluation 2
    job_request_2 = schemas.EvaluationRequest(
        dataset_names=[finalized_dataset],
        model_names=[finalized_model],
        parameters=schemas.EvaluationParameters(
            task_type=enums.TaskType.SEMANTIC_SEGMENTATION,
        ),
    )
    created = core.create_or_get_evaluations(db, job_request_2)
    assert len(created) == 1
    evaluation_2 = created[0].id

    # keep evaluation 2 constant, run evaluation 1
    assert (
        core.count_active_evaluations(
            db=db,
            dataset_names=[finalized_dataset],
            model_names=[finalized_model],
        )
        == 2
    )

    core.set_evaluation_status(
        db, evaluation_1, enums.EvaluationStatus.RUNNING
    )

    assert (
        core.count_active_evaluations(
            db=db,
            dataset_names=[finalized_dataset],
            model_names=[finalized_model],
        )
        == 2
    )

    core.set_evaluation_status(db, evaluation_1, enums.EvaluationStatus.DONE)

    assert (
        core.count_active_evaluations(
            db=db,
            dataset_names=[finalized_dataset],
            model_names=[finalized_model],
        )
        == 1
    )

    # create evaluation 3
    job_request_3 = schemas.EvaluationRequest(
        dataset_names=[finalized_dataset],
        model_names=[finalized_model],
        parameters=schemas.EvaluationParameters(
            task_type=enums.TaskType.OBJECT_DETECTION,
        ),
    )
    evaluation_3 = core.create_or_get_evaluations(db, job_request_3)
    assert len(evaluation_3) == 1
    evaluation_3 = evaluation_3[0].id

    assert (
        core.count_active_evaluations(
            db=db,
            dataset_names=[finalized_dataset],
            model_names=[finalized_model],
        )
        == 2
    )

    # set both evaluations 2 & 3 to running

    core.set_evaluation_status(
        db, evaluation_2, enums.EvaluationStatus.RUNNING
    )
    core.set_evaluation_status(
        db, evaluation_3, enums.EvaluationStatus.RUNNING
    )

    # test a failed run and then a successful run on evaluation 2

    assert (
        core.count_active_evaluations(
            db=db,
            dataset_names=[finalized_dataset],
            model_names=[finalized_model],
        )
        == 2
    )

    core.set_evaluation_status(db, evaluation_2, enums.EvaluationStatus.FAILED)

    assert (
        core.count_active_evaluations(
            db=db,
            dataset_names=[finalized_dataset],
            model_names=[finalized_model],
        )
        == 1
    )

    core.set_evaluation_status(
        db, evaluation_2, enums.EvaluationStatus.RUNNING
    )

    assert (
        core.count_active_evaluations(
            db=db,
            dataset_names=[finalized_dataset],
            model_names=[finalized_model],
        )
        == 2
    )

    core.set_evaluation_status(db, evaluation_2, enums.EvaluationStatus.DONE)

    assert (
        core.count_active_evaluations(
            db=db,
            dataset_names=[finalized_dataset],
            model_names=[finalized_model],
        )
        == 1
    )

    core.set_evaluation_status(
        db, evaluation_2, enums.EvaluationStatus.DELETING
    )

    assert (
        core.count_active_evaluations(
            db=db,
            dataset_names=[finalized_dataset],
            model_names=[finalized_model],
        )
        == 1
    )

    # finish evaluation 3

    core.set_evaluation_status(db, evaluation_3, enums.EvaluationStatus.DONE)

    assert (
        core.count_active_evaluations(
            db=db,
            dataset_names=[finalized_dataset],
            model_names=[finalized_model],
        )
        == 0
    )


def test_delete_evaluations(
    db: Session,
    dataset_name: str,
    model_name: str,
    gt_clfs_create: list[schemas.GroundTruth],
    pred_clfs_create: list[schemas.Prediction],
):
    crud.create_dataset(
        db=db,
        dataset=schemas.Dataset(name=dataset_name),
    )
    for gt in gt_clfs_create:
        gt.dataset_name = dataset_name
        crud.create_groundtruths(db=db, groundtruths=[gt])
    crud.finalize(db=db, dataset_name=dataset_name)

    crud.create_model(db=db, model=schemas.Model(name=model_name))
    for pd in pred_clfs_create:
        pd.dataset_name = dataset_name
        pd.model_name = model_name
        crud.create_predictions(db=db, predictions=[pd])
    crud.finalize(db=db, model_name=model_name, dataset_name=dataset_name)

    job_request = schemas.EvaluationRequest(
        dataset_names=[dataset_name],
        model_names=[model_name],
        parameters=schemas.EvaluationParameters(
            task_type=enums.TaskType.CLASSIFICATION,
        ),
    )

    # create clf evaluation
    resp = crud.create_or_get_evaluations(
        db=db,
        job_request=job_request,
    )
    assert len(resp) == 1
    evaluation = db.query(models.Evaluation).one_or_none()
    assert evaluation

    for status in [
        enums.EvaluationStatus.PENDING,
        enums.EvaluationStatus.RUNNING,
    ]:

        # set status
        try:
            evaluation.status = status
            db.commit()
        except IntegrityError as e:
            db.rollback()
            raise e

        # check quantities
        assert db.scalar(func.count(models.Evaluation.id)) == 1
        assert db.scalar(func.count(models.Metric.id)) == 22
        assert db.scalar(func.count(models.ConfusionMatrix.id)) == 2

        # attempt to delete evaluation with PENDING status
        with pytest.raises(exceptions.EvaluationRunningError):
            core.delete_evaluations(db=db, evaluation_ids=[evaluation.id])

    # check quantities
    assert db.scalar(func.count(models.Evaluation.id)) == 1
    assert db.scalar(func.count(models.Metric.id)) == 22
    assert db.scalar(func.count(models.ConfusionMatrix.id)) == 2

    # set status to deleting
    try:
        evaluation.status = enums.EvaluationStatus.DELETING
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise e

    # attempt to delete evaluation with DELETING status
    # should do nothing as another worker is handling it.
    core.delete_evaluations(db=db, evaluation_ids=[evaluation.id])

    # check quantities
    assert db.scalar(func.count(models.Evaluation.id)) == 1
    assert db.scalar(func.count(models.Metric.id)) == 22
    assert db.scalar(func.count(models.ConfusionMatrix.id)) == 2

    # set status to done
    try:
        evaluation.status = enums.EvaluationStatus.DONE
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise e

    # attempt to delete evaluation with DONE status
    core.delete_evaluations(db=db, evaluation_ids=[evaluation.id])

    # check quantities
    assert db.scalar(func.count(models.Evaluation.id)) == 0
    assert db.scalar(func.count(models.Metric.id)) == 0
    assert db.scalar(func.count(models.ConfusionMatrix.id)) == 0

    # create clf evaluation (again)
    resp = crud.create_or_get_evaluations(
        db=db,
        job_request=job_request,
    )
    assert len(resp) == 1
    evaluation = db.query(models.Evaluation).one_or_none()
    assert evaluation

    # set status to failed
    try:
        evaluation.status = enums.EvaluationStatus.FAILED
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise e

    # check quantities
    assert db.scalar(func.count(models.Evaluation.id)) == 1
    assert db.scalar(func.count(models.Metric.id)) == 22
    assert db.scalar(func.count(models.ConfusionMatrix.id)) == 2

    # attempt to delete evaluation with DONE status
    core.delete_evaluations(db=db, evaluation_ids=[evaluation.id])

    # check quantities
    assert db.scalar(func.count(models.Evaluation.id)) == 0
    assert db.scalar(func.count(models.Metric.id)) == 0
    assert db.scalar(func.count(models.ConfusionMatrix.id)) == 0
