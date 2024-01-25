import pytest
from sqlalchemy.orm import Session

from velour_api import enums, exceptions, schemas
from velour_api.backend import core, models
from velour_api.backend.core.evaluation import (
    _fetch_evaluation_from_subrequest,
    _verify_ready_to_evaluate,
)


@pytest.fixture
def created_dataset(db: Session, dataset_name: str) -> str:
    dataset = schemas.Dataset(name=dataset_name)
    core.create_dataset(db, dataset=dataset)
    core.create_groundtruth(
        db=db,
        groundtruth=schemas.GroundTruth(
            datum=schemas.Datum(uid="uid1", dataset_name=dataset_name),
            annotations=[
                schemas.Annotation(
                    task_type=enums.TaskType.CLASSIFICATION,
                    labels=[schemas.Label(key="k1", value="v1")],
                )
            ],
        ),
    )
    core.create_groundtruth(
        db=db,
        groundtruth=schemas.GroundTruth(
            datum=schemas.Datum(uid="uid2", dataset_name=dataset_name),
            annotations=[
                schemas.Annotation(
                    task_type=enums.TaskType.DETECTION,
                    labels=[schemas.Label(key="k1", value="v1")],
                )
            ],
        ),
    )
    core.create_groundtruth(
        db=db,
        groundtruth=schemas.GroundTruth(
            datum=schemas.Datum(uid="uid3", dataset_name=dataset_name),
            annotations=[
                schemas.Annotation(
                    task_type=enums.TaskType.SEGMENTATION,
                    labels=[schemas.Label(key="k1", value="v1")],
                )
            ],
        ),
    )
    return dataset_name


@pytest.fixture
def created_model(db: Session, model_name: str, created_dataset: str) -> str:
    model = schemas.Model(name=model_name)
    core.create_model(db, model=model)
    core.create_prediction(
        db=db,
        prediction=schemas.Prediction(
            model_name=model_name,
            datum=schemas.Datum(uid="uid1", dataset_name=created_dataset),
            annotations=[
                schemas.Annotation(
                    task_type=enums.TaskType.CLASSIFICATION,
                    labels=[schemas.Label(key="k1", value="v1", score=1.0)],
                )
            ],
        ),
    )
    core.create_prediction(
        db=db,
        prediction=schemas.Prediction(
            model_name=model_name,
            datum=schemas.Datum(uid="uid2", dataset_name=created_dataset),
            annotations=[
                schemas.Annotation(
                    task_type=enums.TaskType.DETECTION,
                    labels=[schemas.Label(key="k1", value="v1", score=1.0)],
                )
            ],
        ),
    )
    core.create_prediction(
        db=db,
        prediction=schemas.Prediction(
            model_name=model_name,
            datum=schemas.Datum(uid="uid3", dataset_name=created_dataset),
            annotations=[
                schemas.Annotation(
                    task_type=enums.TaskType.SEGMENTATION,
                    labels=[schemas.Label(key="k1", value="v1")],
                )
            ],
        ),
    )
    core.set_dataset_status(db, created_dataset, enums.TableStatus.FINALIZED)
    return model_name


def test__verify_ready_to_evaluate(
    db: Session,
    dataset_name: str,
    model_name: str,
):
    # test empty dataset list
    with pytest.raises(RuntimeError) as e:
        _verify_ready_to_evaluate(db=db, dataset_list=[], model_list=[])
    assert "empty list of datasets" in str(e)

    dataset = core.create_dataset(
        db, dataset=schemas.Dataset(name=dataset_name)
    )

    # test empty model list
    with pytest.raises(RuntimeError) as e:
        _verify_ready_to_evaluate(db=db, dataset_list=[dataset], model_list=[])
    assert "empty list of models" in str(e)

    model = core.create_model(db=db, model=schemas.Model(name=model_name))

    # test dataset in state `enums.TableStatus.CREATING`
    with pytest.raises(exceptions.DatasetNotFinalizedError):
        _verify_ready_to_evaluate(
            db=db, dataset_list=[dataset], model_list=[model]
        )

    core.create_groundtruth(
        db=db,
        groundtruth=schemas.GroundTruth(
            datum=schemas.Datum(uid="uid1", dataset_name=dataset_name),
            annotations=[
                schemas.Annotation(
                    task_type=enums.TaskType.CLASSIFICATION,
                    labels=[schemas.Label(key="k1", value="v1")],
                )
            ],
        ),
    )
    core.set_dataset_status(db, dataset_name, enums.TableStatus.FINALIZED)

    # test model in state `enums.TableStatus.CREATING`
    with pytest.raises(exceptions.ModelNotFinalizedError):
        _verify_ready_to_evaluate(
            db=db, dataset_list=[dataset], model_list=[model]
        )

    # create a prediction
    # automatically finalizes over dataset
    core.create_prediction(
        db=db,
        prediction=schemas.Prediction(
            model_name=model_name,
            datum=schemas.Datum(uid="uid1", dataset_name=dataset_name),
            annotations=[
                schemas.Annotation(
                    task_type=enums.TaskType.CLASSIFICATION,
                    labels=[schemas.Label(key="k1", value="v1", score=1.0)],
                )
            ],
        ),
    )

    # both dataset and model should be in valid finalized states
    _verify_ready_to_evaluate(
        db=db, dataset_list=[dataset], model_list=[model]
    )

    second_model = core.create_model(
        db=db, model=schemas.Model(name="second_model")
    )
    core.create_prediction(
        db=db,
        prediction=schemas.Prediction(
            model_name="second_model",
            datum=schemas.Datum(uid="uid1", dataset_name=dataset_name),
            annotations=[
                schemas.Annotation(
                    task_type=enums.TaskType.CLASSIFICATION,
                    labels=[schemas.Label(key="k1", value="v1", score=1.0)],
                )
            ],
        ),
    )

    _verify_ready_to_evaluate(
        db=db,
        dataset_list=[dataset],
        model_list=[model, second_model],
    )

    core.set_model_status(
        db=db,
        dataset_name=dataset_name,
        model_name="second_model",
        status=enums.TableStatus.DELETING,
    )

    # test model in deleting state
    with pytest.raises(exceptions.ModelDoesNotExistError) as e:
        _verify_ready_to_evaluate(
            db=db,
            dataset_list=[dataset],
            model_list=[model, second_model],
        )
    assert "second_model" in str(e)

    core.delete_model(db=db, name="second_model")
    core.set_dataset_status(
        db=db, name=dataset_name, status=enums.TableStatus.DELETING
    )

    # test dataset in deleting state
    with pytest.raises(exceptions.DatasetDoesNotExistError) as e:
        _verify_ready_to_evaluate(
            db=db,
            dataset_list=[dataset],
            model_list=[model, second_model],
        )
    assert dataset_name in str(e)

    # test invalid dataset status
    with pytest.raises(ValueError) as e:
        dataset.status = "arbitrary_invalid_str"
        _verify_ready_to_evaluate(
            db=db,
            dataset_list=[dataset],
            model_list=[model, second_model],
        )
    assert "arbitrary_invalid_str" in str(e)

    # IMPORTANT: reset the status to deleting or teardown fails.
    dataset.status = enums.TableStatus.DELETING.value


def test__fetch_evaluation_from_subrequest(
    db: Session,
    created_dataset: str,
    created_model: str,
):
    # create evaluation 1
    job_request_1 = schemas.EvaluationRequest(
        model_names=[created_model],
        datum_filter=schemas.Filter(dataset_names=[created_dataset]),
        parameters=schemas.EvaluationParameters(
            task_type=enums.TaskType.CLASSIFICATION,
        ),
    )
    created_1, _ = core.create_or_get_evaluations(db, job_request_1)
    assert len(created_1) == 1

    # create evaluation 2
    job_request_2 = schemas.EvaluationRequest(
        model_names=[created_model],
        datum_filter=schemas.Filter(dataset_names=[created_dataset]),
        parameters=schemas.EvaluationParameters(
            task_type=enums.TaskType.SEGMENTATION,
        ),
    )
    created_2, _ = core.create_or_get_evaluations(db, job_request_2)
    assert len(created_2) == 1

    # test fetching a subrequest
    subrequest = schemas.EvaluationRequest(
        model_names=[created_model],
        datum_filter=schemas.Filter(
            dataset_names=[created_dataset],
        ),
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
        subrequest.model_names = [created_model, "some_other_model"]
        _fetch_evaluation_from_subrequest(db=db, subrequest=subrequest)


def test_create_evaluation(
    db: Session,
    created_dataset: str,
    created_model: str,
):
    job_request_1 = schemas.EvaluationRequest(
        model_names=[created_model],
        datum_filter=schemas.Filter(dataset_names=[created_dataset]),
        parameters=schemas.EvaluationParameters(
            task_type=enums.TaskType.CLASSIFICATION,
        ),
    )
    created, existing = core.create_or_get_evaluations(db, job_request_1)
    assert len(existing) == 0
    assert len(created) == 1
    evaluation_id = created[0].id

    assert (
        core.get_evaluation_status(db, evaluation_id)
        == enums.EvaluationStatus.PENDING
    )

    # test duplication check
    created, existing = core.create_or_get_evaluations(db, job_request_1)
    assert len(created) == 0
    assert len(existing) == 1
    assert existing[0].id == evaluation_id

    assert (
        core.get_evaluation_status(db, evaluation_id)
        == enums.EvaluationStatus.PENDING
    )

    rows = db.query(models.Evaluation).all()
    assert len(rows) == 1
    assert rows[0].id == evaluation_id
    assert rows[0].model_name == created_model
    assert (
        rows[0].datum_filter
        == schemas.Filter(
            dataset_names=[created_dataset],
        ).model_dump()
    )
    assert (
        rows[0].parameters
        == schemas.EvaluationParameters(
            task_type=enums.TaskType.CLASSIFICATION
        ).model_dump()
    )

    # test - bad request
    with pytest.raises(exceptions.EvaluationRequestError) as e:
        job_request_1 = schemas.EvaluationRequest(
            model_names=[created_model],
            datum_filter=schemas.Filter(dataset_names=["some_other_dataset"]),
            parameters=schemas.EvaluationParameters(
                task_type=enums.TaskType.CLASSIFICATION,
            ),
        )
        core.create_or_get_evaluations(db, job_request_1)
    assert "No datasets" in str(e)
    with pytest.raises(exceptions.EvaluationRequestError) as e:
        job_request_1 = schemas.EvaluationRequest(
            model_names=["some_other_model"],
            datum_filter=schemas.Filter(dataset_names=[created_dataset]),
            parameters=schemas.EvaluationParameters(
                task_type=enums.TaskType.CLASSIFICATION,
            ),
        )
        core.create_or_get_evaluations(db, job_request_1)
    assert "No models" in str(e)


def test_fetch_evaluation_from_id(
    db: Session,
    created_dataset: str,
    created_model: str,
):
    # create evaluation 1
    job_request_1 = schemas.EvaluationRequest(
        model_names=[created_model],
        datum_filter=schemas.Filter(dataset_names=[created_dataset]),
        parameters=schemas.EvaluationParameters(
            task_type=enums.TaskType.CLASSIFICATION,
        ),
    )
    created_1, _ = core.create_or_get_evaluations(db, job_request_1)
    assert len(created_1) == 1
    evaluation_id_1 = created_1[0].id

    # create evaluation 2
    job_request_2 = schemas.EvaluationRequest(
        model_names=[created_model],
        datum_filter=schemas.Filter(dataset_names=[created_dataset]),
        parameters=schemas.EvaluationParameters(
            task_type=enums.TaskType.SEGMENTATION,
        ),
    )
    created_2, _ = core.create_or_get_evaluations(db, job_request_2)
    assert len(created_2) == 1
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
        == enums.TaskType.SEGMENTATION
    )


def test_get_evaluations(
    db: Session,
    created_dataset: str,
    created_model: str,
):
    # create evaluation 1
    job_request_1 = schemas.EvaluationRequest(
        model_names=[created_model],
        datum_filter=schemas.Filter(dataset_names=[created_dataset]),
        parameters=schemas.EvaluationParameters(
            task_type=enums.TaskType.CLASSIFICATION,
        ),
    )
    created_1, _ = core.create_or_get_evaluations(db, job_request_1)
    assert len(created_1) == 1

    # create evaluation 2
    job_request_2 = schemas.EvaluationRequest(
        model_names=[created_model],
        datum_filter=schemas.Filter(dataset_names=[created_dataset]),
        parameters=schemas.EvaluationParameters(
            task_type=enums.TaskType.SEGMENTATION,
        ),
    )
    created_2, _ = core.create_or_get_evaluations(db, job_request_2)
    assert len(created_2) == 1

    # test get by dataset
    evaluations_by_dataset = core.get_evaluations(
        db=db,
        dataset_names=[created_dataset],
    )
    assert len(evaluations_by_dataset) == 2

    # test get by model
    evaluations_by_model = core.get_evaluations(
        db=db,
        model_names=[created_model],
    )
    assert len(evaluations_by_model) == 2

    # test get by id
    evaluations_by_id = core.get_evaluations(
        db=db,
        evaluation_ids=[created_1[0].id, created_2[0].id],
    )
    assert len(evaluations_by_id) == 2

    # make sure stratifying works by dataset and evaluation id
    evaluations_by_dataset_and_eval_id = core.get_evaluations(
        db=db,
        evaluation_ids=[created_1[0].id],
        dataset_names=[created_dataset],
    )
    assert len(evaluations_by_dataset_and_eval_id) == 1
    assert evaluations_by_dataset_and_eval_id[0].id == created_1[0].id

    # make sure stratifying works by model and evaluation id
    evaluations_by_model_and_eval_id = core.get_evaluations(
        db=db,
        evaluation_ids=[created_2[0].id],
        model_names=[created_model],
    )
    assert len(evaluations_by_model_and_eval_id) == 1
    assert evaluations_by_model_and_eval_id[0].id == created_2[0].id

    # make sure stratifying works by dataset, model and evaluation id
    evaluations_by_dataset_model_eval_id = core.get_evaluations(
        db=db,
        evaluation_ids=[created_2[0].id],
        dataset_names=[created_dataset],
        model_names=[created_model],
    )
    assert len(evaluations_by_dataset_model_eval_id) == 1
    assert evaluations_by_dataset_model_eval_id[0].id == created_2[0].id

    # make sure stratifying works by dataset and model
    evaluations_by_dataset_model_eval_id = core.get_evaluations(
        db=db,
        dataset_names=[created_dataset],
        model_names=[created_model],
    )
    assert len(evaluations_by_dataset_model_eval_id) == 2


def test_get_evaluation_requests_from_model(
    db: Session, created_dataset: str, created_model: str
):
    # create evaluation 1
    job_request_1 = schemas.EvaluationRequest(
        model_names=[created_model],
        datum_filter=schemas.Filter(dataset_names=[created_dataset]),
        parameters=schemas.EvaluationParameters(
            task_type=enums.TaskType.CLASSIFICATION,
        ),
    )
    core.create_or_get_evaluations(db, job_request_1)

    # create evaluation 2
    job_request_2 = schemas.EvaluationRequest(
        model_names=[created_model],
        datum_filter=schemas.Filter(dataset_names=[created_dataset]),
        parameters=schemas.EvaluationParameters(
            task_type=enums.TaskType.SEGMENTATION,
        ),
    )
    core.create_or_get_evaluations(db, job_request_2)

    eval_requests = core.get_evaluation_requests_from_model(db, created_model)

    assert len(eval_requests) == 2

    for eval_request in eval_requests:
        assert eval_request.model_name == created_model
        assert eval_request.datum_filter.dataset_names == [created_dataset]

    assert {
        eval_request.parameters.task_type for eval_request in eval_requests
    } == {enums.TaskType.CLASSIFICATION, enums.TaskType.SEGMENTATION}

    assert {eval_request.id for eval_request in eval_requests} == {1, 2}


def test_evaluation_status(
    db: Session,
    created_dataset: str,
    created_model: str,
):
    # create evaluation 1
    job_request_1 = schemas.EvaluationRequest(
        model_names=[created_model],
        datum_filter=schemas.Filter(dataset_names=[created_dataset]),
        parameters=schemas.EvaluationParameters(
            task_type=enums.TaskType.CLASSIFICATION,
        ),
    )
    created_1, existing = core.create_or_get_evaluations(db, job_request_1)
    assert len(existing) == 0
    assert len(created_1) == 1
    evaluation_id = created_1[0].id

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

    # test
    assert (
        core.get_evaluation_status(db, evaluation_id)
        == enums.EvaluationStatus.DELETING
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
            db, evaluation_id, enums.EvaluationStatus.DONE
        )
    with pytest.raises(exceptions.EvaluationStateError):
        core.set_evaluation_status(
            db, evaluation_id, enums.EvaluationStatus.FAILED
        )


def test_count_active_evaluations(
    db: Session,
    created_dataset: str,
    created_model: str,
):
    # create evaluation 1
    job_request_1 = schemas.EvaluationRequest(
        model_names=[created_model],
        datum_filter=schemas.Filter(dataset_names=[created_dataset]),
        parameters=schemas.EvaluationParameters(
            task_type=enums.TaskType.CLASSIFICATION,
        ),
    )
    created, _ = core.create_or_get_evaluations(db, job_request_1)
    assert len(created) == 1
    evaluation_1 = created[0].id

    # create evaluation 2
    job_request_2 = schemas.EvaluationRequest(
        model_names=[created_model],
        datum_filter=schemas.Filter(dataset_names=[created_dataset]),
        parameters=schemas.EvaluationParameters(
            task_type=enums.TaskType.SEGMENTATION,
        ),
    )
    created, _ = core.create_or_get_evaluations(db, job_request_2)
    assert len(created) == 1
    evaluation_2 = created[0].id

    # keep evaluation 2 constant, run evaluation 1
    assert (
        core.count_active_evaluations(
            db=db,
            dataset_names=[created_dataset],
            model_names=[created_model],
        )
        == 2
    )

    core.set_evaluation_status(
        db, evaluation_1, enums.EvaluationStatus.RUNNING
    )

    assert (
        core.count_active_evaluations(
            db=db,
            dataset_names=[created_dataset],
            model_names=[created_model],
        )
        == 2
    )

    core.set_evaluation_status(db, evaluation_1, enums.EvaluationStatus.DONE)

    assert (
        core.count_active_evaluations(
            db=db,
            dataset_names=[created_dataset],
            model_names=[created_model],
        )
        == 1
    )

    # create evaluation 3
    job_request_3 = schemas.EvaluationRequest(
        model_names=[created_model],
        datum_filter=schemas.Filter(dataset_names=[created_dataset]),
        parameters=schemas.EvaluationParameters(
            task_type=enums.TaskType.DETECTION,
        ),
    )
    evaluation_3, _ = core.create_or_get_evaluations(db, job_request_3)
    evaluation_3 = evaluation_3[0].id

    assert (
        core.count_active_evaluations(
            db=db,
            dataset_names=[created_dataset],
            model_names=[created_model],
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
            dataset_names=[created_dataset],
            model_names=[created_model],
        )
        == 2
    )

    core.set_evaluation_status(db, evaluation_2, enums.EvaluationStatus.FAILED)

    assert (
        core.count_active_evaluations(
            db=db,
            dataset_names=[created_dataset],
            model_names=[created_model],
        )
        == 1
    )

    core.set_evaluation_status(
        db, evaluation_2, enums.EvaluationStatus.RUNNING
    )

    assert (
        core.count_active_evaluations(
            db=db,
            dataset_names=[created_dataset],
            model_names=[created_model],
        )
        == 2
    )

    core.set_evaluation_status(db, evaluation_2, enums.EvaluationStatus.DONE)

    assert (
        core.count_active_evaluations(
            db=db,
            dataset_names=[created_dataset],
            model_names=[created_model],
        )
        == 1
    )

    core.set_evaluation_status(
        db, evaluation_2, enums.EvaluationStatus.DELETING
    )

    assert (
        core.count_active_evaluations(
            db=db,
            dataset_names=[created_dataset],
            model_names=[created_model],
        )
        == 1
    )

    # finish evaluation 3

    core.set_evaluation_status(db, evaluation_3, enums.EvaluationStatus.DONE)

    assert (
        core.count_active_evaluations(
            db=db,
            dataset_names=[created_dataset],
            model_names=[created_model],
        )
        == 0
    )
