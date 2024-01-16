import pytest
from sqlalchemy.orm import Session

from velour_api import enums, exceptions, schemas
from velour_api.backend import core, models


@pytest.fixture
def created_dataset(db: Session, dataset_name: str) -> str:
    dataset = schemas.Dataset(name=dataset_name)
    core.create_dataset(db, dataset=dataset)
    return dataset_name


@pytest.fixture
def created_model(db: Session, model_name: str) -> str:
    model = schemas.Model(name=model_name)
    core.create_model(db, model=model)
    return model_name


def test_create_evaluation(
    db: Session,
    created_dataset: str,
    created_model: str,
):
    # create finalized dataset (automatically finalizes model as both are empty)
    core.set_dataset_status(db, created_dataset, enums.TableStatus.FINALIZED)

    job_request_1 = schemas.EvaluationRequest(
        model_filter=schemas.Filter(model_names=[created_model]),
        evaluation_filter=schemas.Filter(
            dataset_names=[created_dataset],
            task_types=[enums.TaskType.CLASSIFICATION],
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
    assert rows[0].model_filter == schemas.Filter(model_names=[created_model])
    assert rows[0].evaluation_filter == job_request_1.evaluation_filter.model_dump()
    assert rows[0].parameters == None


def test_fetch_evaluation_from_id(
    db: Session,
    created_dataset: str,
    created_model: str,
):
    # create finalized dataset (automatically finalizes model as both are empty)
    core.set_dataset_status(db, created_dataset, enums.TableStatus.FINALIZED)

    # create evaluation 1
    job_request_1 = schemas.EvaluationRequest(
        model_filter=schemas.Filter(model_names=[created_model]),
        evaluation_filter=schemas.Filter(
            dataset_names=[created_dataset],
            task_types=[enums.TaskType.CLASSIFICATION],
        ),
    )
    created_1, _ = core.create_or_get_evaluations(db, job_request_1)
    assert len(created_1) == 1
    evaluation_id_1 = created_1[0].id

    # create evaluation 2
    job_request_2 = schemas.EvaluationRequest(
        model_filter=schemas.Filter(model_names=[created_model]),
        evaluation_filter=schemas.Filter(
            dataset_names=[created_dataset],
            task_types=[enums.TaskType.SEGMENTATION],
        ),
    )
    created_2, _ = core.create_or_get_evaluations(db, job_request_2)
    assert len(created_2) == 1
    evaluation_id_2 = created_2[0].id

    fetched_evaluation = core.fetch_evaluation_from_id(db, evaluation_id_1)
    assert fetched_evaluation.id == evaluation_id_1
    assert fetched_evaluation.evaluation_filter["task_types"][0] == enums.TaskType.CLASSIFICATION

    fetched_evaluation = core.fetch_evaluation_from_id(db, evaluation_id_2)
    assert fetched_evaluation.id == evaluation_id_2
    assert fetched_evaluation.evaluation_filter["task_types"][0] == enums.TaskType.SEGMENTATION


def test_get_evaluation_id_from_job_request(
    db: Session,
    created_dataset: str,
    created_model: str,
):
    # create finalized dataset (automatically finalizes model as both are empty)
    core.set_dataset_status(db, created_dataset, enums.TableStatus.FINALIZED)

    # create evaluation 1
    job_request_1 = schemas.EvaluationRequest(
        model_filter=schemas.Filter(model_names=[created_model]),
        evaluation_filter=schemas.Filter(
            dataset_names=[created_dataset],
            task_types=[enums.TaskType.CLASSIFICATION],
        ),
    )
    created_1, _ = core.create_or_get_evaluations(db, job_request_1)
    assert len(created_1) == 1
    evaluation_1 = created_1[0].id

    # create evaluation 2
    job_request_2 = schemas.EvaluationRequest(
        model_filter=schemas.Filter(model_names=[created_model]),
        evaluation_filter=schemas.Filter(
            dataset_names=[created_dataset],
            task_types=[enums.TaskType.SEGMENTATION],
        ),
    )
    created_2, _ = core.create_or_get_evaluations(db, job_request_2)
    assert len(created_2) == 1
    evaluation_2 = created_2[0].id

    fetched_ids = core.get_evaluation_ids(db, job_request_1)
    assert len(fetched_ids) == 1
    assert fetched_ids[0] == evaluation_1

    fetched_ids = core.get_evaluation_ids(db, job_request_2)
    assert len(fetched_ids) == 1
    assert fetched_ids[0] == evaluation_2


# TODO 
# def test_get_evaluations(
#     db: Session,
#     created_dataset: str,
#     created_model: str,
# ):
#     # create finalized dataset (automatically finalizes model as both are empty)
#     core.set_dataset_status(db, created_dataset, enums.TableStatus.FINALIZED)

#     # create evaluation 1
#     job_request_1 = schemas.EvaluationRequest(
#         model_filter=schemas.Filter(model_names=[created_model]),
#         evaluation_filter=schemas.Filter(
#             dataset_names=[created_dataset],
#             task_types=[enums.TaskType.CLASSIFICATION],
#         ),
#     )
#     created_1, _ = core.create_or_get_evaluations(db, job_request_1)
#     assert len(created_1) == 1

#     # create evaluation 2
#     job_request_2 = schemas.EvaluationRequest(
#         model_filter=schemas.Filter(model_names=[created_model]),
#         evaluation_filter=schemas.Filter(
#             dataset_names=[created_dataset],
#             task_types=[enums.TaskType.SEGMENTATION],
#         ),
#     )
#     created_2, _ = core.create_or_get_evaluations(db, job_request_2)
#     assert len(created_2) == 1

#     get_request = schemas.EvaluationRequest(
#         model_filter=schemas.Filter(model_names=[created_model]),
#         evaluation_filter=schemas.Filter(
#             dataset_names=[created_dataset],
#             task_types=[
#                 enums.TaskType.CLASSIFICATION,
#                 enums.TaskType.SEGMENTATION,
#             ],
#         ),
#     )
#     created, existing = core.create_or_get_evaluations(
#         db=db,
#         job_request=get_request,
#     )
#     assert len(created) == 0
#     assert len(existing) == 2
#     task_types = {e.evaluation_filter.task_types[0] for e in existing if len(e.evaluation_filter.task_types) == 1}
#     assert task_types == {enums.TaskType.CLASSIFICATION, enums.TaskType.SEGMENTATION}


def test_evaluation_status(
    db: Session,
    created_dataset: str,
    created_model: str,
):
    # create finalized dataset (automatically finalizes model as both are empty)
    core.set_dataset_status(db, created_dataset, enums.TableStatus.FINALIZED)

    # create evaluation 1
    job_request_1 = schemas.EvaluationRequest(
        model_filter=schemas.Filter(model_names=[created_model]),
        evaluation_filter=schemas.Filter(
            dataset_names=[created_dataset],
            task_types=[enums.TaskType.CLASSIFICATION],
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


def test_check_for_active_evaluations(
    db: Session,
    created_dataset: str,
    created_model: str,
):
    # create finalized dataset (automatically finalizes model as both are empty)
    core.set_dataset_status(db, created_dataset, enums.TableStatus.FINALIZED)

    # create evaluation 1
    job_request_1 = schemas.EvaluationRequest(
        model_filter=schemas.Filter(model_names=[created_model]),
        evaluation_filter=schemas.Filter(
            dataset_names=[created_dataset],
            task_types=[enums.TaskType.CLASSIFICATION],
        )
    )
    created, _ = core.create_or_get_evaluations(db, job_request_1)
    assert len(created) == 1
    evaluation_1 = created[0].id

    # create evaluation 2
    job_request_2 = schemas.EvaluationRequest(
        model_filter=schemas.Filter(model_names=[created_model]),
        evaluation_filter=schemas.Filter(
            dataset_names=[created_dataset],
            task_types=[enums.TaskType.SEGMENTATION],
        )
    )
    created, _ = core.create_or_get_evaluations(db, job_request_2)
    assert len(created) == 1
    evaluation_2 = created[0].id

    # keep evaluation 2 constant, run evaluation 1

    from sqlalchemy import select, func
    print(db.scalar(select(func.count()).select_from(models.Evaluation)))

    assert (
        core.check_for_active_evaluations(db, created_dataset, created_model)
        == 2
    )

    core.set_evaluation_status(
        db, evaluation_1, enums.EvaluationStatus.RUNNING
    )

    assert (
        core.check_for_active_evaluations(db, created_dataset, created_model)
        == 2
    )

    core.set_evaluation_status(db, evaluation_1, enums.EvaluationStatus.DONE)

    assert (
        core.check_for_active_evaluations(db, created_dataset, created_model)
        == 1
    )

    # create evaluation 3
    job_request_3 = schemas.EvaluationRequest(
        model_filter=schemas.Filter(model_names=[created_model]),
        evaluation_filter=schemas.Filter(
            dataset_names=[created_dataset],
            task_types=[enums.TaskType.DETECTION],
        )
    )
    evaluation_3, _ = core.create_or_get_evaluations(db, job_request_3)
    evaluation_3 = evaluation_3[0].id

    assert (
        core.check_for_active_evaluations(db, created_dataset, created_model)
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
        core.check_for_active_evaluations(db, created_dataset, created_model)
        == 2
    )

    core.set_evaluation_status(db, evaluation_2, enums.EvaluationStatus.FAILED)

    assert (
        core.check_for_active_evaluations(db, created_dataset, created_model)
        == 1
    )

    core.set_evaluation_status(
        db, evaluation_2, enums.EvaluationStatus.RUNNING
    )

    assert (
        core.check_for_active_evaluations(db, created_dataset, created_model)
        == 2
    )

    core.set_evaluation_status(db, evaluation_2, enums.EvaluationStatus.DONE)

    assert (
        core.check_for_active_evaluations(db, created_dataset, created_model)
        == 1
    )

    core.set_evaluation_status(
        db, evaluation_2, enums.EvaluationStatus.DELETING
    )

    assert (
        core.check_for_active_evaluations(db, created_dataset, created_model)
        == 1
    )

    # finish evaluation 3

    core.set_evaluation_status(db, evaluation_3, enums.EvaluationStatus.DONE)

    assert (
        core.check_for_active_evaluations(db, created_dataset, created_model)
        == 0
    )
