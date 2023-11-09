""" These tests require a redis instance either running unauthenticated
at localhost:6379 or with the following enviornment variables set:
REDIS_HOST, REDIS_PORT, REDIS_PASSWORD, REDIS_USERNAME, REDIS_DB
"""
from unittest.mock import MagicMock  # , patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from velour_api import crud, enums, exceptions, schemas
from velour_api.backend import database, jobs

# from velour_api.enums import JobStatus
# from velour_api.exceptions import JobDoesNotExistError
# from velour_api.schemas import Job

dset_name = "test_dataset"
model_name = "test_model"


@pytest.fixture
def client() -> TestClient:
    database.create_db = MagicMock()
    database.make_session = MagicMock()
    from velour_api import main

    main.get_db = MagicMock()

    return TestClient(main.app)


@pytest.fixture(autouse=True)
def setup_and_teardown():
    """The setup checks that the redis db is empty and
    the teardown flushes it
    """
    jobs.connect_to_redis()
    if len(jobs.r.keys()) != 0:
        raise RuntimeError("redis database is not-empty")
    yield
    jobs.r.flushdb()


@pytest.fixture
def gt_clfs_create(
    img1: schemas.Datum,
    img2: schemas.Datum,
) -> list[schemas.GroundTruth]:
    return [
        schemas.GroundTruth(
            datum=img1,
            annotations=[
                schemas.Annotation(
                    task_type=enums.TaskType.CLASSIFICATION,
                    labels=[
                        schemas.Label(key="k1", value="v1"),
                        schemas.Label(key="k2", value="v2"),
                    ],
                ),
            ],
        ),
        schemas.GroundTruth(
            datum=img2,
            annotations=[
                schemas.Annotation(
                    task_type=enums.TaskType.CLASSIFICATION,
                    labels=[schemas.Label(key="k2", value="v3")],
                ),
            ],
        ),
    ]


@pytest.fixture
def pred_clfs_create(
    img1: schemas.Datum, img2: schemas.Datum
) -> list[schemas.Prediction]:
    return [
        schemas.Prediction(
            model=model_name,
            datum=img1,
            annotations=[
                schemas.Annotation(
                    task_type=enums.TaskType.CLASSIFICATION,
                    labels=[
                        schemas.Label(key="k1", value="v1", score=0.2),
                        schemas.Label(key="k1", value="v2", score=0.8),
                        schemas.Label(key="k4", value="v4", score=1.0),
                    ],
                ),
            ],
        ),
        schemas.Prediction(
            model=model_name,
            datum=img2,
            annotations=[
                schemas.Annotation(
                    task_type=enums.TaskType.CLASSIFICATION,
                    labels=[
                        schemas.Label(key="k2", value="v2", score=1.0),
                        schemas.Label(key="k3", value="v3", score=0.87),
                        schemas.Label(key="k3", value="v0", score=0.13),
                    ],
                ),
            ],
        ),
    ]


def test_job_status():
    dataset_name = "md"
    model_name = "ds"
    stateflow = schemas.Stateflow(
        datasets={
            dataset_name: schemas.stateflow.DatasetState(
                status=enums.State.READY,
                models={
                    model_name: schemas.stateflow.InferenceState(
                        status=enums.State.READY, jobs={}
                    )
                },
            )
        }
    )

    # check that job id: 0 is non-existent
    assert stateflow.get_job_status(0) is None

    # test invalid transitions from `None`
    with pytest.raises(exceptions.JobDoesNotExistError) as e:
        stateflow.set_job_status(
            dataset_name, model_name, 0, enums.JobStatus.PROCESSING
        )
    assert "does not exist" in str(e)
    with pytest.raises(exceptions.JobDoesNotExistError) as e:
        stateflow.set_job_status(
            dataset_name, model_name, 0, enums.JobStatus.DONE
        )
    assert "does not exist" in str(e)
    with pytest.raises(exceptions.JobDoesNotExistError) as e:
        stateflow.set_job_status(
            dataset_name, model_name, 0, enums.JobStatus.FAILED
        )
    assert "does not exist" in str(e)

    # check that nothing affected the state
    assert stateflow.get_job_status(0) is None

    """test valid transition"""
    stateflow.set_job_status(
        dataset_name, model_name, 0, enums.JobStatus.PENDING
    )

    # test invalid transitions from `PENDING`
    with pytest.raises(exceptions.JobStateError) as e:
        stateflow.set_job_status(
            dataset_name, model_name, 0, enums.JobStatus.DONE
        )
    assert "`JobStatus.PENDING` to `JobStatus.DONE`" in str(e)
    with pytest.raises(exceptions.JobStateError) as e:
        stateflow.set_job_status(
            dataset_name, model_name, 0, enums.JobStatus.FAILED
        )
    assert "`JobStatus.PENDING` to `JobStatus.FAILED`" in str(e)

    # test removing PENDING job
    with pytest.raises(exceptions.JobStateError) as e:
        stateflow.remove_job(dataset_name, model_name, 0)
    assert "cannot remove an actively running job." in str(e)

    """test valid transition"""
    stateflow.set_job_status(
        dataset_name, model_name, 0, enums.JobStatus.PROCESSING
    )

    # test removing PROCESSING job
    with pytest.raises(exceptions.JobStateError) as e:
        stateflow.remove_job(dataset_name, model_name, 0)
    assert "cannot remove an actively running job." in str(e)

    # test invalid transitions from `PROCESSING`
    with pytest.raises(exceptions.JobStateError) as e:
        stateflow.set_job_status(
            dataset_name, model_name, 0, enums.JobStatus.PENDING
        )
    assert "`JobStatus.PROCESSING` to `JobStatus.PENDING`" in str(e)

    """test valid transition"""
    stateflow.set_job_status(
        dataset_name, model_name, 0, enums.JobStatus.FAILED
    )

    # test invalid transitions from `DONE`
    with pytest.raises(exceptions.JobStateError) as e:
        stateflow.set_job_status(
            dataset_name, model_name, 0, enums.JobStatus.PROCESSING
        )
    assert "`JobStatus.FAILED` to `JobStatus.PROCESSING`" in str(e)
    with pytest.raises(exceptions.JobStateError) as e:
        stateflow.set_job_status(
            dataset_name, model_name, 0, enums.JobStatus.DONE
        )
    assert "`JobStatus.FAILED` to `JobStatus.DONE`" in str(e)

    """test valid transition"""
    stateflow.set_job_status(
        dataset_name, model_name, 0, enums.JobStatus.PENDING
    )
    stateflow.set_job_status(
        dataset_name, model_name, 0, enums.JobStatus.PROCESSING
    )
    stateflow.set_job_status(dataset_name, model_name, 0, enums.JobStatus.DONE)

    # test invalid transitions from `DONE`
    with pytest.raises(exceptions.JobStateError) as e:
        stateflow.set_job_status(
            dataset_name, model_name, 0, enums.JobStatus.PENDING
        )
    assert "`JobStatus.DONE` to `JobStatus.PENDING`" in str(e)
    with pytest.raises(exceptions.JobStateError) as e:
        stateflow.set_job_status(
            dataset_name, model_name, 0, enums.JobStatus.PROCESSING
        )
    assert "`JobStatus.DONE` to `JobStatus.PROCESSING`" in str(e)
    with pytest.raises(exceptions.JobStateError) as e:
        stateflow.set_job_status(
            dataset_name, model_name, 0, enums.JobStatus.FAILED
        )
    assert "`JobStatus.DONE` to `JobStatus.FAILED`" in str(e)

    """test job removal"""
    stateflow.remove_job(dataset_name, model_name, 0)

    """confirm removal"""
    assert stateflow.get_job_status(0) is None


def test_stateflow_dataset(db: Session):
    # should have no record of dataset
    with pytest.raises(exceptions.DatasetDoesNotExistError) as e:
        crud.get_backend_state(dataset_name=dset_name)
    assert "does not exist" in str(e)

    # create dataset
    crud.create_dataset(
        db=db,
        dataset=schemas.Dataset(
            name=dset_name,
        ),
    )

    # `create_dataset` does not affect the stateflow
    assert crud.get_backend_state(dataset_name=dset_name) == enums.State.NONE

    # create a groundtruth
    crud.create_groundtruth(
        db=db,
        groundtruth=schemas.GroundTruth(
            datum=schemas.Datum(
                dataset=dset_name,
                uid="uid1",
            ),
            annotations=[
                schemas.Annotation(
                    task_type=enums.TaskType.CLASSIFICATION,
                    labels=[schemas.Label(key="k", value="v")],
                )
            ],
        ),
    )

    # `create_groundtruth` transitions dataset state into CREATE
    assert crud.get_backend_state(dataset_name=dset_name) == enums.State.CREATE

    # finalize dataset
    crud.finalize(db=db, dataset_name=dset_name)

    # `finalize` transitions dataset state into READY
    assert crud.get_backend_state(dataset_name=dset_name) == enums.State.READY

    # delete dataset
    crud.delete(db=db, dataset_name=dset_name)

    # after delete operation completes the record is removed
    with pytest.raises(exceptions.DatasetDoesNotExistError) as e:
        crud.get_backend_state(dataset_name=dset_name)
    assert "does not exist" in str(e)


def test_stateflow_model(db: Session):
    # create dataset
    crud.create_dataset(
        db=db,
        dataset=schemas.Dataset(
            name=dset_name,
        ),
    )
    crud.create_groundtruth(
        db=db,
        groundtruth=schemas.GroundTruth(
            datum=schemas.Datum(
                dataset=dset_name,
                uid="uid1",
            ),
            annotations=[
                schemas.Annotation(
                    task_type=enums.TaskType.CLASSIFICATION,
                    labels=[schemas.Label(key="k", value="v")],
                )
            ],
        ),
    )
    crud.finalize(db=db, dataset_name=dset_name)

    # check that no record exists for model
    with pytest.raises(exceptions.ModelInferencesDoNotExist) as e:
        crud.get_backend_state(dataset_name=dset_name, model_name=model_name)
    assert "do not exist" in str(e)

    # create model
    crud.create_model(
        db=db,
        model=schemas.Model(
            name=model_name,
        ),
    )

    # check that no record exists for model as no predictions have been added
    with pytest.raises(exceptions.ModelInferencesDoNotExist) as e:
        crud.get_backend_state(dataset_name=dset_name, model_name=model_name)
    assert "do not exist" in str(e)

    # create predictions
    crud.create_prediction(
        db=db,
        prediction=schemas.Prediction(
            model=model_name,
            datum=schemas.Datum(
                dataset=dset_name,
                uid="uid1",
            ),
            annotations=[
                schemas.Annotation(
                    task_type=enums.TaskType.CLASSIFICATION,
                    labels=[
                        schemas.Label(key="k", value="v", score=0.9),
                        schemas.Label(key="k", value="w", score=0.1),
                    ],
                )
            ],
        ),
    )

    # `create_prediction` transitions model state to CREATE
    assert (
        crud.get_backend_state(dataset_name=dset_name, model_name=model_name)
        == enums.State.CREATE
    )

    # check that evaluation fails before finalization
    with pytest.raises(exceptions.ModelNotFinalizedError) as e:
        crud.create_detection_evaluation(
            db=db,
            job_request=schemas.EvaluationJob(
                model=model_name,
                dataset=dset_name,
                settings=schemas.EvaluationSettings(
                    parameters=schemas.DetectionParameters(
                        iou_thresholds_to_compute=[0.2, 0.6],
                        iou_thresholds_to_keep=[0.2],
                    ),
                    filters=schemas.Filter(
                        annotation_types=[enums.AnnotationType.BOX],
                        label_keys=["class"],
                    ),
                ),
            ),
        )
    assert "has not been finalized" in str(e)

    # finalize model over dataset
    crud.finalize(db=db, dataset_name=dset_name, model_name=model_name)

    # `finalize` transitions dataset state into READY
    assert (
        crud.get_backend_state(dataset_name=dset_name, model_name=model_name)
        == enums.State.READY
    )

    # delete dataset
    crud.delete(db=db, dataset_name=dset_name, model_name=model_name)

    # after delete operation completes the record is removed
    with pytest.raises(exceptions.ModelInferencesDoNotExist) as e:
        crud.get_backend_state(dataset_name=dset_name, model_name=model_name)
    assert "do not exist" in str(e)


def test_stateflow_detection_evaluation(
    db: Session, groundtruths, predictions
):
    job_request = schemas.EvaluationJob(
        model=model_name,
        dataset=dset_name,
        settings=schemas.EvaluationSettings(
            parameters=schemas.DetectionParameters(
                iou_thresholds_to_compute=[0.2, 0.6],
                iou_thresholds_to_keep=[0.2],
            ),
            filters=schemas.Filter(
                annotation_types=[enums.AnnotationType.BOX],
                label_keys=["class"],
            ),
        ),
    )

    # check ready
    assert crud.get_backend_state(dataset_name=dset_name) == enums.State.READY
    assert (
        crud.get_backend_state(dataset_name=dset_name, model_name=model_name)
        == enums.State.READY
    )

    # create evaluation (return AP Response)
    resp = crud.create_detection_evaluation(db=db, job_request=job_request)

    # check in evalutation
    assert (
        crud.get_backend_state(dataset_name=dset_name, model_name=model_name)
        == enums.State.EVALUATE
    )
    assert (
        crud.get_backend_state(dataset_name=dset_name) == enums.State.EVALUATE
    )

    # attempt to delete dataset
    with pytest.raises(exceptions.StateflowError) as e:
        crud.delete(db=db, dataset_name=dset_name)
    assert "evaluation is running." in str(e)

    # attempt to delete model
    with pytest.raises(exceptions.StateflowError) as e:
        crud.delete(db=db, dataset_name=dset_name, model_name=model_name)
    assert f"{enums.State.EVALUATE} to {enums.State.DELETE}" in str(e)

    # run computation (returns nothing on completion)
    crud.compute_detection_metrics(
        db=db,
        job_request=job_request,
        job_id=resp.job_id,
    )

    # check ready
    assert crud.get_backend_state(dataset_name=dset_name) == enums.State.READY
    assert (
        crud.get_backend_state(dataset_name=dset_name, model_name=model_name)
        == enums.State.READY
    )


def test_stateflow_clf_evaluation(
    db: Session,
    gt_clfs_create: list[schemas.GroundTruth],
    pred_clfs_create: list[schemas.Prediction],
):
    # create dataset
    crud.create_dataset(
        db=db,
        dataset=schemas.Dataset(name=dset_name),
    )
    for gt in gt_clfs_create:
        gt.datum.dataset = dset_name
        crud.create_groundtruth(db=db, groundtruth=gt)
    crud.finalize(db=db, dataset_name=dset_name)

    # create model
    crud.create_model(db=db, model=schemas.Model(name=model_name))
    for pd in pred_clfs_create:
        pd.model = model_name
        crud.create_prediction(db=db, prediction=pd)
    crud.finalize(db=db, model_name=model_name, dataset_name=dset_name)

    # create clf request
    job_request = schemas.EvaluationJob(
        model=model_name,
        dataset=dset_name,
    )

    # check dataset READY
    assert crud.get_backend_state(dataset_name=dset_name) == enums.State.READY

    # check inference pair READY
    assert (
        crud.get_backend_state(dataset_name=dset_name, model_name=model_name)
        == enums.State.READY
    )

    # create clf evaluation (returns Clf Response)
    resp = crud.create_clf_evaluation(
        db=db,
        job_request=job_request,
    )

    # check EVALUATE
    assert (
        crud.get_backend_state(dataset_name=dset_name, model_name=model_name)
        == enums.State.EVALUATE
    )
    assert (
        crud.get_backend_state(dataset_name=dset_name) == enums.State.EVALUATE
    )

    # attempt to delete dataset
    with pytest.raises(exceptions.StateflowError) as e:
        crud.delete(db=db, dataset_name=dset_name)
    assert "evaluation is running." in str(e)

    # attempt to delete model
    with pytest.raises(exceptions.StateflowError) as e:
        crud.delete(db=db, dataset_name=dset_name, model_name=model_name)
    assert f"{enums.State.EVALUATE} to {enums.State.DELETE}" in str(e)

    # compute clf metrics
    crud.compute_clf_metrics(
        db=db,
        job_request=job_request,
        job_id=resp.job_id,
    )

    # check READY
    assert crud.get_backend_state(dataset_name=dset_name) == enums.State.READY
    assert (
        crud.get_backend_state(dataset_name=dset_name, model_name=model_name)
        == enums.State.READY
    )
