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
    img1: schemas.Image,
    img2: schemas.Image,
) -> list[schemas.GroundTruth]:
    return [
        schemas.GroundTruth(
            dataset=dset_name,
            datum=img1.to_datum(),
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
            dataset=dset_name,
            datum=img2.to_datum(),
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
    img1: schemas.Image, img2: schemas.Image
) -> list[schemas.Prediction]:
    return [
        schemas.Prediction(
            model=model_name,
            datum=img1.to_datum(),
            annotations=[
                schemas.ScoredAnnotation(
                    task_type=enums.TaskType.CLASSIFICATION,
                    scored_labels=[
                        schemas.ScoredLabel(
                            label=schemas.Label(key="k1", value="v1"),
                            score=0.2,
                        ),
                        schemas.ScoredLabel(
                            label=schemas.Label(key="k1", value="v2"),
                            score=0.8,
                        ),
                        schemas.ScoredLabel(
                            label=schemas.Label(key="k4", value="v4"),
                            score=1.0,
                        ),
                    ],
                ),
            ],
        ),
        schemas.Prediction(
            model=model_name,
            datum=img2.to_datum(),
            annotations=[
                schemas.ScoredAnnotation(
                    task_type=enums.TaskType.CLASSIFICATION,
                    scored_labels=[
                        schemas.ScoredLabel(
                            label=schemas.Label(key="k2", value="v2"),
                            score=1.0,
                        ),
                        schemas.ScoredLabel(
                            label=schemas.Label(key="k3", value="v3"),
                            score=0.87,
                        ),
                        schemas.ScoredLabel(
                            label=schemas.Label(key="k3", value="v0"),
                            score=0.13,
                        ),
                    ],
                ),
            ],
        ),
    ]


def test_evaluation_job():
    # check that job id: 0 is non-existent
    assert jobs.get_evaluation_job(0) is None

    # test invalid transitions from `None`
    with pytest.raises(exceptions.EvaluationJobDoesNotExistError) as e:
        jobs.set_evaluation_job(0, enums.JobStatus.PROCESSING)
    assert "does not exist" in str(e)
    with pytest.raises(exceptions.EvaluationJobDoesNotExistError) as e:
        jobs.set_evaluation_job(0, enums.JobStatus.DONE)
    assert "does not exist" in str(e)
    with pytest.raises(exceptions.EvaluationJobDoesNotExistError) as e:
        jobs.set_evaluation_job(0, enums.JobStatus.FAILED)
    assert "does not exist" in str(e)

    # check that nothing affected the state
    assert jobs.get_evaluation_job(0) is None

    """test valid transition"""
    jobs.set_evaluation_job(0, enums.JobStatus.PENDING)

    # test invalid transitions from `PENDING`
    with pytest.raises(exceptions.EvaluationJobStateError) as e:
        jobs.set_evaluation_job(0, enums.JobStatus.DONE)
    assert "JobStatus.PENDING =/=> JobStatus.DONE" in str(e)
    with pytest.raises(exceptions.EvaluationJobStateError) as e:
        jobs.set_evaluation_job(0, enums.JobStatus.FAILED)
    assert "JobStatus.PENDING =/=> JobStatus.FAILED" in str(e)

    # test removing PENDING job
    with pytest.raises(exceptions.EvaluationJobStateError) as e:
        jobs.remove_evaluation_job(0)
    assert "cannot remove an actively running job." in str(e)

    """test valid transition"""
    jobs.set_evaluation_job(0, enums.JobStatus.PROCESSING)

    # test removing PROCESSING job
    with pytest.raises(exceptions.EvaluationJobStateError) as e:
        jobs.remove_evaluation_job(0)
    assert "cannot remove an actively running job." in str(e)

    # test invalid transitions from `PROCESSING`
    with pytest.raises(exceptions.EvaluationJobStateError) as e:
        jobs.set_evaluation_job(0, enums.JobStatus.PENDING)
    assert "JobStatus.PROCESSING =/=> JobStatus.PENDING" in str(e)

    """test valid transition"""
    jobs.set_evaluation_job(0, enums.JobStatus.FAILED)

    # test invalid transitions from `DONE`
    with pytest.raises(exceptions.EvaluationJobStateError) as e:
        jobs.set_evaluation_job(0, enums.JobStatus.PROCESSING)
    assert "JobStatus.FAILED =/=> JobStatus.PROCESSING" in str(e)
    with pytest.raises(exceptions.EvaluationJobStateError) as e:
        jobs.set_evaluation_job(0, enums.JobStatus.DONE)
    assert "JobStatus.FAILED =/=> JobStatus.DONE" in str(e)

    """test valid transition"""
    jobs.set_evaluation_job(0, enums.JobStatus.PENDING)
    jobs.set_evaluation_job(0, enums.JobStatus.PROCESSING)
    jobs.set_evaluation_job(0, enums.JobStatus.DONE)

    # test invalid transitions from `DONE`
    with pytest.raises(exceptions.EvaluationJobStateError) as e:
        jobs.set_evaluation_job(0, enums.JobStatus.PENDING)
    assert "JobStatus.DONE =/=> JobStatus.PENDING" in str(e)
    with pytest.raises(exceptions.EvaluationJobStateError) as e:
        jobs.set_evaluation_job(0, enums.JobStatus.PROCESSING)
    assert "JobStatus.DONE =/=> JobStatus.PROCESSING" in str(e)
    with pytest.raises(exceptions.EvaluationJobStateError) as e:
        jobs.set_evaluation_job(0, enums.JobStatus.FAILED)
    assert "JobStatus.DONE =/=> JobStatus.FAILED" in str(e)

    """test job removal"""
    jobs.remove_evaluation_job(0)


def test_stateflow_dataset(db: Session):

    # should have no record of dataset
    assert crud.get_status(dataset_name=dset_name) is None

    # create dataset
    crud.create_dataset(
        db=db,
        dataset=schemas.Dataset(
            name=dset_name,
        ),
    )

    # `create_dataset` does not affect the stateflow
    assert crud.get_status(dataset_name=dset_name) is None

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
    assert crud.get_status(dataset_name=dset_name) == enums.Stateflow.CREATE

    # finalize dataset
    crud.finalize(db=db, dataset_name=dset_name)

    # `finalize` transitions dataset state into READY
    assert crud.get_status(dataset_name=dset_name) == enums.Stateflow.READY

    # delete dataset
    crud.delete(db=db, dataset_name=dset_name)

    # after delete operation completes the record is removed
    assert crud.get_status(dataset_name=dset_name) is None


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
    assert (
        crud.get_status(dataset_name=dset_name, model_name=model_name) is None
    )

    # create model
    crud.create_model(
        db=db,
        model=schemas.Model(
            name=model_name,
        ),
    )

    # check that no record exists for model as no predictions have been added
    assert (
        crud.get_status(dataset_name=dset_name, model_name=model_name) is None
    )

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
                schemas.ScoredAnnotation(
                    task_type=enums.TaskType.CLASSIFICATION,
                    scored_labels=[
                        schemas.ScoredLabel(
                            label=schemas.Label(key="k", value="v"), score=0.9
                        ),
                        schemas.ScoredLabel(
                            label=schemas.Label(key="k", value="w"), score=0.1
                        ),
                    ],
                )
            ],
        ),
    )

    # `create_prediction` transitions model state to CREATE
    assert (
        crud.get_status(dataset_name=dset_name, model_name=model_name)
        == enums.Stateflow.CREATE
    )

    # check that evaluation fails before finalization
    with pytest.raises(exceptions.StateflowError) as e:
        crud.create_ap_evaluation(
            db=db,
            request_info=schemas.APRequest(
                settings=schemas.EvaluationSettings(
                    model=model_name,
                    dataset=dset_name,
                    task_type=enums.TaskType.DETECTION,
                    gt_type=enums.AnnotationType.BOX,
                    pd_type=enums.AnnotationType.BOX,
                    label_key="class",
                ),
                iou_thresholds=[0.2, 0.6],
                ious_to_keep=[0.2],
            ),
        )
    assert "invalid transititon from create to evaluate" in str(e)

    # finalize model over dataset
    crud.finalize(db=db, dataset_name=dset_name, model_name=model_name)

    # `finalize` transitions dataset state into READY
    assert (
        crud.get_status(dataset_name=dset_name, model_name=model_name)
        == enums.Stateflow.READY
    )

    # delete dataset
    crud.delete(db=db, dataset_name=dset_name, model_name=model_name)

    # after delete operation completes the record is removed
    assert (
        crud.get_status(dataset_name=dset_name, model_name=model_name) is None
    )


def test_stateflow_ap_evalutation(db: Session, groundtruths, predictions):
    request_info = schemas.APRequest(
        settings=schemas.EvaluationSettings(
            model=model_name,
            dataset=dset_name,
            task_type=enums.TaskType.DETECTION,
            gt_type=enums.AnnotationType.BOX,
            pd_type=enums.AnnotationType.BOX,
            label_key="class",
        ),
        iou_thresholds=[0.2, 0.6],
        ious_to_keep=[0.2],
    )

    # check ready
    assert crud.get_status(dataset_name=dset_name) == enums.Stateflow.READY
    assert (
        crud.get_status(dataset_name=dset_name, model_name=model_name)
        == enums.Stateflow.READY
    )

    # create evaluation (return AP Response)
    resp = crud.create_ap_evaluation(db=db, request_info=request_info)

    # check in evalutation
    assert (
        crud.get_status(dataset_name=dset_name, model_name=model_name)
        == enums.Stateflow.EVALUATE
    )
    assert crud.get_status(dataset_name=dset_name) == enums.Stateflow.EVALUATE

    # attempt to delete dataset
    with pytest.raises(ValueError) as e:
        crud.delete(db=db, dataset_name=dset_name)
    assert (
        "cannot transition to delete as a evaluation is currently running."
        in str(e)
    )

    # attempt to delete model
    with pytest.raises(exceptions.StateflowError) as e:
        crud.delete(db=db, dataset_name=dset_name, model_name=model_name)
    assert "invalid transititon from evaluate to delete" in str(e)

    # run computation (returns nothing on completion)
    crud.compute_ap_metrics(
        db=db,
        request_info=request_info,
        evaluation_settings_id=resp.evaluation_settings_id,
    )

    # check ready
    assert crud.get_status(dataset_name=dset_name) == enums.Stateflow.READY
    assert (
        crud.get_status(dataset_name=dset_name, model_name=model_name)
        == enums.Stateflow.READY
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
    request_info = schemas.ClfMetricsRequest(
        settings=schemas.EvaluationSettings(
            model=model_name, dataset=dset_name
        )
    )

    # check READY
    assert crud.get_status(dataset_name=dset_name) == enums.Stateflow.READY
    assert (
        crud.get_status(dataset_name=dset_name, model_name=model_name)
        == enums.Stateflow.READY
    )

    # create clf evaluation (returns Clf Response)
    resp = crud.create_clf_evaluation(
        db=db,
        request_info=request_info,
    )

    # check EVALUATE
    assert (
        crud.get_status(dataset_name=dset_name, model_name=model_name)
        == enums.Stateflow.EVALUATE
    )
    assert crud.get_status(dataset_name=dset_name) == enums.Stateflow.EVALUATE

    # attempt to delete dataset
    with pytest.raises(ValueError) as e:
        crud.delete(db=db, dataset_name=dset_name)
    assert (
        "cannot transition to delete as a evaluation is currently running."
        in str(e)
    )

    # attempt to delete model
    with pytest.raises(exceptions.StateflowError) as e:
        crud.delete(db=db, dataset_name=dset_name, model_name=model_name)
    assert "invalid transititon from evaluate to delete" in str(e)

    # compute clf metrics
    crud.compute_clf_metrics(
        db=db,
        request_info=request_info,
        evaluation_settings_id=resp.evaluation_settings_id,
    )

    # check READY
    assert crud.get_status(dataset_name=dset_name) == enums.Stateflow.READY
    assert (
        crud.get_status(dataset_name=dset_name, model_name=model_name)
        == enums.Stateflow.READY
    )


# NOTE: Jobs will be added in PR 2


# def test_add_job():
#     job = Job()
#     jobs.add_job(job)

#     assert jobs.r.get(job.uid) is not None


# def test_get_job():
#     """test that we can add a job to redis and get it back and test that
#     we get an error if a job with a given uid does not exist
#     """
#     job = Job()
#     jobs.add_job(job)

#     retrieved_job = jobs.get_job(job.uid)

#     assert retrieved_job.dict() == job.dict()

#     with pytest.raises(JobDoesNotExistError) as exc_info:
#         jobs.get_job("asdasd")

#     assert "Job with uid" in str(exc_info)


# def test_wrap_metric_computation():
#     """Test that job transition status works"""

#     def f():
#         assert (
#             job.status == JobStatus.PROCESSING == jobs.get_job(job.uid).status
#         )
#         return 1

#     job, wrapped_f = jobs.wrap_metric_computation(f)

#     assert job.status == JobStatus.PENDING == jobs.get_job(job.uid).status
#     wrapped_f()
#     assert job.status == JobStatus.DONE == jobs.get_job(job.uid).status
#     assert (
#         job.evaluation_settings_id
#         == 1
#         == jobs.get_job(job.uid).evaluation_settings_id
#     )

#     def g():
#         assert (
#             job.status == JobStatus.PROCESSING == jobs.get_job(job.uid).status
#         )
#         raise Exception

#     job, wrapped_g = jobs.wrap_metric_computation(g)
#     assert job.status == JobStatus.PENDING == jobs.get_job(job.uid).status
#     with pytest.raises(Exception):
#         wrapped_g()
#     assert job.status == JobStatus.FAILED == jobs.get_job(job.uid).status


# @patch("velour_api.main.crud")
# @patch("velour_api.schemas.uuid4")
# def test_create_ap_metrics_endpoint(uuid4, crud, client: TestClient):
#     """This tests the create AP metrics endpoint, making sure the background tasks
#     and job status functions properly.
#     """
#     crud.validate_create_ap_metrics.return_value = ([], [])
#     # prescribe the job id so the job itself knows what it is
#     uuid4.return_value = "1"

#     example_json = {
#         "settings": {
#             "model_name": "",
#             "dataset_name": "",
#             "model_pred_task_type": "Bounding Box Object Detection",
#             "dataset_gt_task_type": "Bounding Box Object Detection",
#             "label_key": "class",
#         },
#     }

#     # create a patch of the create ap metrics method that checks
#     # that the job it corresponds to is in the processing state
#     def patch_create_ap_metrics(*args, **kwargs):
#         assert jobs.get_job("1").status == JobStatus.PROCESSING
#         return 2

#     crud.create_ap_metrics = patch_create_ap_metrics

#     resp = client.post("/ap-metrics", json=example_json)
#     assert resp.status_code == 202

#     cm_resp = resp.json()
#     job_id = cm_resp["job_id"]
#     assert job_id == "1"
#     job = jobs.get_job(job_id)
#     assert job.status == JobStatus.DONE
#     assert job.evaluation_settings_id == 2
