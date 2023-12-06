import os
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from velour_api import auth, crud, enums, exceptions, logger, schemas
from velour_api.api_utils import _split_query_params
from velour_api.backend import database, jobs
from velour_api.settings import auth_settings

token_auth_scheme = auth.OptionalHTTPBearer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    database.create_db()
    jobs.connect_to_redis()
    yield


app = FastAPI(
    root_path=os.getenv("API_ROOT_PATH", ""),
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


logger.info(
    f"API started {'WITHOUT' if auth_settings.no_auth else 'WITH'} authentication"
)


def get_db():
    db = database.make_session()
    try:
        yield db
    finally:
        db.close()


""" GROUNDTRUTHS """


@app.post(
    "/groundtruths",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
    tags=["GroundTruths"],
)
def create_groundtruths(
    gt: schemas.GroundTruth, db: Session = Depends(get_db)
):
    try:
        crud.create_groundtruth(db=db, groundtruth=gt)
    except (
        exceptions.DatasetDoesNotExistError,
        exceptions.DatumDoesNotExistError,
    ) as e:
        raise HTTPException(status_code=404, detail=str(e))
    except (
        exceptions.DatasetFinalizedError,
        exceptions.DatumAlreadyExistsError,
    ) as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.get(
    "/groundtruths/dataset/{dataset_name}/datum/{uid}",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
    tags=["GroundTruths"],
)
def get_groundtruth(
    dataset_name: str, uid: str, db: Session = Depends(get_db)
) -> schemas.GroundTruth | None:
    try:
        return crud.get_groundtruth(
            db=db,
            dataset_name=dataset_name,
            datum_uid=uid,
        )
    except (
        exceptions.DatumDoesNotExistError,
        exceptions.DatasetDoesNotExistError,
    ) as e:
        raise HTTPException(status_code=404, detail=str(e))


""" PREDICTIONS """


@app.post(
    "/predictions",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
    tags=["Predictions"],
)
def create_predictions(
    pd: schemas.Prediction,
    db: Session = Depends(get_db),
):
    try:
        crud.create_prediction(db=db, prediction=pd)
    except (
        exceptions.DatasetDoesNotExistError,
        exceptions.ModelDoesNotExistError,
        exceptions.DatumDoesNotExistError,
    ) as e:
        raise HTTPException(status_code=404, detail=str(e))
    except (
        exceptions.DatasetNotFinalizedError,
        exceptions.ModelFinalizedError,
    ) as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.get(
    "/predictions/model/{model_name}/dataset/{dataset_name}/datum/{uid}",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
    tags=["Predictions"],
)
def get_prediction(
    model_name: str, dataset_name: str, uid: str, db: Session = Depends(get_db)
) -> schemas.Prediction | None:
    try:
        return crud.get_prediction(
            db=db,
            model_name=model_name,
            dataset_name=dataset_name,
            datum_uid=uid,
        )
    except (
        exceptions.DatumDoesNotExistError,
        exceptions.DatasetDoesNotExistError,
    ) as e:
        raise HTTPException(status_code=404, detail=str(e))


""" LABELS """


@app.get(
    "/labels",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
    tags=["Labels"],
)
def get_all_labels(db: Session = Depends(get_db)) -> list[schemas.Label]:
    return crud.get_all_labels(db=db)


@app.get(
    "/labels/dataset/{dataset_name}",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
    tags=["Labels"],
)
def get_labels_from_dataset(
    dataset_name: str, db: Session = Depends(get_db)
) -> list[schemas.Label]:
    try:
        return crud.get_dataset_labels(
            db=db,
            filters=schemas.Filter(
                dataset_names=[dataset_name],
            ),
        )
    except exceptions.DatasetDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get(
    "/labels/model/{model_name}",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
    tags=["Labels"],
)
def get_labels_from_model(
    model_name: str, db: Session = Depends(get_db)
) -> list[schemas.Label]:
    try:
        return crud.get_model_labels(
            db=db,
            filters=schemas.Filter(
                models_names=[model_name],
            ),
        )
    except exceptions.ModelDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))


""" DATASET """


@app.post(
    "/datasets",
    status_code=201,
    dependencies=[Depends(token_auth_scheme)],
    tags=["Datasets"],
)
def create_dataset(dataset: schemas.Dataset, db: Session = Depends(get_db)):
    try:
        crud.create_dataset(db=db, dataset=dataset)
    except (exceptions.DatasetAlreadyExistsError,) as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.get(
    "/datasets",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
    tags=["Datasets"],
)
def get_datasets(db: Session = Depends(get_db)) -> list[schemas.Dataset]:
    return crud.get_datasets(db=db)


@app.get(
    "/datasets/{dataset_name}",
    dependencies=[Depends(token_auth_scheme)],
    tags=["Datasets"],
)
def get_dataset(
    dataset_name: str, db: Session = Depends(get_db)
) -> schemas.Dataset:
    try:
        return crud.get_dataset(db=db, dataset_name=dataset_name)
    except exceptions.DatasetDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get(
    "/datasets/{dataset_name}/status",
    dependencies=[Depends(token_auth_scheme)],
    tags=["Datasets"],
)
def get_dataset_status(
    dataset_name: str, db: Session = Depends(get_db)
) -> enums.State:
    try:
        resp = crud.get_backend_state(dataset_name=dataset_name)
        return resp
    except exceptions.DatasetDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.put(
    "/datasets/{dataset_name}/finalize",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
    tags=["Datasets"],
)
def finalize_dataset(dataset_name: str, db: Session = Depends(get_db)):
    try:
        crud.finalize(db=db, dataset_name=dataset_name)
    except exceptions.DatasetIsEmptyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except exceptions.DatasetDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.delete(
    "/datasets/{dataset_name}",
    dependencies=[Depends(token_auth_scheme)],
    tags=["Datasets"],
)
def delete_dataset(
    dataset_name: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    logger.debug(f"request to delete dataset {dataset_name}")
    try:
        background_tasks.add_task(
            crud.delete,
            db=db,
            dataset_name=dataset_name,
        )
    except exceptions.DatasetDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except exceptions.StateflowError as e:
        raise HTTPException(status_code=409, detail=str(e))


""" DATUMS """


@app.get(
    "/data/dataset/{dataset_name}",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
    tags=["Datums"],
)
def get_datums(
    dataset_name: str, db: Session = Depends(get_db)
) -> list[schemas.Datum]:
    try:
        return crud.get_datums(
            db=db,
            request=schemas.Filter(
                dataset_names=[dataset_name],
            ),
        )
    except (
        exceptions.DatumDoesNotExistError,
        exceptions.DatasetDoesNotExistError,
    ) as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get(
    "/data/dataset/{dataset_name}/uid/{uid}",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
    tags=["Datums"],
)
def get_datum(
    dataset_name: str, uid: str, db: Session = Depends(get_db)
) -> schemas.Datum | None:
    try:
        return crud.get_datum(
            db=db,
            dataset_name=dataset_name,
            uid=uid,
        )
    except (
        exceptions.DatumDoesNotExistError,
        exceptions.DatasetDoesNotExistError,
    ) as e:
        raise HTTPException(status_code=404, detail=str(e))


""" MODELS """


@app.post(
    "/models",
    status_code=201,
    dependencies=[Depends(token_auth_scheme)],
    tags=["Models"],
)
def create_model(model: schemas.Model, db: Session = Depends(get_db)):
    try:
        crud.create_model(db=db, model=model)
    except (exceptions.ModelAlreadyExistsError,) as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.get(
    "/models",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
    tags=["Models"],
)
def get_models(db: Session = Depends(get_db)) -> list[schemas.Model]:
    return crud.get_models(db=db)


@app.get(
    "/models/{model_name}",
    dependencies=[Depends(token_auth_scheme)],
    tags=["Models"],
)
def get_model(model_name: str, db: Session = Depends(get_db)) -> schemas.Model:
    try:
        return crud.get_model(db=db, model_name=model_name)
    except exceptions.ModelDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.put(
    "/models/{model_name}/datasets/{dataset_name}/finalize",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
    tags=["Models"],
)
def finalize_inferences(
    dataset_name: str, model_name: str, db: Session = Depends(get_db)
):
    try:
        crud.finalize(
            db=db,
            model_name=model_name,
            dataset_name=dataset_name,
        )
    except (
        exceptions.DatasetIsEmptyError,
        exceptions.ModelIsEmptyError,
    ) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except (
        exceptions.DatasetDoesNotExistError,
        exceptions.ModelDoesNotExistError,
    ) as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.delete(
    "/models/{model_name}",
    dependencies=[Depends(token_auth_scheme)],
    tags=["Models"],
)
def delete_model(model_name: str, db: Session = Depends(get_db)):
    try:
        crud.delete(db=db, model_name=model_name)
    except exceptions.ModelDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except exceptions.StateflowError as e:
        raise HTTPException(status_code=409, detail=str(e))


""" EVALUATION """


@app.post(
    "/evaluations",
    status_code=202,
    dependencies=[Depends(token_auth_scheme)],
    tags=["Evaluations"],
)
def create_evaluation(
    job_request: schemas.EvaluationJob,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> schemas.CreateClfMetricsResponse | schemas.CreateAPMetricsResponse | schemas.CreateSemanticSegmentationMetricsResponse:
    try:
        # create evaluation
        # add metric computation to background tasks
        if job_request.task_type == enums.TaskType.CLASSIFICATION:
            resp = crud.create_clf_evaluation(db=db, job_request=job_request)
            background_tasks.add_task(
                crud.compute_clf_metrics,
                db=db,
                job_request=job_request,
                job_id=resp.job_id,
            )
        elif job_request.task_type == enums.TaskType.DETECTION:
            resp = crud.create_detection_evaluation(
                db=db, job_request=job_request
            )
            background_tasks.add_task(
                crud.compute_detection_metrics,
                db=db,
                job_request=job_request,
                job_id=resp.job_id,
            )
        elif job_request.task_type == enums.TaskType.SEGMENTATION:
            resp = crud.create_semantic_segmentation_evaluation(
                db=db, job_request=job_request
            )
            background_tasks.add_task(
                crud.compute_semantic_segmentation_metrics,
                db=db,
                job_request=job_request,
                job_id=resp.job_id,
            )
        else:
            raise ValueError(
                f"Evaluation method for task type `{str(job_request.task_type)}` does not exist."
            )
        return resp
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except (
        exceptions.DatasetNotFinalizedError,
        exceptions.ModelNotFinalizedError,
    ) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except exceptions.StateflowError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.get(
    "/evaluations",
    dependencies=[Depends(token_auth_scheme)],
    response_model_exclude_none=True,
    tags=["Evaluations"],
)
def get_bulk_evaluations(
    datasets: str = None,
    models: str = None,
    db: Session = Depends(get_db),
) -> list[schemas.Evaluation]:
    """
    Returns all metrics associated with user-supplied dataset and model names. Users
    may query using model names, dataset names, or both. All metrics for all specified
    models and datasets will be returned in a list of Evaluations.

    Parameters
    ----------
    datasets
        An optional set of dataset names to return metrics for
    models
        An optional set of model names to return metrics for
    """
    model_names = _split_query_params(models)
    dataset_names = _split_query_params(datasets)

    try:
        return crud.get_evaluations(
            db=db, dataset_names=dataset_names, model_names=model_names
        )
    except (ValueError,) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except (
        exceptions.DatasetDoesNotExistError,
        exceptions.ModelDoesNotExistError,
    ) as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get(
    "/evaluations/dataset/{dataset_name}",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
    tags=["Evaluations"],
)
def get_evaluation_jobs_for_dataset(
    dataset_name: str,
) -> dict[str, list[int]]:
    """Returns all of the job ids for a given dataset."""
    return crud.get_evaluation_ids_for_dataset(dataset_name=dataset_name)


@app.get(
    "/evaluations/model/{model_name}",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
    tags=["Evaluations"],
)
def get_evaluation_jobs_for_model(
    model_name: str,
) -> dict[str, list[int]]:
    """Returns all of the job ids for a given model."""
    return crud.get_evaluation_ids_for_model(model_name=model_name)


@app.get(
    "/evaluations/{job_id}",
    dependencies=[Depends(token_auth_scheme)],
    response_model_exclude_none=True,
    tags=["Evaluations"],
)
def get_evaluation(
    job_id: int,
    db: Session = Depends(get_db),
) -> schemas.Evaluation:
    try:
        status = crud.get_evaluation_status(
            job_id=job_id,
        )
        if status != enums.JobStatus.DONE:
            raise HTTPException(
                status_code=404,
                detail=f"No metrics for job {job_id} since its status is {status}",
            )
        output = crud.get_evaluations(db=db, job_ids=[job_id])
        return output[0]
    except (
        exceptions.JobDoesNotExistError,
        AttributeError,
    ) as e:
        if "'NoneType' object has no attribute 'metrics'" in str(e):
            raise HTTPException(
                status_code=404, detail="Evaluation ID does not exist."
            )
        raise HTTPException(status_code=404, detail=str(e))


@app.get(
    "/evaluations/{job_id}/status",
    dependencies=[Depends(token_auth_scheme)],
    tags=["Evaluations"],
)
def get_evaluation_status(job_id: int) -> enums.JobStatus:
    try:
        return crud.get_evaluation_status(
            job_id=job_id,
        )
    except exceptions.JobDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get(
    "/evaluations/{job_id}/settings",
    dependencies=[Depends(token_auth_scheme)],
    response_model_exclude_none=True,
    tags=["Evaluations"],
)
def get_evaluation_job(
    job_id: int,
    db: Session = Depends(get_db),
) -> schemas.EvaluationJob:
    try:
        if job := crud.get_evaluation_jobs(db=db, job_ids=[job_id]):
            return job[0]
        else:
            raise exceptions.JobDoesNotExistError(job_id)
    except exceptions.JobDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))


""" AUTHENTICATION """


@app.get(
    "/user",
    tags=["Authentication"],
)
def user(
    token: HTTPAuthorizationCredentials | None = Depends(token_auth_scheme),
) -> schemas.User:
    token_payload = auth.verify_token(token)
    return schemas.User(email=token_payload.get("email"))
