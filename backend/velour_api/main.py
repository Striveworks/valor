import os

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    HTTPException,
    Request,
    status,
)
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from velour_api import auth, crud, enums, exceptions, logger, schemas
from velour_api.backend import database, jobs
from velour_api.settings import auth_settings

token_auth_scheme = auth.OptionalHTTPBearer()


app = FastAPI(root_path=os.getenv("API_ROOT_PATH", ""))
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

database.create_db()

logger.info(
    f"API started {'WITHOUT' if auth_settings.no_auth else 'WITH'} authentication"
)


def get_db():
    db = database.make_session()
    try:
        yield db
    finally:
        db.close()


@app.post("/groundtruth", dependencies=[Depends(token_auth_scheme)])
def create_groundtruths(
    gt: schemas.GroundTruth, db: Session = Depends(get_db)
):
    try:
        crud.create_groundtruth(db=db, groundtruth=gt)
    except exceptions.DatasetIsFinalizedError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.post("/prediction", dependencies=[Depends(token_auth_scheme)])
def create_predictions(
    pd: schemas.Prediction,
    db: Session = Depends(get_db),
):
    try:
        crud.create_prediction(db=db, prediction=pd)
    except (
        exceptions.ModelDoesNotExistError,
        exceptions.DatumDoesNotExistError,
    ) as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get(
    "/datasets", status_code=200, dependencies=[Depends(token_auth_scheme)]
)
def get_datasets(db: Session = Depends(get_db)) -> list[schemas.Dataset]:
    return crud.get_datasets(db)


@app.post(
    "/datasets", status_code=201, dependencies=[Depends(token_auth_scheme)]
)
def create_dataset(dataset: schemas.Dataset, db: Session = Depends(get_db)):
    try:
        crud.create_dataset(db=db, dataset=dataset)
    except exceptions.DatasetAlreadyExistsError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.get("/datasets/{dataset_name}", dependencies=[Depends(token_auth_scheme)])
def get_dataset(
    dataset_name: str, db: Session = Depends(get_db)
) -> schemas.Dataset:
    try:
        return crud.get_dataset(db, name=dataset_name)
    except exceptions.DatasetDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.put(
    "/datasets/{dataset_name}/finalize",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
)
def finalize_dataset(dataset_name: str, db: Session = Depends(get_db)):
    try:
        crud.finalize(db, dataset_name)
    except exceptions.DatasetDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get(
    "/datasets/{dataset_name}/labels",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
)
def get_dataset_labels(
    dataset_name: str, db: Session = Depends(get_db)
) -> list[schemas.LabelDistribution]:
    try:
        return crud.get_labels(db, dataset_name=dataset_name)
    except exceptions.DatasetDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))


# @app.get(
#     "/datasets/{dataset_name}/info",
#     status_code=200,
#     dependencies=[Depends(token_auth_scheme)],
# )
# def get_dataset_info(
#     dataset_name: str, db: Session = Depends(get_db)
# ) -> schemas.Info:
#     try:
#         return crud.get_dataset_info(db, dataset_name)
#     except exceptions.DatasetDoesNotExistError as e:
#         raise HTTPException(status_code=404, detail=str(e))


@app.get(
    "/datasets/{dataset_name}/datums",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
)
def get_dataset_datums(
    dataset_name: str, db: Session = Depends(get_db)
) -> list[schemas.Datum]:
    try:
        return crud.get_datums(
            db, 
            dataset_name,
        )
    except exceptions.DatasetDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))
    

@app.get(
    "/datasets/{dataset_name}/datums/filter/{data_type}",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
)
def get_dataset_datums(
    dataset_name: str, data_type: str, db: Session = Depends(get_db)
) -> list[schemas.Datum]:
    try:
        return crud.get_datums(
            db, 
            dataset_name, 
            filter=data_type
        )
    except exceptions.DatasetDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get(
    "/datasets/{dataset_name}/datums/{uid}",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
)
def get_groundtruth(
    dataset_name: str, uid: str, db: Session = Depends(get_db)
) -> schemas.Datum | None:
    try:
        return crud.get_datum(
            db,
            dataset_name=dataset_name,
            uid=uid,
        )
    except (
        exceptions.DatumDoesNotExistError,
        exceptions.DatasetDoesNotExistError,
    ) as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get(
    "/datasets/{dataset_name}/datums/{uid}/groundtruth",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
)
def get_groundtruth(
    dataset_name: str, uid: str, db: Session = Depends(get_db)
) -> schemas.GroundTruth | None:
    try:
        return crud.get_groundtruth(
            db,
            dataset_name=dataset_name,
            datum_uid=uid,
        )
    except (
        exceptions.DatumDoesNotExistError,
        exceptions.DatasetDoesNotExistError,
    ) as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.delete(
    "/datasets/{dataset_name}", dependencies=[Depends(token_auth_scheme)]
)
def delete_dataset(
    dataset_name: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> str | None:
    logger.debug(f"request to delete dataset {dataset_name}")
    try:
        return crud.delete_dataset(db, dataset_name)
    except exceptions.DatasetDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/models", status_code=200, dependencies=[Depends(token_auth_scheme)])
def get_models(db: Session = Depends(get_db)) -> list[schemas.Model]:
    return crud.get_models(db)


@app.post(
    "/models", status_code=201, dependencies=[Depends(token_auth_scheme)]
)
def create_model(model: schemas.Model, db: Session = Depends(get_db)):
    try:
        crud.create_model(db=db, model=model)
    except exceptions.ModelAlreadyExistsError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.get("/models/{model_name}", dependencies=[Depends(token_auth_scheme)])
def get_model(model_name: str, db: Session = Depends(get_db)) -> schemas.Model:
    try:
        return crud.get_model(db=db, name=model_name)
    except exceptions.ModelDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.delete("/models/{model_name}", dependencies=[Depends(token_auth_scheme)])
def delete_model(model_name: str, db: Session = Depends(get_db)) -> str | None:
    try:
        return crud.delete_model(db, model_name)
    except exceptions.ModelDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get(
    "/models/{model_name}/datum/{uid}/prediction",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
)
def get_prediction(
    model_name: str, uid: str, db: Session = Depends(get_db)
) -> schemas.Prediction | None:
    try:
        return crud.get_prediction(
            db,
            model_name=model_name,
            datum_uid=uid,
        )
    except (
        exceptions.ImageDoesNotExistError,
        exceptions.DatasetDoesNotExistError,
    ) as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get(
    "/models/{model_name}/labels",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
)
def get_model_labels(
    model_name: str, db: Session = Depends(get_db)
) -> list[schemas.ScoredLabelDistribution]:
    try:
        return crud.get_labels_from_model(db, model_name)
    except exceptions.DatasetDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get(
    "/models/{model_name}/evaluation-settings",
    dependencies=[Depends(token_auth_scheme)],
    response_model_exclude_none=True,
)
def get_model_evaluations(
    model_name: str, db: Session = Depends(get_db)
) -> list[schemas.EvaluationSettings]:
    try:
        return crud.get_model_evaluation_settings(db, model_name)
    except exceptions.ModelDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get(
    "/evaluation-settings/{evaluation_settings_id}",
    dependencies=[Depends(token_auth_scheme)],
    response_model_exclude_none=True,
)
def get_evaluation_settings(
    evaluation_settings_id: str, db: Session = Depends(get_db)
):
    try:
        return crud.get_evaluation_settings_from_id(db, evaluation_settings_id)
    except exceptions.ModelDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get(
    "/models/{model_name}/evaluation-settings/{evaluation_settings_id}/metrics",
    dependencies=[Depends(token_auth_scheme)],
)
def get_model_evaluation_metrics(
    model_name: str, evaluation_settings_id: str, db: Session = Depends(get_db)
) -> list[schemas.Metric]:
    # TODO: verify evaluation_settings_id corresponds to given model_name
    return crud.get_metrics_from_evaluation_settings_id(
        db, evaluation_settings_id
    )


@app.get(
    "/models/{model_name}/evaluation-settings/{evaluation_settings_id}/confusion-matrices",
    dependencies=[Depends(token_auth_scheme)],
    response_model_exclude_none=True,
)
def get_model_confusion_matrices(
    evaluation_settings_id: str, db: Session = Depends(get_db)
) -> list[schemas.ConfusionMatrixResponse]:
    return crud.get_confusion_matrices_from_evaluation_settings_id(
        db, evaluation_settings_id
    )


@app.post(
    "/ap-metrics", status_code=202, dependencies=[Depends(token_auth_scheme)]
)
def create_ap_metrics(
    data: schemas.APRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> schemas.CreateAPMetricsResponse:
    try:
        # @TODO: Make this actually a job
        job_id = crud.create_ap_metrics(db, data)
        disjoint_labels = crud.get_disjoint_labels(
            db,
            dataset_name=data.settings.dataset_name,
            model_name=data.settings.model_name,
        )
        return schemas.CreateAPMetricsResponse(
            missing_pred_labels=disjoint_labels["dataset"],
            ignored_pred_labels=disjoint_labels["model"],
            job_id=job_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except (
        exceptions.DatasetIsDraftError,
        exceptions.InferencesAreNotFinalizedError,
    ) as e:
        raise HTTPException(status_code=405, detail=str(e))


@app.post(
    "/clf-metrics", status_code=202, dependencies=[Depends(token_auth_scheme)]
)
def create_clf_metrics(
    data: schemas.ClfMetricsRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> schemas.CreateClfMetricsResponse:
    try:
        # @TODO: Make this actually a job
        job_id = crud.create_clf_metrics(db, data)
        disjoint_keys = crud.get_disjoint_keys(
            db,
            dataset_name=data.settings.dataset_name,
            model_name=data.settings.model_name,
        )
        return schemas.CreateClfMetricsResponse(
            missing_pred_keys=disjoint_keys["dataset"],
            ignored_pred_keys=disjoint_keys["model"],
            job_id=job_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except (
        exceptions.DatasetIsDraftError,
        exceptions.InferencesAreNotFinalizedError,
    ) as e:
        raise HTTPException(status_code=405, detail=str(e))


@app.post("/labels", status_code=200, dependencies=[Depends(token_auth_scheme)])
def get_labels(
    request: schemas.Filter, db: Session = Depends(get_db)
) -> list[schemas.Label]:
    return crud.get_labels(db, request)


@app.get("/user")
def user(
    token: HTTPAuthorizationCredentials | None = Depends(token_auth_scheme),
) -> schemas.User:
    token_payload = auth.verify_token(token)
    return schemas.User(email=token_payload.get("email"))


@app.get("/jobs/{job_id}", dependencies=[Depends(token_auth_scheme)])
def get_job(job_id: str) -> schemas.Job:
    try:
        return jobs.get_job(job_id)
    except exceptions.JobDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get(
    "/jobs/{job_id}/metrics",
    dependencies=[Depends(token_auth_scheme)],
    response_model_exclude_none=True,
)
def get_job_metrics(
    job_id: str, db: Session = Depends(get_db)
) -> list[schemas.Metric]:
    try:
        job = jobs.get_job(job_id)
        if job.status != enums.JobStatus.DONE:
            raise HTTPException(
                status_code=404,
                detail=f"No metrics for job since its status is {job.status}",
            )
        return crud.get_metrics_from_evaluation_settings_id(
            db, job.evaluation_settings_id
        )
    except exceptions.JobDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get(
    "/jobs/{job_id}/confusion-matrices",
    dependencies=[Depends(token_auth_scheme)],
    response_model_exclude_none=True,
)
def get_job_confusion_matrices(
    job_id: str, db: Session = Depends(get_db)
) -> list[schemas.ConfusionMatrixResponse]:
    try:
        job = jobs.get_job(job_id)
        if job.status != enums.JobStatus.DONE:
            raise HTTPException(
                status_code=404,
                detail=f"No metrics for job since its status is {job.status}",
            )
        return crud.get_confusion_matrices_from_evaluation_settings_id(
            db, job.evaluation_settings_id
        )

    except exceptions.JobDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get(
    "/jobs/{job_id}/settings",
    dependencies=[Depends(token_auth_scheme)],
    response_model_exclude_none=True,
)
def get_job_settings(
    job_id: str, db: Session = Depends(get_db)
) -> schemas.EvaluationSettings:
    try:
        job = jobs.get_job(job_id)
        if job.status != enums.JobStatus.DONE:
            raise HTTPException(
                status_code=404,
                detail=f"No settings for job since its status is {job.status}",
            )
        return crud.get_evaluation_settings_from_id(
            db, job.evaluation_settings_id
        )
    except exceptions.JobDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.put(
    "/models/{model_name}/inferences/{dataset_name}/finalize",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
)
def finalize_inferences(
    model_name: str, dataset_name: str, db: Session = Depends(get_db)
):
    try:
        crud.finalize(
            db, dataset_name=dataset_name, model_name=model_name
        )
    except (
        exceptions.DatasetDoesNotExistError,
        exceptions.ModelDoesNotExistError,
    ) as e:
        raise HTTPException(status_code=404, detail=str(e))
