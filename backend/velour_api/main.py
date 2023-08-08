import os

from fastapi import (  # Request,; status,
    BackgroundTasks,
    Depends,
    FastAPI,
    HTTPException,
)

# from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

# from fastapi.responses import JSONResponse
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
    return crud.get_datasets(db=db)


@app.post(
    "/datasets", status_code=201, dependencies=[Depends(token_auth_scheme)]
)
def create_dataset(dataset: schemas.Dataset, db: Session = Depends(get_db)):
    try:
        crud.create_dataset(db=db, dataset=dataset)
    except (exceptions.DatasetAlreadyExistsError,) as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.get("/datasets/{dataset_name}", dependencies=[Depends(token_auth_scheme)])
def get_dataset(
    dataset_name: str, db: Session = Depends(get_db)
) -> schemas.Dataset:
    try:
        return crud.get_dataset(db=db, dataset_name=dataset_name)
    except exceptions.DatasetDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.put(
    "/datasets/{dataset_name}/finalize",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
)
def finalize_dataset(dataset_name: str, db: Session = Depends(get_db)):
    try:
        crud.finalize(db=db, dataset_name=dataset_name)
    except exceptions.DatasetDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.put(
    "/datasets/{dataset_name}/finalize/{model_name}",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
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
        exceptions.DatasetDoesNotExistError,
        exceptions.ModelDoesNotExistError,
    ) as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get(
    "/datasets/{dataset_name}/labels",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
)
def get_dataset_labels(
    dataset_name: str, db: Session = Depends(get_db)
) -> list[schemas.Label]:
    try:
        return crud.get_labels(
            db=db,
            request=schemas.Filter(
                dataset_names=[dataset_name],
                allow_predictions=False,
            ),
        )
    except exceptions.DatasetDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get(
    "/datasets/{dataset_name}/data",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
)
def get_dataset_datums(
    dataset_name: str, db: Session = Depends(get_db)
) -> list[schemas.Datum]:
    try:
        return crud.get_datums(
            db=db,
            request=schemas.Filter(
                dataset_names=[dataset_name],
                allow_predictions=False,
            ),
        )
    except exceptions.DatasetDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get(
    "/datasets/{dataset_name}/data/filter/{data_type}",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
)
def get_filtered_dataset_datums(
    dataset_name: str, data_type: str, db: Session = Depends(get_db)
) -> list[schemas.Datum]:
    try:
        return crud.get_datums(
            db=db, dataset_name=dataset_name, filter=data_type
        )
    except exceptions.DatasetDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get(
    "/datasets/{dataset_name}/data/{uid}",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
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


@app.get(
    "/datasets/{dataset_name}/data/{uid}/groundtruth",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
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


@app.delete(
    "/datasets/{dataset_name}", dependencies=[Depends(token_auth_scheme)]
)
def delete_dataset(
    dataset_name: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    logger.debug(f"request to delete dataset {dataset_name}")
    try:
        crud.delete(db=db, dataset_name=dataset_name)
    except (
        exceptions.DatasetDoesNotExistError,
        exceptions.StateflowError,
    ) as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/models", status_code=200, dependencies=[Depends(token_auth_scheme)])
def get_models(db: Session = Depends(get_db)) -> list[schemas.Model]:
    return crud.get_models(db=db)


@app.post(
    "/models", status_code=201, dependencies=[Depends(token_auth_scheme)]
)
def create_model(model: schemas.Model, db: Session = Depends(get_db)):
    try:
        crud.create_model(db=db, model=model)
    except (exceptions.ModelAlreadyExistsError,) as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.get("/models/{model_name}", dependencies=[Depends(token_auth_scheme)])
def get_model(model_name: str, db: Session = Depends(get_db)) -> schemas.Model:
    try:
        return crud.get_model(db=db, model_name=model_name)
    except exceptions.ModelDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.delete("/models/{model_name}", dependencies=[Depends(token_auth_scheme)])
def delete_model(model_name: str, db: Session = Depends(get_db)):
    try:
        crud.delete(db=db, model_name=model_name)
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
            db=db,
            model_name=model_name,
            datum_uid=uid,
        )
    except (
        exceptions.DatumDoesNotExistError,
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
) -> list[schemas.Label]:
    try:
        return crud.get_labels(
            db=db,
            request=schemas.Filter(
                model_name=[model_name],
                allow_groundtruths=False,
            ),
        )
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
        return crud.get_model_evaluation_settings(db=db, model_name=model_name)
    except exceptions.ModelDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get(
    "/evaluation-settings/{evaluation_settings_id}",
    dependencies=[Depends(token_auth_scheme)],
    response_model_exclude_none=True,
)
def get_evaluation_settings(
    evaluation_settings_id: int, db: Session = Depends(get_db)
):
    try:
        return crud.get_evaluation_settings_from_id(
            db=db, evaluation_settings_id=evaluation_settings_id
        )
    except exceptions.ModelDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get(
    "/models/{model_name}/evaluation-settings/{evaluation_settings_id}/metrics",
    dependencies=[Depends(token_auth_scheme)],
)
def get_model_evaluation_metrics(
    model_name: str, evaluation_settings_id: int, db: Session = Depends(get_db)
) -> list[schemas.Metric]:
    # TODO: verify evaluation_settings_id corresponds to given model_name
    return crud.get_metrics_from_evaluation_settings_id(
        db=db, evaluation_settings_id=evaluation_settings_id
    )


@app.get(
    "/models/{model_name}/evaluation-settings/{evaluation_settings_id}/confusion-matrices",
    dependencies=[Depends(token_auth_scheme)],
    response_model_exclude_none=True,
)
def get_model_confusion_matrices(
    evaluation_settings_id: int, db: Session = Depends(get_db)
) -> list[schemas.ConfusionMatrixResponse]:
    return crud.get_confusion_matrices_from_evaluation_settings_id(
        db=db, evaluation_settings_id=evaluation_settings_id
    )


@app.post(
    "/ap-metrics", status_code=202, dependencies=[Depends(token_auth_scheme)]
)
def create_ap_metrics(
    request_info: schemas.APRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> schemas.CreateAPMetricsResponse:
    try:
        # create evaluation
        resp = crud.create_ap_evaluation(db=db, request_info=request_info)
        # add metric computation to background tasks
        background_tasks.add_task(
            crud.compute_ap_metrics,
            db=db,
            request_info=request_info,
            evaluation_settings_id=resp.job_id,
        )
        # return AP Response
        return resp
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except (
        exceptions.DatasetIsNotFinalizedError,
        exceptions.InferencesAreNotFinalizedError,
    ) as e:
        raise HTTPException(status_code=405, detail=str(e))
    except (exceptions.StateflowError,) as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.post(
    "/clf-metrics", status_code=202, dependencies=[Depends(token_auth_scheme)]
)
def create_clf_metrics(
    request_info: schemas.ClfMetricsRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> schemas.CreateClfMetricsResponse:
    try:
        # create evaluation
        resp = crud.create_clf_evaluation(db=db, request_info=request_info)
        # add metric computation to background tasks
        background_tasks.add_task(
            crud.compute_clf_metrics,
            db=db,
            request_info=request_info,
            evaluation_settings_id=resp.job_id,
        )
        # return Clf Response
        return resp
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except (
        exceptions.DatasetIsNotFinalizedError,
        exceptions.InferencesAreNotFinalizedError,
    ) as e:
        raise HTTPException(status_code=405, detail=str(e))
    except exceptions.StateflowError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.get("/labels", status_code=200, dependencies=[Depends(token_auth_scheme)])
def get_labels(db: Session = Depends(get_db)) -> list[schemas.Label]:
    return crud.get_labels(db=db)


@app.get("/user")
def user(
    token: HTTPAuthorizationCredentials | None = Depends(token_auth_scheme),
) -> schemas.User:
    token_payload = auth.verify_token(token)
    return schemas.User(email=token_payload.get("email"))


@app.get(
    "/evaluations/{evaluation_settings_id}",
    dependencies=[Depends(token_auth_scheme)],
)
def get_evaluation_status(evaluation_settings_id: int) -> enums.JobStatus:
    try:
        return jobs.get_evaluation_job(evaluation_settings_id)
    except exceptions.EvaluationJobDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get(
    "/evaluations/{evaluation_settings_id}/metrics",
    dependencies=[Depends(token_auth_scheme)],
    response_model_exclude_none=True,
)
def get_evaluation_metrics(
    evaluation_settings_id: int, db: Session = Depends(get_db)
) -> list[schemas.Metric]:
    try:
        status = jobs.get_evaluation_job(evaluation_settings_id)
        if status != enums.JobStatus.DONE:
            raise HTTPException(
                status_code=404,
                detail=f"No metrics for job {evaluation_settings_id} since its status is {status}",
            )
        return crud.get_metrics_from_evaluation_settings_id(
            db=db, evaluation_settings_id=evaluation_settings_id
        )
    except (
        exceptions.EvaluationJobDoesNotExistError,
        AttributeError,
    ) as e:
        if "'NoneType' object has no attribute 'metrics'" in str(e):
            raise HTTPException(
                status_code=404, detail="Evaluation ID does not exist."
            )
        raise HTTPException(status_code=404, detail=str(e))


@app.get(
    "/evaluations/{evaluation_settings_id}/confusion-matrices",
    dependencies=[Depends(token_auth_scheme)],
    response_model_exclude_none=True,
)
def get_job_confusion_matrices(
    evaluation_settings_id: int, db: Session = Depends(get_db)
) -> list[schemas.ConfusionMatrixResponse]:
    try:
        status = jobs.get_evaluation_job(evaluation_settings_id)
        if status != enums.JobStatus.DONE:
            raise HTTPException(
                status_code=404,
                detail=f"No metrics for job {evaluation_settings_id} since its status is {status}",
            )
        return crud.get_confusion_matrices_from_evaluation_settings_id(
            db=db, evaluation_settings_id=evaluation_settings_id
        )
    except exceptions.EvaluationJobDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get(
    "/evaluations/{evaluation_settings_id}/settings",
    dependencies=[Depends(token_auth_scheme)],
    response_model_exclude_none=True,
)
def get_job_settings(
    evaluation_settings_id: int, db: Session = Depends(get_db)
) -> schemas.EvaluationSettings:
    try:
        status = jobs.get_evaluation_job(evaluation_settings_id)
        if status != enums.JobStatus.DONE:
            raise HTTPException(
                status_code=404,
                detail=f"No settings for job {evaluation_settings_id} since its status is {status}",
            )
        return crud.get_evaluation_settings_from_id(
            db=db, evaluation_settings_id=evaluation_settings_id
        )
    except exceptions.EvaluationJobDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))
