import json
import os

from fastapi import (  # Request,; status,
    BackgroundTasks,
    Depends,
    FastAPI,
    HTTPException,
    Request,
)

# from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

# from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from velour_api import auth, crud, enums, exceptions, logger, schemas
from velour_api.backend import database
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


def _split_query_params(param_string: str | None) -> list[str] | None:
    """Split GET query parameters and return a list when possible."""
    if not param_string:
        return None
    elif "," in param_string:
        return param_string.split(",")
    else:
        return [param_string]


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
    except exceptions.DatasetDoesNotExistError as e:
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


app.get(
    "/models/{model_name}/dataset/{dataset_name}/status",
    dependencies=[Depends(token_auth_scheme)],
    tags=["Models"],
)


def get_inference_status(
    model_name: str, dataset_name: str, db: Session = Depends(get_db)
) -> enums.State:
    try:
        return crud.get_backend_state(
            dataset_name=dataset_name, model_name=model_name
        )
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
    "/evaluations/ap-metrics",
    status_code=202,
    dependencies=[Depends(token_auth_scheme)],
    tags=["Evaluations"],
)
def create_detection_metrics(
    job_request: schemas.EvaluationJob,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> schemas.CreateAPMetricsResponse:
    try:
        # create evaluation
        resp = crud.create_detection_evaluation(db=db, job_request=job_request)
        # add metric computation to background tasks
        background_tasks.add_task(
            crud.compute_detection_metrics,
            db=db,
            job_request=job_request,
            job_id=resp.job_id,
        )
        # return AP Response
        return resp
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except (
        exceptions.DatasetNotFinalizedError,
        exceptions.ModelNotFinalizedError,
    ) as e:
        raise HTTPException(status_code=405, detail=str(e))
    except (exceptions.StateflowError,) as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.post(
    "/evaluations/clf-metrics",
    status_code=202,
    dependencies=[Depends(token_auth_scheme)],
    tags=["Evaluations"],
)
def create_clf_metrics(
    job_request: schemas.EvaluationJob,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> schemas.CreateClfMetricsResponse:
    try:
        # create evaluation
        resp = crud.create_clf_evaluation(db=db, job_request=job_request)
        # add metric computation to background tasks
        background_tasks.add_task(
            crud.compute_clf_metrics,
            db=db,
            job_request=job_request,
            job_id=resp.job_id,
        )
        # return Clf Response
        return resp
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except (
        exceptions.DatasetNotFinalizedError,
        exceptions.ModelNotFinalizedError,
    ) as e:
        raise HTTPException(status_code=405, detail=str(e))
    except exceptions.StateflowError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.post(
    "/evaluations/semantic-segmentation-metrics",
    status_code=202,
    dependencies=[Depends(token_auth_scheme)],
    tags=["Evaluations"],
)
def create_semantic_segmentation_metrics(
    job_request: schemas.EvaluationJob,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> schemas.CreateSemanticSegmentationMetricsResponse:
    try:
        # create evaluation
        resp = crud.create_semantic_segmentation_evaluation(
            db=db, job_request=job_request
        )

        # add metric computation to background tasks
        background_tasks.add_task(
            crud.compute_semantic_segmentation_metrics,
            db=db,
            job_request=job_request,
            job_id=resp.job_id,
        )
        return resp
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except (
        exceptions.DatasetNotFinalizedError,
        exceptions.ModelNotFinalizedError,
    ) as e:
        raise HTTPException(status_code=405, detail=str(e))
    except exceptions.StateflowError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.get(
    "/evaluations/datasets/{dataset_name}",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
    tags=["Evaluations"],
)
def get_evaluation_jobs_for_dataset(dataset_name: str) -> dict[str, list[int]]:
    """Returns all of the job ids for a given dataset."""
    return crud.get_evaluation_jobs_for_dataset(dataset_name)


@app.get(
    "/evaluations/models/{model_name}",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
    tags=["Evaluations"],
)
def get_evaluation_jobs_for_model(model_name: str) -> dict[str, list[int]]:
    """Returns all of the job ids for a given model."""
    return crud.get_evaluation_jobs_for_model(model_name)


@app.get(
    "/evaluations/",
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
        return crud.get_bulk_evaluations(
            db=db, dataset_names=dataset_names, model_names=model_names
        )
    except (
        exceptions.DatasetDoesNotExistError,
        exceptions.ModelDoesNotExistError,
        ValueError,
    ) as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get(
    "/ranked-evaluations/",
    dependencies=[Depends(token_auth_scheme)],
    response_model_exclude_none=True,
    tags=["Evaluations"],
)
def get_ranked_evaluations(
    request: Request,
    db: Session = Depends(get_db),
) -> list[schemas.Evaluation]:
    """
    Returns all metrics associated with a particular dataset, ranked according to user inputs

    Parameters
    ----------
    dataset_name
        The dataset name for which to fetch metrics for.
    metric
        The metric to use when ranking evaluations (e.g., "mAP")
    parameters
        The metric parameters to filter on when computing the ranking (e.g., {'iou':.5}). Will raise a ValueError if the user supplies a metric which requires more granular parameters.
    label_keys
        The list of label keys to filter on (e.g., ['key1'])
    rank_from_highest_value_to_lowest_value
        A boolean to indicate whether the metric values should be ranked from highest to lowest
    """
    params = dict(request.query_params)

    try:
        output = crud.get_ranked_evaluations(
            db=db,
            dataset_name=params["dataset_name"],
            metric=params["metric"],
            parameters=json.loads(params["parameters"]),
            label_keys=json.loads(params["label_keys"]),
            rank_from_highest_value_to_lowest_value=bool(
                params["rank_from_highest_value_to_lowest_value"]
            ),
        )
        return output
    except (
        exceptions.DatasetDoesNotExistError,
        exceptions.ModelDoesNotExistError,
        ValueError,
    ) as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get(
    "/evaluations/{job_id}",
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
        return crud.get_evaluation_job_from_id(db=db, evaluation_id=job_id)
    except exceptions.JobDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get(
    "/evaluations/{job_id}/metrics",
    dependencies=[Depends(token_auth_scheme)],
    response_model_exclude_none=True,
    tags=["Evaluations"],
)
def get_evaluation_metrics(
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
        output = crud.get_metrics_from_evaluation_ids(
            db=db, evaluation_id=[job_id]
        )
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
