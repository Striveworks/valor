import os
from contextlib import asynccontextmanager

import sqlalchemy
from fastapi import APIRouter, BackgroundTasks, Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from velour_api import __version__ as api_version
from velour_api import (
    api_utils,
    auth,
    crud,
    enums,
    exceptions,
    logger,
    schemas,
)
from velour_api.backend import database
from velour_api.logging import LoggingRoute
from velour_api.settings import auth_settings

token_auth_scheme = auth.OptionalHTTPBearer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    database.create_db()
    yield


app = FastAPI(
    root_path=os.getenv("API_ROOT_PATH", ""),
    lifespan=lifespan,
)
router = APIRouter(route_class=LoggingRoute)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


logger.info(
    "API server %s started %s authentication",
    api_version,
    "WITHOUT" if auth_settings.no_auth else "WITH",
)


def get_db():
    db = database.make_session()
    try:
        yield db
    finally:
        db.close()


""" GROUNDTRUTHS """


@router.post(
    "/groundtruths",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
    tags=["GroundTruths"],
)
def create_groundtruths(
    gt: schemas.GroundTruth, db: Session = Depends(get_db)
):
    """
    Create a groundtruth in the database.

    POST Endpoint: `/groundtruths`

    Parameters
    ----------
    gt : schemas.GroundTruth
        The groundtruth to add to the database.
    db : Session
        The database session to use. This parameter is a sqlalchemy dependency and shouldn't be submitted by the user.

    Raises
    ------
    HTTPException (404)
        If the dataset or datum doesn't exist.
    HTTPException (409)
        If the dataset has been finalized, or if the datum already exists.
    """
    try:
        crud.create_groundtruth(db=db, groundtruth=gt)
    except Exception as e:
        raise exceptions.create_http_error(e)


@router.get(
    "/groundtruths/dataset/{dataset_name}/datum/{uid}",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
    tags=["GroundTruths"],
)
def get_groundtruth(
    dataset_name: str, uid: str, db: Session = Depends(get_db)
) -> schemas.GroundTruth | None:
    """
    Fetch a groundtruth from the database.

    GET Endpoint: `/groundtruths/dataset/{dataset_name}/datum/{uid}`

    Parameters
    ----------
    dataset_name : str
        The name of the dataset to fetch the groundtruth from.
    uid : str
        The UID of the groundtruth.
    db : Session
        The database session to use. This parameter is a sqlalchemy dependency and shouldn't be submitted by the user.

    Returns
    -------
    schemas.GroundTruth
        Thee groundtruth requested by the user.

    Raises
    ------
    HTTPException (404)
        If the dataset or datum does not exist.
    """
    try:
        return crud.get_groundtruth(
            db=db,
            dataset_name=dataset_name,
            datum_uid=uid,
        )
    except Exception as e:
        raise exceptions.create_http_error(e)


""" PREDICTIONS """


@router.post(
    "/predictions",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
    tags=["Predictions"],
)
def create_predictions(
    pd: schemas.Prediction,
    db: Session = Depends(get_db),
):
    """
    Create a prediction in the database.

    POST Endpoint: `/predictions`

    Parameters
    ----------
    pd : schemas.Prediction
        The prediction to add to the database.
    db : Session
        The database session to use. This parameter is a sqlalchemy dependency and shouldn't be submitted by the user.

    Raises
    ------
    HTTPException (404)
        If the dataset, model, or datum doesn't exist.
    HTTPException (409)
        If the model has been finalized, or if the dataset has not been finalized.
    """
    try:
        crud.create_prediction(db=db, prediction=pd)
    except Exception as e:
        raise exceptions.create_http_error(e)


@router.get(
    "/predictions/model/{model_name}/dataset/{dataset_name}/datum/{uid}",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
    tags=["Predictions"],
)
def get_prediction(
    model_name: str, dataset_name: str, uid: str, db: Session = Depends(get_db)
) -> schemas.Prediction | None:
    """
    Fetch a prediction from the database.

    GET Endpoint: `/predictions/model/{model_name}/dataset/{dataset_name}/datum/{uid}`

    Parameters
    ----------
    model_name : str
        The name of the model associated with the prediction.
    dataset_name : str
        The name of the dataset associated with the prediction.
    uid : str
        The UID associated with the prediction.
    db : Session
        The database session to use. This parameter is a sqlalchemy dependency and shouldn't be submitted by the user.

    Returns
    -------
    schemas.Prediction
        The requested prediction.

    Raises
    ------
    HTTPException (404)
        If the dataset or datum doesn't exist.
    """
    try:
        return crud.get_prediction(
            db=db,
            model_name=model_name,
            dataset_name=dataset_name,
            datum_uid=uid,
        )
    except Exception as e:
        raise exceptions.create_http_error(e)


""" LABELS """


@router.get(
    "/labels",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
    tags=["Labels"],
)
def get_all_labels(db: Session = Depends(get_db)) -> list[schemas.Label]:
    """
    Fetch all labels in the database.

    GET Endpoint: `/labels`

    Parameters
    ----------
    db : Session
        The database session to use. This parameter is a sqlalchemy dependency and shouldn't be submitted by the user.

    Returns
    -------
    List[schemas.Label]
        A list of all labels in the database.
    """
    try:
        return crud.get_all_labels(db=db)
    except Exception as e:
        raise exceptions.create_http_error(e)


@router.get(
    "/labels/dataset/{dataset_name}",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
    tags=["Labels"],
)
def get_labels_from_dataset(
    dataset_name: str, db: Session = Depends(get_db)
) -> list[schemas.Label]:
    """
    Fetch all labels for a particular dataset from the database.

    GET Endpoint: `/labels/dataset/{dataset_name}`

    Parameters
    ----------
    dataset_name : str
        The name of the dataset.
    db : Session
        The database session to use. This parameter is a sqlalchemy dependency and shouldn't be submitted by the user.

    Returns
    -------
    List[schemas.Label]
        A list of all labels associated with the dataset in the database.

    Raises
    ------
    HTTPException (404)
        If the dataset doesn't exist.
    """
    try:
        return crud.get_dataset_labels(
            db=db,
            filters=schemas.Filter(
                dataset_names=[dataset_name],
            ),
        )
    except Exception as e:
        raise exceptions.create_http_error(e)


@router.get(
    "/labels/model/{model_name}",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
    tags=["Labels"],
)
def get_labels_from_model(
    model_name: str, db: Session = Depends(get_db)
) -> list[schemas.Label]:
    """
    Fetch all labels for a particular model from the database.

    GET Endpoint: `/labels/model/{model_name}`

    Parameters
    ----------
    model_name : str
        The name of the model.
    db : Session
        The database session to use. This parameter is a sqlalchemy dependency and shouldn't be submitted by the user.

    Returns
    -------
    List[schemas.Label]
        A list of all labels associated with the model in the database.

    Raises
    ------
    HTTPException (404)
        If the model doesn't exist.
    """
    try:
        return crud.get_model_labels(
            db=db,
            filters=schemas.Filter(
                model_names=[model_name],
            ),
        )
    except Exception as e:
        raise exceptions.create_http_error(e)


""" DATASET """


@router.post(
    "/datasets",
    status_code=201,
    dependencies=[Depends(token_auth_scheme)],
    tags=["Datasets"],
)
def create_dataset(dataset: schemas.Dataset, db: Session = Depends(get_db)):
    """
    Create a dataset in the database.

    POST Endpoint: `/datasets`

    Parameters
    ----------
    dataset : schemas.Dataset
        The dataset to add to the database.
    db : Session
        The database session to use. This parameter is a sqlalchemy dependency and shouldn't be submitted by the user.

    Raises
    ------
    HTTPException (409)
        If the dataset already exists.
    """
    try:
        crud.create_dataset(db=db, dataset=dataset)
    except Exception as e:
        raise exceptions.create_http_error(e)


@router.get(
    "/datasets",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
    tags=["Datasets"],
)
def get_datasets(db: Session = Depends(get_db)) -> list[schemas.Dataset]:
    """
    Fetch all datasets from the database.

    GET Endpoint: `/datasets`

    Parameters
    ----------
    db : Session
        The database session to use. This parameter is a sqlalchemy dependency and shouldn't be submitted by the user.

    Returns
    -------
    List[schemas.Dataset]
        A list of all datasets stored in the database.
    """
    try:
        return crud.get_datasets(db=db)
    except Exception as e:
        raise exceptions.create_http_error(e)


@router.get(
    "/datasets/{dataset_name}",
    dependencies=[Depends(token_auth_scheme)],
    tags=["Datasets"],
)
def get_dataset(
    dataset_name: str, db: Session = Depends(get_db)
) -> schemas.Dataset:
    """
    Fetch a particular dataset from the database.

    GET Endpoint: `/datasets/{dataset_name}`

    Parameters
    ----------
    dataset_name : str
        The name of the dataset.
    db : Session
        The database session to use. This parameter is a sqlalchemy dependency and shouldn't be submitted by the user.

    Returns
    -------
    schemas.Dataset
        The requested dataset.

    Raises
    ------
    HTTPException (404)
        If the dataset doesn't exist.
    """
    try:
        return crud.get_dataset(db=db, dataset_name=dataset_name)
    except Exception as e:
        raise exceptions.create_http_error(e)


@router.get(
    "/datasets/{dataset_name}/status",
    dependencies=[Depends(token_auth_scheme)],
    tags=["Datasets"],
)
def get_dataset_status(
    dataset_name: str, db: Session = Depends(get_db)
) -> enums.TableStatus:
    """
    Fetch the status of a dataset.

    GET Endpoint: `/datasets/{dataset_name}/status`

    Parameters
    ----------
    dataset_name : str
        The name of the dataset.
    db : Session
        The database session to use. This parameter is a sqlalchemy dependency and shouldn't be submitted by the user.

    Returns
    -------
    enums.TableStatus
        The requested state.

    Raises
    ------
    HTTPException (404)
        If the dataset doesn't exist.
    """
    try:
        resp = crud.get_table_status(db=db, dataset_name=dataset_name)
        return resp
    except Exception as e:
        raise exceptions.create_http_error(e)


@router.get(
    "/datasets/{dataset_name}/summary",
    dependencies=[Depends(token_auth_scheme)],
    tags=["Datasets"],
)
def get_dataset_summary(
    dataset_name: str, db: Session = Depends(get_db)
) -> schemas.DatasetSummary:
    """
    Get the summary of a dataset.

    GET Endpoint: `/datasets/{dataset_name}/summary`

    Parameters
    ----------
    dataset_name : str
        The name of the dataset.
    db : Session
        The database session to use. This parameter is a sqlalchemy dependency and shouldn't be submitted by the user.

    Returns
    -------
    schemas.DatasetSummary
        The dataset summary.

    Raises
    ------
    HTTPException (404)
        If the dataset doesn't exist.
    """
    try:
        resp = crud.get_dataset_summary(db=db, name=dataset_name)
        return resp
    except Exception as e:
        raise exceptions.create_http_error(e)


@router.put(
    "/datasets/{dataset_name}/finalize",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
    tags=["Datasets"],
)
def finalize_dataset(dataset_name: str, db: Session = Depends(get_db)):
    """
    Finalizes a dataset for evaluation.

    PUT Endpoint: `/datasets/{dataset_name}/finalize`

    Parameters
    ----------
    dataset_name : str
        The name of the dataset.
    db : Session
        The database session to use. This parameter is a sqlalchemy dependency and shouldn't be submitted by the user.

    Raises
    ------
    HTTPException (409)
        If the dataset is empty.
    HTTPException (404)
        If the dataset doesn't exist.

    """
    try:
        crud.finalize(db=db, dataset_name=dataset_name)
    except Exception as e:
        raise exceptions.create_http_error(e)


@router.delete(
    "/datasets/{dataset_name}",
    dependencies=[Depends(token_auth_scheme)],
    tags=["Datasets"],
)
def delete_dataset(
    dataset_name: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Delete a dataset from the database.

    DELETE Endpoint: `/datasets/{dataset_name}`

    Parameters
    ----------
    dataset_name : str
        The name of the dataset.
    background_tasks: BackgroundTasks
        A FastAPI `BackgroundTasks` object to process the deletion asyncronously. This parameter is a FastAPI dependency and shouldn't be submitted by the user.
    db : Session
        The database session to use. This parameter is a sqlalchemy dependency and shouldn't be submitted by the user.

    Raises
    ------
    HTTPException (404)
        If the dataset doesn't exist.
    HTTPException (409)
        If the dataset isn't in the correct state to be deleted.
    """
    logger.debug(f"request to delete dataset {dataset_name}")
    try:
        crud.delete(db=db, dataset_name=dataset_name)
    except Exception as e:
        raise exceptions.create_http_error(e)


""" DATUMS """


@router.get(
    "/data/dataset/{dataset_name}",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
    tags=["Datums"],
)
def get_datums(
    dataset_name: str, db: Session = Depends(get_db)
) -> list[schemas.Datum]:
    """
    Fetch all datums for a particular dataset.

    GET Endpoint: `/data/dataset/{dataset_name}`

    Parameters
    ----------
    dataset_name : str
        The name of the dataset.
    db : Session
        The database session to use. This parameter is a sqlalchemy dependency and shouldn't be submitted by the user.

    Returns
    -------
    List[schemas.Datum]
        A list of datums.

    Raises
    ------
    HTTPException (404)
        If the dataset or datum doesn't exist.
    """
    try:
        return crud.get_datums(
            db=db,
            request=schemas.Filter(
                dataset_names=[dataset_name],
            ),
        )
    except Exception as e:
        raise exceptions.create_http_error(e)


@router.get(
    "/data/dataset/{dataset_name}/uid/{uid}",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
    tags=["Datums"],
)
def get_datum(
    dataset_name: str, uid: str, db: Session = Depends(get_db)
) -> schemas.Datum | None:
    """
    Fetch a particular datum.

    GET Endpoint: `/data/dataset/{dataset_name}/uid/{uid}`

    Parameters
    ----------
    dataset_name : str
        The name of the dataset.
    uid : str
        The UID of the datum.
    db : Session
        The database session to use. This parameter is a sqlalchemy dependency and shouldn't be submitted by the user.

    Returns
    -------
    schemas.Datum
        The requested datum.

    Raises
    ------
    HTTPException (404)
        If the dataset or datum doesn't exist.
    """
    try:
        return crud.get_datum(
            db=db,
            dataset_name=dataset_name,
            uid=uid,
        )
    except Exception as e:
        raise exceptions.create_http_error(e)


""" MODELS """


@router.post(
    "/models",
    status_code=201,
    dependencies=[Depends(token_auth_scheme)],
    tags=["Models"],
)
def create_model(model: schemas.Model, db: Session = Depends(get_db)):
    """
    Create a model in the database.

    POST Endpoint: `/models`

    Parameters
    ----------
    model : schemas.Model
        The model to add to the database.
    db : Session
        The database session to use. This parameter is a sqlalchemy dependency and shouldn't be submitted by the user.

    Raises
    ------
    HTTPException (404)
        If the dataset or datum doesn't exist.
    HTTPException (409)
        If the dataset has been finalized, or if the datum already exists.
    """
    try:
        crud.create_model(db=db, model=model)
    except Exception as e:
        raise exceptions.create_http_error(e)


@router.get(
    "/models",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
    tags=["Models"],
)
def get_models(db: Session = Depends(get_db)) -> list[schemas.Model]:
    """
    Fetch all models in the database.

    GET Endpoint: `/models`

    Parameters
    ----------
    db : Session
        The database session to use. This parameter is a sqlalchemy dependency and shouldn't be submitted by the user.

    Returns
    -------
    List[schemas.Model]
        A list of models.
    """
    return crud.get_models(db=db)


@router.get(
    "/models/{model_name}",
    dependencies=[Depends(token_auth_scheme)],
    tags=["Models"],
)
def get_model(model_name: str, db: Session = Depends(get_db)) -> schemas.Model:
    """
    Fetch a particular model.

    GET Endpoint: `/models/{model_name}`

    Parameters
    ----------
    model_name : str
        The name of the model.
    db : Session
        The database session to use. This parameter is a sqlalchemy dependency and shouldn't be submitted by the user.

    Returns
    -------
    schemas.Model
        The requested model.

    Raises
    ------
    HTTPException (404)
        If the model datum doesn't exist.
    """
    try:
        return crud.get_model(db=db, model_name=model_name)
    except Exception as e:
        raise exceptions.create_http_error(e)


@router.get(
    "/models/{model_name}/dataset/{dataset_name}/status",
    dependencies=[Depends(token_auth_scheme)],
    tags=["Models"],
)
def get_model_status(
    dataset_name: str, model_name: str, db: Session = Depends(get_db)
) -> enums.TableStatus:
    """
    Fetch the status of a model over a dataset.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset.
    model_name : str
        The name of the model.
    db : Session
        The database session to use. This parameter is a sqlalchemy dependency and shouldn't be submitted by the user.

    Returns
    -------
    enums.TableStatus
        The requested state.

    Raises
    ------
    HTTPException (404)
        If the model doesn't exist.
    """
    try:
        return crud.get_table_status(
            db=db, dataset_name=dataset_name, model_name=model_name
        )
    except Exception as e:
        raise exceptions.create_http_error(e)


@router.put(
    "/models/{model_name}/datasets/{dataset_name}/finalize",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
    tags=["Models"],
)
def finalize_inferences(
    dataset_name: str, model_name: str, db: Session = Depends(get_db)
):
    """
    Finalize a model prior to evaluation.

    PUT Endpoint: `/models/{model_name}/datasets/{dataset_name}/finalize`

    Parameters
    ----------
    dataset_name : str
        The name of the dataset.
    model_name : str
        The name of the model.
    db : Session
        The database session to use. This parameter is a sqlalchemy dependency and shouldn't be submitted by the user.


    Raises
    ------
    HTTPException (400)
        If the dataset or model are empty.
    HTTPException (404)
        If the dataset or model do not exist.
    """
    try:
        crud.finalize(
            db=db,
            model_name=model_name,
            dataset_name=dataset_name,
        )
    except Exception as e:
        raise exceptions.create_http_error(e)


@router.delete(
    "/models/{model_name}",
    dependencies=[Depends(token_auth_scheme)],
    tags=["Models"],
)
def delete_model(
    model_name: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Delete a model from the database.

    DELETE Endpoint: `/models/{model_name}`

    Parameters
    ----------
    model_name : str
        The name of the model.
    db : Session
        The database session to use. This parameter is a sqlalchemy dependency and shouldn't be submitted by the user.

    Raises
    ------
    HTTPException (404)
        If the model doesn't exist.
    HTTPException (409)
        If the model isn't in the correct state to be deleted.
    """
    try:
        crud.delete(db=db, model_name=model_name)
    except Exception as e:
        raise exceptions.create_http_error(e)


""" EVALUATION """


@router.post(
    "/evaluations",
    status_code=202,
    dependencies=[Depends(token_auth_scheme)],
    tags=["Evaluations"],
)
def create_or_get_evaluations(
    job_request: schemas.EvaluationRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> list[schemas.EvaluationResponse]:
    """
    Create a new evaluation.

    POST Endpoint: `/evaluations`

    Parameters
    ----------
    job_request: schemas.EvaluationJob
        The job request for the evaluation.
    background_tasks: BackgroundTasks
        A FastAPI `BackgroundTasks` object to process the creation asyncronously. This parameter is a FastAPI dependency and shouldn't be submitted by the user.
    db : Session
        The database session to use. This parameter is a sqlalchemy dependency and shouldn't be submitted by the user.

    Returns
    -------
    list[schemas.EvaluationResponse]
        A list of evaluation response objects.

    Raises
    ------
    HTTPException (400)
        If the task type of the evaluation job doesn't exist, or if another ValueError is thrown.
    HTTPException (404)
        If the dataset or model does not exist.
    HTTPException (405)
        If the dataset or model hasn't been finalized.
    HTTPException (409)
        If there is a state exception when creating the evaluation.
    """
    try:
        return crud.create_or_get_evaluations(
            db=db,
            job_request=job_request,
            task_handler=background_tasks,
        )
    except Exception as e:
        raise exceptions.create_http_error(e)


@router.get(
    "/evaluations",
    dependencies=[Depends(token_auth_scheme)],
    response_model_exclude_none=True,
    tags=["Evaluations"],
)
def get_evaluations(
    datasets: str = None,
    models: str = None,
    evaluation_ids: str = None,
    db: Session = Depends(get_db),
) -> list[schemas.EvaluationResponse]:
    """
    Fetch all metrics associated with user-supplied dataset and model names. Users
    may query using model names, dataset names, or both. All metrics for all specified
    models and datasets will be returned in a list of Evaluations.

    This endpoint can handle multiple dataset and model names. For example, you can use
    `/evaluations?models=first_model,second_model&datasets=test_dataset` to get all evaluations
    related to `test_dataset` and either `first_model` or `second_model`.

    GET Endpoint: `/evaluations`

    Parameters
    ----------
    datasets : str
        An optional set of dataset names to return metrics for
    models : str
        An optional set of model names to return metrics for
    evaluation_ids : str
        An optional set of evaluation_ids to return metrics for
    db : Session
        The database session to use. This parameter is a sqlalchemy dependency and shouldn't be submitted by the user.

    Returns
    -------
    List[schemas.Evaluation]
        A list of evaluations.

    Raises
    ------
    HTTPException (400)
        If a ValueError is thrown.
    HTTPException (404)
        If the dataset or model doesn't exist.
    """
    model_names = api_utils._split_query_params(models)
    dataset_names = api_utils._split_query_params(datasets)
    evaluation_ids_str = api_utils._split_query_params(evaluation_ids)

    if evaluation_ids_str:
        try:
            evaluation_ids_ints = [int(id) for id in evaluation_ids_str]
        except Exception as e:
            raise exceptions.create_http_error(e)
    else:
        evaluation_ids_ints = None

    try:
        return crud.get_evaluations(
            db=db,
            evaluation_ids=evaluation_ids_ints,
            dataset_names=dataset_names,
            model_names=model_names,
        )
    except Exception as e:
        raise exceptions.create_http_error(e)


""" AUTHENTICATION """


@router.get(
    "/user",
    tags=["Authentication"],
)
def get_user(
    token: HTTPAuthorizationCredentials | None = Depends(token_auth_scheme),
) -> schemas.User:
    """
    Verify a user and return their email address.

    GET Endpoint: `/user`

    Parameters
    ----------
    token: HTTPAuthorizationCredentials
        The auth token for the user.

    Returns
    -------
    schemas.User
        A response object containing information about the user.
    """
    token_payload = auth.verify_token(token)
    return schemas.User(email=token_payload.get("email"))


@router.get(
    "/api-version",
    tags=["Info"],
    dependencies=[Depends(token_auth_scheme)],
)
def get_api_version() -> schemas.APIVersion:
    """
    Return the API's version.

    GET Endpoint: `/api-version`

    Returns
    -------
    schemas.APIVersion
        A response object containing the API's version number.
    """
    return schemas.APIVersion(api_version=api_version)


""" STATUS """


@router.get(
    "/health",
    tags=["Status"],
)
def health():
    """
    Return 200 if the service is up.

    GET Endpoint: `/health`

    Returns
    -------
    schemas.Health
        A response indicating that the service is up and running.
    """
    return schemas.Health(status="ok")


@router.get(
    "/ready",
    tags=["Status"],
)
def ready(db: Session = Depends(get_db)):
    """
    Return 200 if the service is up and connected to the database.

    GET Endpoint: `/ready`

    Returns
    -------
    schemas.Readiness
        A response indicating that the service is up and connected to the database.
    """
    try:
        db.execute(sqlalchemy.text("select 1"))
        return schemas.Readiness(status="ok")
    except Exception:
        exceptions.create_http_error(
            error=exceptions.ServiceUnavailable(
                "Could not connect to postgresql."
            )
        )
