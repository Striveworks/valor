import os
from typing import Annotated

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from valor_api import __version__ as api_version
from valor_api import api_utils, auth, crud, enums, exceptions, logger, schemas
from valor_api.backend import database
from valor_api.logging import (
    handle_http_exception,
    handle_request_validation_exception,
    handle_unhandled_exception,
    log_endpoint_middleware,
)
from valor_api.settings import auth_settings

token_auth_scheme = auth.OptionalHTTPBearer()


app = FastAPI(root_path=os.getenv("API_ROOT_PATH", ""))
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.middleware("http")(log_endpoint_middleware)
app.exception_handler(RequestValidationError)(
    handle_request_validation_exception
)
app.exception_handler(HTTPException)(handle_http_exception)
app.exception_handler(Exception)(handle_unhandled_exception)


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


@app.post(
    "/groundtruths",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
    tags=["GroundTruths"],
)
def create_groundtruths(
    groundtruths: list[schemas.GroundTruth], db: Session = Depends(get_db)
):
    """
    Create a ground truth in the database.

    POST Endpoint: `/groundtruths`

    Parameters
    ----------
    groundtruths : list[schemas.GroundTruth]
        The ground truths to add to the database.
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
        for groundtruth in groundtruths:
            crud.create_groundtruth(db=db, groundtruth=groundtruth)
    except Exception as e:
        raise exceptions.create_http_error(e)


@app.get(
    "/groundtruths/dataset/{dataset_name}/datum/{uid}",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
    tags=["GroundTruths"],
)
def get_groundtruth(
    dataset_name: str, uid: str, db: Session = Depends(get_db)
) -> schemas.GroundTruth | None:
    """
    Fetch a ground truth from the database.

    GET Endpoint: `/groundtruths/dataset/{dataset_name}/datum/{uid}`

    Parameters
    ----------
    dataset_name : str
        The name of the dataset to fetch the ground truth from.
    uid : str
        The UID of the ground truth.
    db : Session
        The database session to use. This parameter is a sqlalchemy dependency and shouldn't be submitted by the user.

    Returns
    -------
    schemas.GroundTruth
        Thee ground truth requested by the user.

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


@app.post(
    "/predictions",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
    tags=["Predictions"],
)
def create_predictions(
    predictions: list[schemas.Prediction],
    db: Session = Depends(get_db),
):
    """
    Create a prediction in the database.

    POST Endpoint: `/predictions`

    Parameters
    ----------
    predictions : list[schemas.Prediction]
        The predictions to add to the database.
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
        for prediction in predictions:
            crud.create_prediction(db=db, prediction=prediction)
    except Exception as e:
        raise exceptions.create_http_error(e)


@app.get(
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


@app.get(
    "/labels",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
    tags=["Labels"],
    description="Fetch labels using optional JSON strings as query parameters.",
)
def get_labels(
    filters: schemas.FilterQueryParams = Depends(),
    db: Session = Depends(get_db),
) -> list[schemas.Label]:
    """
    Fetch all labels in the database.

    GET Endpoint: `/labels`

    Parameters
    ----------
    filters : schemas.FilterQueryParams, optional
        An optional filter to constrain results by.
    db : Session
        The database session to use. This parameter is a sqlalchemy dependency and shouldn't be submitted by the user.

    Returns
    -------
    list[schemas.Label]
        A list of all labels in the database.
    """
    try:
        return crud.get_labels(
            db=db,
            filters=schemas.convert_filter_query_params_to_filter_obj(filters),
        )
    except Exception as e:
        raise exceptions.create_http_error(e)


@app.get(
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
    list[schemas.Label]
        A list of all labels associated with the dataset in the database.

    Raises
    ------
    HTTPException (404)
        If the dataset doesn't exist.
    """
    try:
        return crud.get_labels(
            db=db,
            filters=schemas.Filter(
                dataset_names=[dataset_name],
            ),
            ignore_prediction_labels=True,
        )
    except Exception as e:
        raise exceptions.create_http_error(e)


@app.get(
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
    list[schemas.Label]
        A list of all labels associated with the model in the database.

    Raises
    ------
    HTTPException (404)
        If the model doesn't exist.
    """
    try:
        return crud.get_labels(
            db=db,
            filters=schemas.Filter(
                model_names=[model_name],
            ),
            ignore_groundtruth_labels=True,
        )
    except Exception as e:
        raise exceptions.create_http_error(e)


""" DATASET """


@app.post(
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


@app.get(
    "/datasets",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
    tags=["Datasets"],
    description="Fetch datasets using optional JSON strings as query parameters.",
)
def get_datasets(
    filters: schemas.FilterQueryParams = Depends(),
    db: Session = Depends(get_db),
) -> list[schemas.Dataset]:
    """
    Fetch all datasets from the database.

    GET Endpoint: `/datasets`

    Parameters
    ----------
    filters : schemas.FilterQueryParams, optional
        An optional filter to constrain results by. All fields should be specified as strings in a JSON.
    db : Session
        The database session to use. This parameter is a sqlalchemy dependency and shouldn't be submitted by the user.

    Returns
    -------
    list[schemas.Dataset]
        A list of all datasets stored in the database.
    """
    try:
        return crud.get_datasets(
            db=db,
            filters=schemas.convert_filter_query_params_to_filter_obj(filters),
        )
    except Exception as e:
        raise exceptions.create_http_error(e)


@app.get(
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


@app.get(
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


@app.get(
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


@app.put(
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


@app.get(
    "/data",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
    tags=["Datums"],
    description="Fetch datums using optional JSON strings as query parameters.",
)
def get_datums(
    filters: schemas.FilterQueryParams = Depends(),
    db: Session = Depends(get_db),
) -> list[schemas.Datum]:
    """
    Fetch all datums for a particular dataset.

    GET Endpoint: `/data`

    Parameters
    ----------
    filters : schemas.FilterQueryParams, optional
        An optional filter to constrain results by.
    db : Session
        The database session to use. This parameter is a sqlalchemy dependency and shouldn't be submitted by the user.

    Returns
    -------
    list[schemas.Datum]
        A list of datums.

    Raises
    ------
    HTTPException (404)
        If the dataset or datum doesn't exist.
    """
    try:
        return crud.get_datums(
            db=db,
            filters=schemas.convert_filter_query_params_to_filter_obj(filters),
        )
    except Exception as e:
        raise exceptions.create_http_error(e)


@app.get(
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
        datums = crud.get_datums(
            db=db,
            filters=schemas.Filter(
                dataset_names=[dataset_name],
                datum_uids=[uid],
            ),
        )

        if len(datums) == 0:
            raise exceptions.DatumDoesNotExistError(uid=uid)

        return datums[0]
    except Exception as e:
        raise exceptions.create_http_error(e)


""" MODELS """


@app.post(
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


@app.get(
    "/models",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
    tags=["Models"],
    description="Fetch models using optional JSON strings as query parameters.",
)
def get_models(
    filters: schemas.FilterQueryParams = Depends(),
    db: Session = Depends(get_db),
) -> list[schemas.Model]:
    """
    Fetch all models in the database.

    GET Endpoint: `/models`

    Parameters
    ----------
    filters : schemas.FilterQueryParams, optional
        An optional filter to constrain results by.
    db : Session
        The database session to use. This parameter is a sqlalchemy dependency and shouldn't be submitted by the user.

    Returns
    -------
    list[schemas.Model]
        A list of models.
    """
    return crud.get_models(
        db=db,
        filters=schemas.convert_filter_query_params_to_filter_obj(filters),
    )


@app.get(
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


@app.get(
    "/models/{model_name}/eval-requests",
    dependencies=[Depends(token_auth_scheme)],
    tags=["Models"],
)
def get_model_eval_requests(
    model_name: str, db: Session = Depends(get_db)
) -> list[schemas.EvaluationResponse]:
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
    list[EvaluationResponse]
        The evaluation requessts associated to the model

    Raises
    ------
    HTTPException (404)
        If the model doesn't exist.
    """
    try:
        return crud.get_evaluation_requests_from_model(
            db=db, model_name=model_name
        )
    except Exception as e:
        raise exceptions.create_http_error(e)


@app.get(
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


@app.put(
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


@app.delete(
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


@app.post(
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


@app.get(
    "/evaluations",
    dependencies=[Depends(token_auth_scheme)],
    response_model_exclude_none=True,
    tags=["Evaluations"],
)
def get_evaluations(
    datasets: str | None = None,
    models: str | None = None,
    evaluation_ids: str | None = None,
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
    list[schemas.Evaluation]
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


@app.post("/token", tags=["Authentication"])
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()]
) -> str:
    if not auth.authenticate_user(form_data.username, form_data.password):
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = auth.create_token(data={"some data key": "some data value"})
    return access_token


@app.get(
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


@app.get(
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


@app.get(
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
        database.check_db_connection(db=db, timeout=0)
        return schemas.Readiness(status="ok")
    except Exception:
        raise exceptions.create_http_error(
            error=exceptions.ServiceUnavailable(
                "Could not connect to postgresql."
            )
        )
