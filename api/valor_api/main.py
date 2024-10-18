import json
import os
from typing import Annotated

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    HTTPException,
    Query,
    Response,
)
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


""" Classifications """


@app.post(
    "/classifications/{dataset_name}/{model_name}",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
    tags=["Classification"],
)
def create_classifications(
    dataset_name: str,
    model_name: str,
    classifications: list[schemas.Classification],
    ignore_existing_datums: bool = False,
    db: Session = Depends(get_db),
):
    try:
        pass
    except Exception as e:
        raise exceptions.create_http_error(e)


@app.get(
    "/classifications/{dataset_name}/{model_name}/{uid}",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
    tags=["Classification"],
)
def get_classification(
    dataset_name: str, model_name: str, uid: str, db: Session = Depends(get_db)
) -> schemas.Classification | None:
    try:
        pass
    except Exception as e:
        raise exceptions.create_http_error(e)


@app.post(
    "/classifications/{dataset_name}/{model_name}/evaluate",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
    tags=["Classification"],
)
def evaluate_classifications(
    dataset_name: str,
    model_name: str,
    parameters: schemas.Filter,
    ignore_existing_datums: bool = False,
    db: Session = Depends(get_db),
):
    try:
        pass
    except Exception as e:
        raise exceptions.create_http_error(e)


@app.get(
    "/classifications/{dataset_name}/{model_name}/evaluate",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
    tags=["Classification"],
)
def evaluate_filtered_classifications(
    dataset_name: str,
    model_name: str,
    ignore_existing_datums: bool = False,
    db: Session = Depends(get_db),
):
    try:
        pass
    except Exception as e:
        raise exceptions.create_http_error(e)


""" Object Detection """


@app.post(
    "/object_detections/{dataset_name}/{model_name}",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
    tags=["Object Detection"],
)
def create_object_detections(
    dataset_name: str,
    model_name: str,
    detections: list[schemas.ObjectDetection],
    ignore_existing_datums: bool = False,
    db: Session = Depends(get_db),
):
    try:
        pass
    except Exception as e:
        raise exceptions.create_http_error(e)


@app.get(
    "/object_detections/{dataset_name}/{model_name}/{uid}",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
    tags=["Object Detection"],
)
def get_object_detection(
    dataset_name: str, model_name: str, uid: str, db: Session = Depends(get_db)
) -> schemas.ObjectDetection | None:
    try:
        pass
    except Exception as e:
        raise exceptions.create_http_error(e)


""" Semantic Segmenation """


@app.post(
    "/semantic_segmentations/{dataset_name}/{model_name}",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
    tags=["Semantic Segmentation"],
)
def create_semantic_segmentations(
    dataset_name: str,
    model_name: str,
    segmentations: list[schemas.SemanticSegmentation],
    ignore_existing_datums: bool = False,
    db: Session = Depends(get_db),
):
    try:
        pass
    except Exception as e:
        raise exceptions.create_http_error(e)


@app.get(
    "/semantic_segmentations/{dataset_name}/{model_name}/{uid}",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
    tags=["Semantic Segmentation"],
)
def get_semantic_segmentation(
    dataset_name: str, model_name: str, uid: str, db: Session = Depends(get_db)
) -> schemas.SemanticSegmentation | None:
    try:
        pass
    except Exception as e:
        raise exceptions.create_http_error(e)


""" Labels """


@app.get(
    "/labels",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
    tags=["Labels"],
    description="Fetch all labels.",
)
def get_labels(
    response: Response,
    offset: int = Query(
        0, description="The start index of the items to return."
    ),
    limit: int = Query(
        -1,
        description="The number of items to return. Returns all items when set to -1.",
    ),
    db: Session = Depends(get_db),
) -> list[str]:
    """
    Fetch all labels in the database.

    GET Endpoint: `/labels`

    Parameters
    ----------
    response: Response
        The FastAPI response object. Used to return a content-range header to the user.
    offset : int, optional
        The start index of the items to return.
    limit : int, optional
        The number of items to return. Returns all items when set to -1.
    db : Session
        The database session to use. This parameter is a sqlalchemy dependency and shouldn't be submitted by the user.

    Returns
    -------
    list[str]
        A list of all labels in the database.
    """
    try:
        pass
    except Exception as e:
        raise exceptions.create_http_error(e)


@app.post(
    "/labels/filter",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
    tags=["Labels"],
    description="Fetch labels using a filter.",
)
def get_filtered_labels(
    response: Response,
    filters: schemas.Filter,
    offset: int = Query(
        0, description="The start index of the items to return."
    ),
    limit: int = Query(
        -1,
        description="The number of items to return. Returns all items when set to -1.",
    ),
    db: Session = Depends(get_db),
) -> list[str]:
    """
    Fetch labels using a filter.

    POST Endpoint: `/labels/filter`

    Parameters
    ----------
    response: Response
        The FastAPI response object. Used to return a content-range header to the user.
    filters : Filter
        The filter to constrain the results by.
    offset : int, optional
        The start index of the items to return.
    limit : int, optional
        The number of items to return. Returns all items when set to -1.
    db : Session
        The database session to use. This parameter is a sqlalchemy dependency and shouldn't be submitted by the user.

    Returns
    -------
    list[str]
        A list of labels.
    """
    try:
        pass
    except Exception as e:
        raise exceptions.create_http_error(e)


@app.get(
    "/labels/dataset/{dataset_name}",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
    tags=["Labels"],
)
def get_labels_from_dataset(
    response: Response,
    dataset_name: str,
    offset: int = Query(
        0, description="The start index of the items to return."
    ),
    limit: int = Query(
        -1,
        description="The number of items to return. Returns all items when set to -1.",
    ),
    db: Session = Depends(get_db),
) -> list[str]:
    """
    Fetch all labels for a particular dataset from the database.

    GET Endpoint: `/labels/dataset/{dataset_name}`

    Parameters
    ----------
    response: Response
        The FastAPI response object. Used to return a content-range header to the user.
    dataset_name : str
        The name of the dataset.
    offset : int, optional
        The start index of the items to return.
    limit : int, optional
        The number of items to return. Returns all items when set to -1. Returns all items when set to -1.
    db : Session
        The database session to use. This parameter is a sqlalchemy dependency and shouldn't be submitted by the user
    -------
    list[str]
        A list of all labels associated with the dataset in the database.

    Raises
    ------
    HTTPException (404)
        If the dataset doesn't exist.
    """
    try:
        pass

    except Exception as e:
        raise exceptions.create_http_error(e)


@app.get(
    "/labels/model/{model_name}",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
    tags=["Labels"],
)
def get_labels_from_model(
    response: Response,
    model_name: str,
    offset: int = Query(
        0, description="The start index of the items to return."
    ),
    limit: int = Query(
        -1,
        description="The number of items to return. Returns all items when set to -1.",
    ),
    db: Session = Depends(get_db),
) -> list[str]:
    """
    Fetch all labels for a particular model from the database.

    GET Endpoint: `/labels/model/{model_name}`

    Parameters
    ----------
    response: Response
        The FastAPI response object. Used to return a content-range header to the user.
    model_name : str
        The name of the model.
    offset : int, optional
        The start index of the items to return.
    limit : int, optional
        The number of items to return. Returns all items when set to -1.
    db : Session
        The database session to use. This parameter is a sqlalchemy dependency and shouldn't be submitted by the user.

    Returns
    -------
    list[str]
        A list of all labels associated with the model in the database.

    Raises
    ------
    HTTPException (404)
        If the model doesn't exist.
    """
    try:
        pass

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
    response: Response,
    offset: int = Query(
        0, description="The start index of the items to return."
    ),
    limit: int = Query(
        -1,
        description="The number of items to return. Returns all items when set to -1.",
    ),
    db: Session = Depends(get_db),
) -> list[schemas.Dataset]:
    """
    Fetch all datasets from the database.

    GET Endpoint: `/datasets`

    Parameters
    ----------
    response: Response
        The FastAPI response object. Used to return a content-range header to the user.
    filters : schemas.FilterQueryParams, optional
        An optional filter to constrain results by. All fields should be specified as strings in a JSON.
    offset : int, optional
        The start index of the items to return.
    limit : int, optional
        The number of items to return. Returns all items when set to -1.
    db : Session
        The database session to use. This parameter is a sqlalchemy dependency and shouldn't be submitted by the user.

    Returns
    -------
    list[schemas.Dataset]
        A list of all datasets stored in the database.
    """
    try:
        pass
    except Exception as e:
        raise exceptions.create_http_error(e)


@app.post(
    "/datasets/filter",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
    tags=["Datasets"],
    description="Fetch datasets using a filter.",
)
def get_filtered_datasets(
    response: Response,
    filters: schemas.Filter,
    offset: int = Query(
        0, description="The start index of the items to return."
    ),
    limit: int = Query(
        -1,
        description="The number of items to return. Returns all items when set to -1.",
    ),
    db: Session = Depends(get_db),
) -> list[schemas.Dataset]:
    """
    Fetch datasets using a filter.

    POST Endpoint: `/datasets/filter`

    Parameters
    ----------
    response: Response
        The FastAPI response object. Used to return a content-range header to the user.
    filters : Filter
        The filter to constrain the results by.
    offset : int, optional
        The start index of the items to return.
    limit : int, optional
        The number of items to return. Returns all items when set to -1.
    db : Session
        The database session to use. This parameter is a sqlalchemy dependency and shouldn't be submitted by the user.

    Returns
    -------
    list[schemas.Datasets]
        A list of datasets.
    """
    try:
        pass
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
        pass
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
        A FastAPI `BackgroundTasks` object to process the deletion asynchronously. This parameter is a FastAPI dependency and shouldn't be submitted by the user.
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
        pass
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
        pass
    except Exception as e:
        raise exceptions.create_http_error(e)


@app.get(
    "/models",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
    tags=["Models"],
    description="Fetch all models.",
    response_model=list[schemas.Model],
)
def get_models(
    response: Response,
    offset: int = Query(
        0, description="The start index of the items to return."
    ),
    limit: int = Query(
        -1,
        description="The number of items to return. Returns all items when set to -1.",
    ),
    db: Session = Depends(get_db),
) -> list[schemas.Model]:
    """
    Fetch all models in the database.

    GET Endpoint: `/models`

    Parameters
    ----------
    response: Response
        The FastAPI response object. Used to return a content-range header to the user.
    offset : int, optional
        The start index of the items to return.
    limit : int, optional
        The number of items to return. Returns all items when set to -1.
    db : Session
        The database session to use. This parameter is a sqlalchemy dependency and shouldn't be submitted by the user.

    Returns
    -------
    list[schemas.Model]
        A list of models.
    """
    pass


@app.post(
    "/models/filter",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
    tags=["Models"],
    description="Fetch models using a filter.",
)
def get_filtered_models(
    response: Response,
    filters: schemas.Filter,
    offset: int = Query(
        0, description="The start index of the items to return."
    ),
    limit: int = Query(
        -1,
        description="The number of items to return. Returns all items when set to -1.",
    ),
    db: Session = Depends(get_db),
) -> list[schemas.Model]:
    """
    Fetch models using a filter.

    POST Endpoint: `/models/filter`

    Parameters
    ----------
    response: Response
        The FastAPI response object. Used to return a content-range header to the user.
    filters : Filter
        The filter to constrain the results by.
    offset : int, optional
        The start index of the items to return.
    limit : int, optional
        The number of items to return. Returns all items when set to -1.
    db : Session
        The database session to use. This parameter is a sqlalchemy dependency and shouldn't be submitted by the user.

    Returns
    -------
    list[schemas.Model]
        A list of models.
    """
    try:
        pass
    except Exception as e:
        raise exceptions.create_http_error(e)


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
        pass
    except Exception as e:
        raise exceptions.create_http_error(e)


@app.get(
    "/models/{model_name}/summary",
    dependencies=[Depends(token_auth_scheme)],
    tags=["Models"],
)
def get_model_eval_requests(
    model_name: str, db: Session = Depends(get_db)
) -> list[schemas.Evaluation]:
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
    list[Evaluation]
        The evaluation requessts associated to the model

    Raises
    ------
    HTTPException (404)
        If the model doesn't exist.
    """
    try:
        return []
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
        pass
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
