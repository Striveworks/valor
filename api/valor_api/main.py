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
        crud.create_groundtruths(
            db=db,
            groundtruths=groundtruths,
            ignore_existing_datums=ignore_existing_datums,
        )
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
        return crud.get_groundtruth(
            db=db,
            dataset_name=dataset_name,
            datum_uid=uid,
        )
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
        crud.create_groundtruths(
            db=db,
            groundtruths=groundtruths,
            ignore_existing_datums=ignore_existing_datums,
        )
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
        crud.create_groundtruths(
            db=db,
            groundtruths=groundtruths,
            ignore_existing_datums=ignore_existing_datums,
        )
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
        crud.create_groundtruths(
            db=db,
            groundtruths=groundtruths,
            ignore_existing_datums=ignore_existing_datums,
        )
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
        return crud.get_groundtruth(
            db=db,
            dataset_name=dataset_name,
            datum_uid=uid,
        )
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
        crud.create_groundtruths(
            db=db,
            groundtruths=groundtruths,
            ignore_existing_datums=ignore_existing_datums,
        )
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
        return crud.get_groundtruth(
            db=db,
            dataset_name=dataset_name,
            datum_uid=uid,
        )
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
        content, headers = crud.get_labels(
            db=db,
            filters=schemas.Filter(),
            offset=offset,
            limit=limit,
        )
        response.headers.update(headers)
        return list(content)
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
        content, headers = crud.get_labels(
            db=db,
            filters=filters,
            offset=offset,
            limit=limit,
        )
        response.headers.update(headers)
        return list(content)
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
        content, headers = crud.get_labels(
            db=db,
            filters=schemas.Filter(
                groundtruths=schemas.Condition(
                    lhs=schemas.Symbol(
                        name=schemas.SupportedSymbol.DATASET_NAME
                    ),
                    rhs=schemas.Value.infer(dataset_name),
                    op=schemas.FilterOperator.EQ,
                )
            ),
            ignore_prediction_labels=True,
            offset=offset,
            limit=limit,
        )
        response.headers.update(headers)
        return list(content)

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
        content, headers = crud.get_labels(
            db=db,
            filters=schemas.Filter(
                groundtruths=schemas.Condition(
                    lhs=schemas.Symbol(
                        name=schemas.SupportedSymbol.MODEL_NAME
                    ),
                    rhs=schemas.Value.infer(model_name),
                    op=schemas.FilterOperator.EQ,
                )
            ),
            ignore_groundtruth_labels=True,
            offset=offset,
            limit=limit,
        )
        response.headers.update(headers)
        return list(content)

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
        content, headers = crud.get_datasets(
            db=db,
            filters=schemas.Filter(),
            offset=offset,
            limit=limit,
        )
        response.headers.update(headers)
        return content
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
        content, headers = crud.get_datasets(
            db=db,
            filters=filters,
            offset=offset,
            limit=limit,
        )
        response.headers.update(headers)
        return list(content)
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
        crud.delete(db=db, dataset_name=dataset_name)
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
    content, headers = crud.get_models(
        db=db,
        filters=schemas.Filter(),
        offset=offset,
        limit=limit,
    )

    response.headers.update(headers)

    return content


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
        content, headers = crud.get_models(
            db=db,
            filters=filters,
            offset=offset,
            limit=limit,
        )
        response.headers.update(headers)
        return list(content)
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
        return crud.get_model(db=db, model_name=model_name)
    except Exception as e:
        raise exceptions.create_http_error(e)


@app.get(
    "/models/{model_name}/summary",
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
    response: Response,
    job_request: schemas.EvaluationRequest,
    background_tasks: BackgroundTasks,
    allow_retries: bool = False,
    db: Session = Depends(get_db),
) -> list[schemas.EvaluationResponse]:
    """
    Create a new evaluation.

    POST Endpoint: `/evaluations`

    Parameters
    ----------
    response: Response
        The FastAPI response object. Used to return a content-range header to the user.
    job_request: schemas.EvaluationJob
        The job request for the evaluation.
    background_tasks: BackgroundTasks
        A FastAPI `BackgroundTasks` object to process the creation asynchronously. This parameter is a FastAPI dependency and shouldn't be submitted by the user.
    allow_retries: bool, default = False
        Determines whether failed evaluations are restarted.
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
            allow_retries=allow_retries,
        )
    except Exception as e:
        raise exceptions.create_http_error(e)


@app.get(
    "/evaluations/",
    dependencies=[Depends(token_auth_scheme)],
    response_model_exclude_none=True,
    tags=["Evaluations"],
)
def get_evaluations(
    response: Response,
    datasets: str
    | None = Query(
        None, description="An optional set of dataset names to constrain by."
    ),
    models: str
    | None = Query(
        None, description="An optional set of model names to constrain by."
    ),
    evaluation_ids: str
    | None = Query(
        None, description="An optional set of evaluation_ids to constrain by."
    ),
    offset: int = Query(
        0, description="The start index of the items to return."
    ),
    limit: int = Query(
        -1,
        description="The number of items to return. Returns all items when set to -1.",
    ),
    metrics_to_sort_by: str | None = None,
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
    response: Response
        The FastAPI response object. Used to return a content-range header to the user.
    datasets : str
        An optional set of dataset names to constrain by.
    models : str
        An optional set of model names to constrain by.
    evaluation_ids : str
        An optional set of evaluation_ids to constrain by.
    db : Session
        The database session to use. This parameter is a sqlalchemy dependency and shouldn't be submitted by the user.
    offset : int, optional
        The start index of the items to return.
    limit : int, optional
        The number of items to return. Returns all items when set to -1.
    metrics_to_sort_by: str, optional
        An optional dict of metric types to sort the evaluations by.

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
    metrics_to_sort_by_ = (
        json.loads(metrics_to_sort_by) if metrics_to_sort_by else None
    )

    api_utils.validate_metrics_to_sort_by(metrics_to_sort_by_)

    if evaluation_ids_str:
        try:
            evaluation_ids_ints = [int(id) for id in evaluation_ids_str]
        except Exception as e:
            raise exceptions.create_http_error(e)
    else:
        evaluation_ids_ints = None

    try:
        content, headers = crud.get_evaluations(
            db=db,
            evaluation_ids=evaluation_ids_ints,
            dataset_names=dataset_names,
            model_names=model_names,
            offset=offset,
            limit=limit,
            metrics_to_sort_by=metrics_to_sort_by_,
        )
        response.headers.update(headers)
        return content
    except Exception as e:
        raise exceptions.create_http_error(e)


@app.delete(
    "/evaluations/{evaluation_id}",
    dependencies=[Depends(token_auth_scheme)],
    tags=["Evaluations"],
)
def delete_evaluation(
    evaluation_id: int,
    db: Session = Depends(get_db),
):
    """
    Delete a evaluation from the database.

    DELETE Endpoint: `/evaluations/{evaluation_id}`

    Parameters
    ----------
    evaluation_id : int
        The evaluation identifier.
    db : Session
        The database session to use. This parameter is a sqlalchemy dependency and shouldn't be submitted by the user.

    Raises
    ------
    HTTPException (404)
        If the evaluation doesn't exist.
    HTTPException (409)
        If the evaluation isn't in the correct state to be deleted.
    """
    logger.debug(f"request to delete evaluation {evaluation_id}")
    try:
        crud.delete_evaluation(db=db, evaluation_id=evaluation_id)
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
