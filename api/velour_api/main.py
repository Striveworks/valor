import os
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from velour_api import auth, crud, enums, exceptions, logger, schemas
from velour_api.api_utils import _split_query_params
from velour_api.backend import database
from velour_api.crud import jobs
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


def create_background_task_with_precheck(
    *_,
    handler: BackgroundTasks,
    task: callable,
    **kwargs,
):
    """
    Runs precheck validation on the stateflow decorator before creating a background task.

    Parameters
    ----------
    handler : BackgroundTasks
        The background task handler to add the task to.
    task : Callable
        The callable task function to be executed.
    **kwargs
        Additional keyword arguments to be passed to the task.
    """
    task(precheck=True, **kwargs)
    handler.add_task(
        task,
        **kwargs,
    )


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
    except exceptions.DatasetAlreadyExistsError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.get(
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
    return crud.get_datasets(db=db)


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
    except exceptions.DatasetDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get(
    "/datasets/{dataset_name}/status",
    dependencies=[Depends(token_auth_scheme)],
    tags=["Datasets"],
)
def get_dataset_status(
    dataset_name: str, db: Session = Depends(get_db)
) -> enums.JobStatus:
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
    enums.JobStatus
        The requested state.

    Raises
    ------
    HTTPException (404)
        If the dataset doesn't exist.
    """
    try:
        resp = crud.get_job_status(dataset_name=dataset_name)
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
        create_background_task_with_precheck(
            handler=background_tasks,
            task=crud.delete,
            db=db,
            dataset_name=dataset_name,
        )
    except exceptions.DatasetDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except (
        exceptions.JobStateError,
        exceptions.DatasetNotFinalizedError,
    ) as e:
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
    except (
        exceptions.DatumDoesNotExistError,
        exceptions.DatasetDoesNotExistError,
    ) as e:
        raise HTTPException(status_code=404, detail=str(e))
    except (exceptions.ModelAlreadyExistsError,) as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.get(
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
    except exceptions.ModelDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get(
    "/models/{model_name}/status",
    dependencies=[Depends(token_auth_scheme)],
    tags=["Models"],
)
def get_model_status(
    model_name: str, db: Session = Depends(get_db)
) -> enums.JobStatus:
    """
    Fetch the status of a model.

    Parameters
    ----------
    model_name : str
        The name of the model.
    db : Session
        The database session to use. This parameter is a sqlalchemy dependency and shouldn't be submitted by the user.

    Returns
    -------
    enums.JobStatus
        The requested state.

    Raises
    ------
    HTTPException (404)
        If the model doesn't exist.
    """
    try:
        return crud.get_job_status(model_name=model_name)
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
        create_background_task_with_precheck(
            handler=background_tasks,
            task=crud.delete,
            db=db,
            model_name=model_name,
        )
    except exceptions.ModelDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except (
        exceptions.JobStateError,
        exceptions.ModelNotFinalizedError,
    ) as e:
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
) -> (
    schemas.CreateClfMetricsResponse
    | schemas.CreateDetectionMetricsResponse
    | schemas.CreateSemanticSegmentationMetricsResponse
):
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
    schemas.CreateClfMetricsResponse | schemas.CreateDetectionMetricsResponse | schemas.CreateSemanticSegmentationMetricsResponse
        An evaluation response object.

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
        # create evaluation
        # add metric computation to background tasks
        if job_request.task_type == enums.TaskType.CLASSIFICATION:
            resp = crud.create_clf_evaluation(db=db, job_request=job_request)
            create_background_task_with_precheck(
                handler=background_tasks,
                task=crud.compute_clf_metrics,
                db=db,
                job_request=job_request,
                job_id=resp.job_id,
            )
        elif job_request.task_type == enums.TaskType.DETECTION:
            resp = crud.create_detection_evaluation(
                db=db, job_request=job_request
            )
            create_background_task_with_precheck(
                handler=background_tasks,
                task=crud.compute_detection_metrics,
                db=db,
                job_request=job_request,
                job_id=resp.job_id,
            )
        elif job_request.task_type == enums.TaskType.SEGMENTATION:
            resp = crud.create_semantic_segmentation_evaluation(
                db=db, job_request=job_request
            )
            create_background_task_with_precheck(
                handler=background_tasks,
                task=crud.compute_semantic_segmentation_metrics,
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
        exceptions.DatasetDoesNotExistError,
        exceptions.ModelDoesNotExistError,
    ) as e:
        raise HTTPException(status_code=404, detail=str(e))
    except (
        exceptions.DatasetNotFinalizedError,
        exceptions.ModelNotFinalizedError,
    ) as e:
        raise HTTPException(status_code=405, detail=str(e))
    except exceptions.JobStateError as e:
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
    "/evaluations/{job_id}",
    dependencies=[Depends(token_auth_scheme)],
    response_model_exclude_none=True,
    tags=["Evaluations"],
)
def get_evaluation(
    job_id: int,
    db: Session = Depends(get_db),
) -> schemas.Evaluation:
    """
    Fetch a particular evaluation by its job ID.

    GET Endpoint: `/evaluations/{job_id}`

    Parameters
    ----------
    job_id : int
        The job ID to fetch the evaluation for.
    db : Session
        The database session to use. This parameter is a sqlalchemy dependency and shouldn't be submitted by the user.

    Returns
    -------
    schemas.Evaluation
        The requested evaluation.

    Raises
    ------
    HTTPException (404)
        If the job doesn't have the correct state.
        If the job ID does not exist
    """
    try:
        status = crud.get_job_status(
            evaluation_id=job_id,
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
    """
    Get the status of an evaluation.

    GET Endpoint: `/evaluations/{job_id}/status`

    Parameters
    ----------
    job_id: int
        The job ID to fetch the status of.

    Returns
    -------
    enums.JobStatus
        The status of the job.

    Raises
    ------
    HTTPException (404)
        If the job doesn't exist.
    """
    return crud.get_job_status(
        evaluation_id=job_id,
    )


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
    """
    Fetch an evaluation job.

    GET Endpoint: `/evaluations/{job_id}/settings`

    Parameters
    ----------
    job_id : int
        The job ID to fetch the evaluation for.
    db : Session
        The database session to use. This parameter is a sqlalchemy dependency and shouldn't be submitted by the user.

    Returns
    -------
    schemas.EvaluationJob
        The requested EvaluationJob.

    Raises
    ------
    HTTPException (404)
        If the job doesn't exist.
    """
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
    """
    Verify a user.

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
