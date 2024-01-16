from sqlalchemy import and_, func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from velour_api import enums, exceptions, schemas
from velour_api.backend import core, models
from velour_api.backend.ops import Query


def _db_metric_to_pydantic_metric(metric: models.Metric) -> schemas.Metric:
    """Apply schemas.Metric to a metric from the database"""
    label = (
        schemas.Label(key=metric.label.key, value=metric.label.value)
        if metric.label
        else None
    )
    return schemas.Metric(
        type=metric.type,
        value=metric.value,
        label=label,
        parameters=metric.parameters,
        group=None,
    )


def _verify_ready_to_evaluate(
    db: Session,
    dataset_list: list[models.Dataset],
    model_list: list[models.Model],
):
    """Verifies that the requested datasets and models exist and are ready for evaluation."""

    for model in model_list:
        for dataset in dataset_list:

            # verify dataset status
            match enums.TableStatus(dataset.status):
                case enums.TableStatus.CREATING:
                    raise exceptions.DatasetNotFinalizedError(dataset.name)
                case enums.TableStatus.DELETING | None:
                    raise exceptions.DatasetDoesNotExistError(dataset.name)
                case enums.TableStatus.FINALIZED:
                    pass
                case _:
                    raise RuntimeError

            # verify model status
            match core.get_model_status(
                db=db,
                dataset_name=dataset.name,
                model_name=model.name,
            ):
                case enums.TableStatus.CREATING:
                    raise exceptions.ModelNotFinalizedError(
                        dataset_name=dataset.name,
                        model_name=model.name,
                    )
                case enums.TableStatus.DELETING | None:
                    raise exceptions.ModelDoesNotExistError(model.name)
                case enums.TableStatus.FINALIZED:
                    pass
                case _:
                    raise RuntimeError


def _split_request(
    db: Session,
    job_request: schemas.EvaluationRequest,
) -> list[schemas.EvaluationRequest]:
    """
    Splits a job request into component requests.
    """

    datasets_to_evaluate = db.query(
        Query(models.Dataset).filter(job_request.evaluation_filter).any()
    ).distinct().all()
    if not datasets_to_evaluate:
        raise exceptions.EvaluationRequestError("No datasets meet the requirements of this request.")

    model_to_evaluate = db.query(
        Query(models.Model).filter(job_request.model_filter).any()
    ).distinct().all()
    if not model_to_evaluate:
        raise exceptions.EvaluationRequestError("No models meet the requirements of this request.")

    # verify all models and datasets are ready for evaluation
    _verify_ready_to_evaluate(
        db=db,
        dataset_list=datasets_to_evaluate,
        model_list=model_to_evaluate,
    )

    # dataset_names
    dataset_names = [dataset.name for dataset in datasets_to_evaluate]

    request_list = []
    for model in model_to_evaluate:
        for task_type in job_request.evaluation_filter.task_types:

            # clean model filter
            model_filter = job_request.model_filter.model_copy()
            model_filter.dataset_names = dataset_names
            model_filter.dataset_metadata = None
            model_filter.dataset_geospatial = None
            model_filter.model_names = [model.name]
            model_filter.model_metadata = None
            model_filter.model_geospatial = None

            # clean evaluation filter
            evaluation_filter = job_request.evaluation_filter.model_copy()
            evaluation_filter.dataset_names = dataset_names
            evaluation_filter.dataset_metadata = None
            evaluation_filter.dataset_geospatial = None
            evaluation_filter.model_names = [model.name]
            evaluation_filter.model_metadata = None
            evaluation_filter.model_geospatial = None
            evaluation_filter.task_types = [task_type]

            # some task_types require parameters and/or special filter handling
            match task_type:
                case enums.TaskType.CLASSIFICATION:
                    request_list.append(
                        schemas.EvaluationRequest(
                            model_filter=model_filter,
                            evaluation_filter=evaluation_filter,
                            parameters=schemas.EvaluationParameters(), # default as clf has no parameterization
                        )
                    )
                case enums.TaskType.DETECTION:
                    if evaluation_filter.annotation_types:
                        annotation_types = evaluation_filter.annotation_types.copy()
                        for atype in annotation_types:
                            evaluation_filter.annotation_types = [atype]
                            request_list.append(
                                schemas.EvaluationRequest(
                                    model_filter=model_filter,
                                    evaluation_filter=evaluation_filter,
                                    parameters=job_request.parameters.model_dump(),
                                )
                            )
                    else:
                        request_list.append(
                            schemas.EvaluationRequest(
                                model_filter=model_filter,
                                evaluation_filter=evaluation_filter,
                                parameters=job_request.parameters.model_dump(),
                            )
                        )
                case enums.TaskType.SEGMENTATION:
                    evaluation_filter.annotation_types = [
                        enums.AnnotationType.RASTER
                    ]
                    request_list.append(
                        schemas.EvaluationRequest(
                            model_filter=model_filter,
                            evaluation_filter=evaluation_filter,
                            parameters=schemas.EvaluationParameters(), # default as clf has no parameterization
                        )
                    )
                case _:
                    raise NotImplementedError
    
    return request_list


def _create_response(
    db: Session,
    evaluation: models.Evaluation,
    **kwargs,
) -> schemas.EvaluationResponse:
    """Converts a evaluation row into a response schema."""
    return schemas.EvaluationResponse(
        id=evaluation.id,
        model_filter=evaluation.model_filter,
        evaluation_filter=evaluation.evaluation_filter,
        parameters=evaluation.parameters,
        status=evaluation.status,
        metrics=[
            _db_metric_to_pydantic_metric(metric)
            for metric in evaluation.metrics
        ],
        confusion_matrices=[
            schemas.ConfusionMatrixResponse(
                label_key=matrix.label_key,
                entries=[
                    schemas.ConfusionMatrixEntry(**entry)
                    for entry in matrix.value
                ],
            )
            for matrix in evaluation.confusion_matrices
        ],
        **kwargs,
    )


def _create_responses(
    db: Session,
    evaluations: list[models.Evaluation],
) -> list[schemas.EvaluationResponse]:
    """
    Takes a list of evaluation rows and returns a matching list of evaluation creation responses.

    Parameters
    ----------
    db : Session
        The database session.
    evaluations : list[models.Evaluation]
        A list of evaluation rows to generate responses for.

    Returns
    -------
    list[schemas.EvaluationResponse]
        A list of evaluations in response format.
    """
    results = []
    for evaluation in evaluations:
        if evaluation.id is None:
            raise exceptions.EvaluationDoesNotExistError()

        model_filter = schemas.Filter(**evaluation.model_filter)
        evaluation_filter = schemas.Filter(**evaluation.evaluation_filter)

        if len(evaluation_filter.task_types) != 1:
            raise RuntimeError

        match evaluation_filter.task_types[0]:
            case enums.TaskType.CLASSIFICATION:
                missing_pred_keys, ignored_pred_keys = core.get_disjoint_keys(
                    db, evaluation_filter, model_filter
                )
                kwargs = {
                    "missing_pred_keys": missing_pred_keys,
                    "ignored_pred_keys": ignored_pred_keys,
                }
            case (enums.TaskType.DETECTION | enums.TaskType.SEGMENTATION):
                (
                    missing_pred_labels,
                    ignored_pred_labels,
                ) = core.get_disjoint_labels(
                    db, evaluation_filter, model_filter
                )
                kwargs = {
                    "missing_pred_labels": missing_pred_labels,
                    "ignored_pred_labels": ignored_pred_labels,
                }
            case _:
                raise NotImplementedError
            
        results.append(
            _create_response(
                db=db,
                evaluation=evaluation,
                **kwargs,
            )
        )
    return results


def check_for_active_evaluations(
    db: Session,
    dataset_name: str | None = None,
    model_name: str | None = None,
) -> int:
    """
    Get the number of active evaluations.

    Parameters
    ----------
    db : Session
        Database session instance.
    dataset_name : str, default=None
        Name of a dataset.
    model_name : str, default=None
        Name of a model.

    Returns
    -------
    int
        Number of active evaluations.
    """
    expr = []
    if dataset_name:
        expr.append(
            models.Evaluation.evaluation_filter["dataset_names"].op("?")(
                dataset_name
            )
        )
    if model_name:
        expr.append(
            models.Evaluation.model_filter["model_names"].op("?")(
                model_name
            )
        )

    return db.scalar(
        select(func.count())
        .select_from(models.Evaluation)
        .where(
            or_(
                models.Evaluation.status == enums.EvaluationStatus.PENDING,
                models.Evaluation.status == enums.EvaluationStatus.RUNNING,
            ),
            *expr,
        )
    )


def fetch_evaluation_from_request(
    db: Session,
    job_request: schemas.EvaluationRequest,
) -> models.Evaluation:
    """
    Fetch the row for an evaluation that matches the provided `EvaluationRequest` attributes.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    model : models.Model
        A model row.
    evaluation_filter : schemas.Filter
        The filter from a `EvaluationRequest`.
    parameters : schemas.DetectionParameters, optional
        Any parameters included from an `EvaluationRequest`. These should be in `dict` form.

    Returns
    -------
    models.Evaluation
        The evaluation row.
    """
    evaluation = (
        db.query(models.Evaluation)
        .where(
            and_(
                models.Evaluation.model_filter == job_request.model_filter.model_dump(),
                models.Evaluation.evaluation_filter
                == job_request.evaluation_filter.model_dump(),
                (
                    models.Evaluation.parameters == job_request.parameters.model_dump()
                    if job_request.parameters
                    else models.Evaluation.parameters.is_(None)
                ),
            )
        )
        .one_or_none()
    )
    return evaluation


def fetch_evaluation_from_id(
    db: Session,
    evaluation_id: int,
) -> models.Evaluation:
    """
    Fetches an evaluation row from the database.

    Parameters
    ----------
    db : Session
        The database session.
    evaluation_id : int
        The id of the evaluation.

    Returns
    -------
    models.Evaluation
        The evaluation row with matching id.

    Raises
    ------
    exceptions.EvaluationDoesNotExistError
        If the evaluation id has no corresponding row in the database.
    """
    evaluation = db.scalar(
        select(models.Evaluation).where(models.Evaluation.id == evaluation_id)
    )
    if evaluation is None:
        raise exceptions.EvaluationDoesNotExistError
    return evaluation


def create_or_get_evaluations(
    db: Session,
    job_request: schemas.EvaluationRequest,
) -> tuple[
    list[schemas.EvaluationResponse],
    list[schemas.EvaluationResponse],
]:
    """
    Creates evaluations from evaluation request.

    If an evaluation already exists, it will be returned as running.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    job_request : schemas.EvaluationRequest
        The evaluations to create.

    Returns
    -------
    tuple[list[schemas.EvaluationResponse], list[schemas.EvaluationResponse]]
        A tuple of evaluation response lists following the pattern (list[created_evaluations], list[existing_evaluations])
    """

    created_rows = []
    existing_rows = []
    for subrequest in _split_request(db, job_request):
        # check if evaluation exists
        if evaluation := fetch_evaluation_from_request(
            db=db,
            job_request=subrequest,
        ):
            existing_rows.append(evaluation)

        # create evaluation row
        else:
            evaluation = models.Evaluation(
                model_filter=subrequest.model_filter.model_dump(),
                evaluation_filter=subrequest.evaluation_filter.model_dump(),
                parameters=subrequest.parameters.model_dump() if subrequest.parameters else None,
                status=enums.EvaluationStatus.PENDING,
            )
            created_rows.append(evaluation)

    try:
        db.add_all(created_rows)
        db.commit()
    except IntegrityError:
        db.rollback()
        raise exceptions.EvaluationAlreadyExistsError()

    return (
        _create_responses(db, created_rows),
        _create_responses(db, existing_rows),
    )


def get_evaluations(
    db: Session,
    job_request: schemas.EvaluationRequest,
) -> list[schemas.EvaluationRequest]:
    """
    Get all evaluations that meet the requested parameters.

    Parameters
    ----------
    db : Session
        The database session.
    job_request : list[schemas.EvaluationRequest]
        The evaluation request to search by.

    Returns
    -------
    list[schemas.EvaluationResponse]
        A list of evaluations in response format.
    """
    evaluations = []
    for subrequest in _split_request(db, job_request):
        if evaluation := fetch_evaluation_from_request(
            db=db,
            job_request=subrequest,
        ):
            evaluations.append(evaluation)
    return _create_responses(db, evaluations)


def get_evaluation_ids(
    db: Session,
    job_request: schemas.EvaluationRequest,
) -> list[int]:
    """
    Get the ids for any evaluations that match the provided `EvaluationRequest`.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    job_request : schemas.EvaluationJob
        The evaluation job to create.

    Returns
    -------
    list[int]
        The ids of any matching evaluations.
    """
    evaluation_ids = []
    for subrequest in _split_request(db, job_request):
        if evaluation := fetch_evaluation_from_request(
            db=db,
            job_request=subrequest,
        ):
            evaluation_ids.append(evaluation.id)
    return evaluation_ids


def get_evaluation_status(
    db: Session,
    evaluation_id: int,
) -> enums.EvaluationStatus:
    """
    Get the status of an evaluation.

    Parameters
    ----------
    db : Session
        The database session.
    evaluation_id : int
        The id of the evaluation.

    Returns
    -------
    enums.EvaluationStatus
        The status of the evaluation.
    """
    evaluation = fetch_evaluation_from_id(db, evaluation_id)
    return enums.EvaluationStatus(evaluation.status)


def set_evaluation_status(
    db: Session,
    evaluation_id: int,
    status: enums.EvaluationStatus,
):
    """
    Set the status of an evaluation.

    Parameters
    ----------
    db : Session
        The database session.
    evaluation_id : int
        The id of the evaluation.
    status : enums.EvaluationStatus
        The desired state of the evaluation.

    Raises
    ------
    exceptions.EvaluationStateError
        If the requested state leads to an illegal transition.
    """
    evaluation = fetch_evaluation_from_id(db, evaluation_id)

    current_status = enums.EvaluationStatus(evaluation.status)
    if status not in current_status.next():
        raise exceptions.EvaluationStateError(
            evaluation_id, current_status, status
        )

    try:
        evaluation.status = status
        db.commit()
    except IntegrityError:
        db.rollback()
        raise exceptions.EvaluationStateError(
            evaluation_id, current_status, status
        )


def get_disjoint_labels_from_evaluation_id(
    db: Session,
    evaluation_id: int,
) -> tuple[list[schemas.Label], list[schemas.Label]]:
    """
    Return a tuple containing the unique labels associated with the model and the request.

    Parameters
    ----------
    db : Session
        The database session.
    job_request : schemas.EvaluationJob

    Returns
    -------
    tuple[list[schemas.Label], list[schemas.Label]]
        A tuple of the disjoint label sets. The tuple follows the form (unique to evaluation filter, unique to model filter).
    """
    evaluation = fetch_evaluation_from_id(db, evaluation_id)
    model_filter = schemas.Filter(**evaluation.model_filter)
    evaluation_filter = schemas.Filter(**evaluation.evaluation_filter)
    return core.get_disjoint_labels(db, evaluation_filter, model_filter)
