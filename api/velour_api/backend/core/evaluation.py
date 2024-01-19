from sqlalchemy import ColumnElement, and_, func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from velour_api import enums, exceptions, schemas
from velour_api.backend import core, models
from velour_api.backend.ops import Query


def _create_name_expr_from_list(
    key: str, names: list[str]
) -> ColumnElement[bool]:
    """
    Creates a sqlalchemy or_ expression from a list of str.

    Note that this is for accessing models.Evaluation with a list of dataset or model names.

    Parameters
    ----------
    key : str
        Key to models.Evaluation.evaluation_fitler. Either "dataset_names" or "model_names".
    names : list[str]
        List of names from either datasets or models.

    Returns
    -------
    ColumnElement[bool]
        The sqlalchemy expression.
    """
    if key == "dataset_names":
        table = models.Evaluation.dataset_filter
    elif key == "model_names":
        table = models.Evaluation.model_filter
    else:
        raise ValueError

    if not names:
        return None
    elif len(names) == 1:
        return table[key].op("?")(names[0])
    else:
        return or_(
            *[
                table[key].op("?")(name)
                for name in names
                if isinstance(name, str)
            ]
        )


def _create_id_expr_from_list(key: str, ids: list[int]) -> ColumnElement[bool]:
    """
    Creates a sqlalchemy or_ expression from a list of int.

    Note that this is for accessing models.Evaluation with a list of ids.

    Parameters
    ----------
    key : str
        Key to models.Evaluation.evaluation_fitler. Either "dataset_names" or "model_names".
    ids : list[int]
        List of evaluations ids.

    Returns
    -------
    ColumnElement[bool]
        The sqlalchemy expression.
    """
    if not ids:
        return None
    elif len(ids) == 1:
        return models.Evaluation.id == ids[0]
    else:
        return or_(
            *[
                models.Evaluation.id == id_
                for id_ in ids
                if isinstance(id_, int)
            ]
        )


def _create_bulk_expression(
    evaluation_ids: list[int] | None = None,
    dataset_names: list[str] | None = None,
    model_names: list[str] | None = None,
) -> ColumnElement[bool]:
    """Creates an expression that queries for evaluations by the input args."""
    expr = []
    if dataset_names:
        expr.append(
            _create_name_expr_from_list("dataset_names", dataset_names)
        )
    if model_names:
        expr.append(_create_name_expr_from_list("model_names", model_names))
    if evaluation_ids:
        expr.append(
            _create_id_expr_from_list("evaluation_ids", evaluation_ids)
        )
    return expr


def _convert_db_metric_to_pydantic_metric(
    db: Session,
    metric: models.Metric,
) -> schemas.Metric:
    """Apply schemas.Metric to a metric from the database"""

    label_row = (
        db.query(
            select(models.Label)
            .where(models.Label.id == metric.label_id)
            .subquery()
        ).one_or_none()
        if metric.label_id
        else None
    )
    label = (
        schemas.Label(key=label_row.key, value=label_row.value)
        if label_row
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

    1. Fetch all datasets and models that conform to their respective filters.
    2. Verify all datasets and models are ready to be evaluated.
    3. For auditability, replace ambiguous dataset/model queries with explicit lists of names.
    4. Create a response per model..
    """

    # 1.a - get all datasets, note this uses the unmodified filter
    datasets_to_evaluate = (
        db.query(
            Query(models.Dataset).filter(job_request.dataset_filter).any()
        )
        .distinct()
        .all()
    )
    if not datasets_to_evaluate:
        raise exceptions.EvaluationRequestError(
            "No datasets meet the requirements of this request."
        )

    # 1.b - get all models, note this uses the unmodified filter
    model_to_evaluate = (
        db.query(Query(models.Model).filter(job_request.model_filter).any())
        .distinct()
        .all()
    )
    if not model_to_evaluate:
        raise exceptions.EvaluationRequestError(
            "No models meet the requirements of this request."
        )

    # 2 - verify all models and datasets are ready for evaluation
    _verify_ready_to_evaluate(
        db=db,
        dataset_list=datasets_to_evaluate,
        model_list=model_to_evaluate,
    )

    # 3.a - convert ambiguous queries into explicit lists of dataset and model names. 
    dataset_filter = job_request.dataset_filter.model_copy()
    model_filter = job_request.model_filter.model_copy()

    dataset_filter.model_names = None
    model_filter.dataset_names = None
     
    dataset_filter.dataset_metadata = None
    model_filter.dataset_metadata = None

    dataset_filter.model_metadata = None
    model_filter.model_metadata = None

    dataset_filter.dataset_geospatial = None
    model_filter.dataset_geospatial = None

    dataset_filter.model_geospatial = None
    model_filter.model_geospatial = None

    # 3. b load dataset names
    dataset_filter.dataset_names = [
        dataset.name for dataset in datasets_to_evaluate
    ]
    
    request_list = []
    for model in model_to_evaluate:
        
        # 3.b - load model name
        model_filter.model_names = [model.name]
        
        # 4. - create request
        request_list.append(
            schemas.EvaluationRequest(
                model_filter=model_filter,
                dataset_filter=dataset_filter,
                parameters=job_request.parameters,
            )
        )
    return request_list


def _create_response(
    db: Session,
    evaluation: models.Evaluation,
    **kwargs,
) -> schemas.EvaluationResponse:
    """Converts a evaluation row into a response schema."""
    metrics = db.query(
        select(models.Metric)
        .where(models.Metric.evaluation_id == evaluation.id)
        .subquery()
    ).all()
    confusion_matrices = db.query(
        select(models.ConfusionMatrix)
        .where(models.ConfusionMatrix.evaluation_id == evaluation.id)
        .subquery()
    ).all()
    return schemas.EvaluationResponse(
        id=evaluation.id,
        model_filter=evaluation.model_filter,
        dataset_filter=evaluation.dataset_filter,
        parameters=evaluation.parameters,
        status=evaluation.status,
        metrics=[
            _convert_db_metric_to_pydantic_metric(db, metric)
            for metric in metrics
        ],
        confusion_matrices=[
            schemas.ConfusionMatrixResponse(
                label_key=matrix.label_key,
                entries=[
                    schemas.ConfusionMatrixEntry(**entry)
                    for entry in matrix.value
                ],
            )
            for matrix in confusion_matrices
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
        dataset_filter = schemas.Filter(**evaluation.dataset_filter)
        parameters = schemas.EvaluationParameters(**evaluation.parameters)

        match parameters.task_type:
            case enums.TaskType.CLASSIFICATION:
                missing_pred_keys, ignored_pred_keys = core.get_disjoint_keys(
                    db, dataset_filter, model_filter
                )
                kwargs = {
                    "missing_pred_keys": missing_pred_keys,
                    "ignored_pred_keys": ignored_pred_keys,
                }
            case (enums.TaskType.DETECTION | enums.TaskType.SEGMENTATION):
                (
                    missing_pred_labels,
                    ignored_pred_labels,
                ) = core.get_disjoint_labels(db, dataset_filter, model_filter)
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


def _fetch_evaluation_from_subrequest(
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
    dataset_filter : schemas.Filter
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
                models.Evaluation.model_filter
                == job_request.model_filter.model_dump(),
                models.Evaluation.dataset_filter
                == job_request.dataset_filter.model_dump(),
                models.Evaluation.parameters
                == job_request.parameters.model_dump(),
            )
        )
        .one_or_none()
    )
    return evaluation


def fetch_evaluations(
    db: Session,
    evaluation_ids: list[int] | None = None,
    dataset_names: list[str] | None = None,
    model_names: list[str] | None = None,
) -> list[models.Evaluation]:
    """
    Returns all evaluations that conform to user-supplied constraints.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    evaluation_ids : list[int], optional
        A list of evaluation job id constraints.
    dataset_names : list[str], optional
        A list of dataset names to constrain by.
    model_names : list[str], optional
        A list of model names to constrain by.

    Returns
    ----------
    list[models.Evaluation]
        A list of evaluations.
    """
    expr = _create_bulk_expression(
        evaluation_ids=evaluation_ids,
        dataset_names=dataset_names,
        model_names=model_names,
    )
    return db.query(models.Evaluation).where(*expr).all()


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
    evaluation = (
        db.query(models.Evaluation)
        .where(models.Evaluation.id == evaluation_id)
        .one_or_none()
    )
    if evaluation is None:
        raise exceptions.EvaluationDoesNotExistError
    return evaluation


def count_active_evaluations(
    db: Session,
    evaluation_ids: list[int] | None = None,
    dataset_names: list[str] | None = None,
    model_names: list[str] | None = None,
) -> int:
    """
    Count the number of active evaluations.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    evaluation_ids : list[int], optional
        A list of evaluation job id constraints.
    dataset_names : list[str], optional
        A list of dataset names to constrain by.
    model_names : list[str], optional
        A list of model names to constrain by.

    Returns
    -------
    int
        Number of active evaluations.
    """
    expr = _create_bulk_expression(
        evaluation_ids=evaluation_ids,
        dataset_names=dataset_names,
        model_names=model_names,
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
        if evaluation := _fetch_evaluation_from_subrequest(
            db=db,
            job_request=subrequest,
        ):
            existing_rows.append(evaluation)

        # create evaluation row
        else:
            evaluation = models.Evaluation(
                model_filter=subrequest.model_filter.model_dump(),
                dataset_filter=subrequest.dataset_filter.model_dump(),
                parameters=subrequest.parameters.model_dump()
                if subrequest.parameters
                else None,
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
    evaluation_ids: list[int] | None = None,
    dataset_names: list[str] | None = None,
    model_names: list[str] | None = None,
) -> list[schemas.EvaluationResponse]:
    """
    Returns all evaluations that conform to user-supplied constraints.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    evaluation_ids : list[int], optional
        A list of evaluation job id constraints.
    dataset_names : list[str], optional
        A list of dataset names to constrain by.
    model_names : list[str], optional
        A list of model names to constrain by.

    Returns
    ----------
    list[schemas.EvaluationResponse]
        A list of evaluations.
    """
    expr = _create_bulk_expression(
        evaluation_ids=evaluation_ids,
        dataset_names=dataset_names,
        model_names=model_names,
    )
    evaluations = db.query(models.Evaluation).where(*expr).all()
    return _create_responses(db, evaluations)


def get_evaluations_from_request(
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
        if evaluation := _fetch_evaluation_from_subrequest(
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
        if evaluation := _fetch_evaluation_from_subrequest(
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
    dataset_filter = schemas.Filter(**evaluation.dataset_filter)
    return core.get_disjoint_labels(db, dataset_filter, model_filter)


def delete_evaluations(
    db: Session,
    evaluation_ids: list[int] | None = None,
    dataset_names: list[str] | None = None,
    model_names: list[str] | None = None,
):
    """
    Deletes all evaluations that match the input args.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    evaluation_ids : list[int], optional
        A list of evaluation job id constraints.
    dataset_names : list[str], optional
        A list of dataset names to constrain by.
    model_names : list[str], optional
        A list of model names to constrain by.
    """
    if count_active_evaluations(
        db=db,
        evaluation_ids=evaluation_ids,
        dataset_names=dataset_names,
        model_names=model_names,
    ):
        raise exceptions.EvaluationRunningError

    evaluations = fetch_evaluations(
        db=db,
        evaluation_ids=evaluation_ids,
        dataset_names=dataset_names,
        model_names=model_names,
    )
    try:
        for evaluation in evaluations:
            db.delete(evaluation)
            db.commit()
    except IntegrityError as e:
        db.rollback()
        raise e
