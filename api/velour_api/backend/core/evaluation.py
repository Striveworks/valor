from sqlalchemy import and_, func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import BinaryExpression

from velour_api import enums, exceptions, schemas
from velour_api.backend import core, models
from velour_api.backend.ops import Query


def _create_dataset_expr_from_list(
    dataset_names: list[str],
) -> BinaryExpression:
    """
    Creates a sqlalchemy or_ expression from a list of str.

    Note that this is for accessing models.Evaluation with a list of dataset names.

    Parameters
    ----------
    dataset_names : list[str]
        List of dataset names.

    Returns
    -------
    BinaryExpression
        The sqlalchemy expression.
    """
    if not dataset_names:
        return None
    elif len(dataset_names) == 1:
        return models.Evaluation.datum_filter["dataset_names"].op("?")(
            dataset_names[0]
        )
    else:
        return or_(
            *[
                models.Evaluation.datum_filter["dataset_names"].op("?")(name)
                for name in dataset_names
            ]
        )


def _create_model_expr_from_list(
    model_names: list[str],
) -> BinaryExpression:
    """
    Creates a sqlalchemy or_ expression from a list of str.

    Note that this is for accessing models.Evaluation with a list of model names.

    Parameters
    ----------
    model_names : list[str]
        List of model names.

    Returns
    -------
    BinaryExpression
        The sqlalchemy expression.
    """
    if not model_names:
        return None
    elif len(model_names) == 1:
        return models.Evaluation.model_name == model_names[0]
    else:
        return or_(
            *[models.Evaluation.model_name == name for name in model_names]
        )


def _create_eval_expr_from_list(ids: list[int]) -> BinaryExpression:
    """
    Creates a sqlalchemy or_ expression from a list of int.

    Note that this is for accessing models.Evaluation with a list of ids.

    Parameters
    ----------
    ids : list[int]
        List of evaluations ids.

    Returns
    -------
    BinaryExpression
        The sqlalchemy expression.
    """
    if not ids:
        return None
    elif len(ids) == 1:
        return models.Evaluation.id == ids[0]
    else:
        return or_(*[models.Evaluation.id == id_ for id_ in ids])


def _create_bulk_expression(
    evaluation_ids: list[int] | None = None,
    dataset_names: list[str] | None = None,
    model_names: list[str] | None = None,
) -> list[BinaryExpression]:
    """Creates an expression used to query evaluations by id, dataset and model."""
    expr = []
    if dataset_names:
        expr.append(_create_dataset_expr_from_list(dataset_names))
    if model_names:
        expr.append(_create_model_expr_from_list(model_names))
    if evaluation_ids:
        expr.append(_create_eval_expr_from_list(evaluation_ids))
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
    if not dataset_list:
        raise RuntimeError("Received an empty list of datasets to verify.")
    elif not model_list:
        raise RuntimeError("Received an empty list of models to verify.")

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
                    raise NotImplementedError(
                        f"A case for `{dataset.status}` has not been implemented."
                    )

            # verify model status
            model_status = core.get_model_status(
                db=db,
                dataset_name=dataset.name,
                model_name=model.name,
            )
            match model_status:
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
                    raise NotImplementedError(
                        f"A case for `{model_status}` has not been implemented."
                    )


def _split_request(
    db: Session,
    job_request: schemas.EvaluationRequest,
) -> list[schemas.EvaluationRequest]:
    """
    Splits a job request into component requests.

    1. Fetch all datasets and models that conform to their respective filters.
    2. Verify all datasets and models are ready to be evaluated.
    3. For auditability, replace ambiguous dataset queries with explicit lists of names.
    4. Create a response per model..
    """

    # 1.a - get all datasets, note this uses the unmodified filter
    datasets_to_evaluate = (
        db.query(Query(models.Dataset).filter(job_request.datum_filter).any())
        .distinct()
        .all()
    )
    if not datasets_to_evaluate:
        raise exceptions.EvaluationRequestError(
            "No datasets meet the requirements of this request."
        )

    # 1.b - get all models, note this uses the unmodified filter
    model_filter = schemas.Filter(model_names=job_request.model_names)
    model_to_evaluate = (
        db.query(Query(models.Model).filter(model_filter).any())
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

    # 3 - create explicit filter with dataset names
    job_request.datum_filter.dataset_names = [
        dataset.name for dataset in datasets_to_evaluate
    ]
    job_request.datum_filter.dataset_metadata = None
    job_request.datum_filter.model_names = None
    job_request.datum_filter.model_metadata = None

    # 4. - create requests
    return [
        schemas.EvaluationRequest(
            model_names=model.name,
            datum_filter=job_request.datum_filter,
            parameters=job_request.parameters,
        )
        for model in model_to_evaluate
    ]


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
        model_name=evaluation.model_name,
        datum_filter=evaluation.datum_filter,
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

        datum_filter = schemas.Filter(**evaluation.datum_filter)
        model_filter = datum_filter.model_copy()
        model_filter.dataset_names = None
        model_filter.model_names = [evaluation.model_name]
        parameters = schemas.EvaluationParameters(**evaluation.parameters)

        match parameters.task_type:
            case enums.TaskType.CLASSIFICATION:
                missing_pred_keys, ignored_pred_keys = core.get_disjoint_keys(
                    db,
                    datum_filter,
                    model_filter,
                    label_map=parameters.label_map,
                )
                kwargs = {
                    "missing_pred_keys": missing_pred_keys,
                    "ignored_pred_keys": ignored_pred_keys,
                }
            case (
                enums.TaskType.OBJECT_DETECTION
                | enums.TaskType.SEMANTIC_SEGMENTATION
            ):
                (
                    missing_pred_labels,
                    ignored_pred_labels,
                ) = core.get_disjoint_labels(
                    db,
                    datum_filter,
                    model_filter,
                    label_map=parameters.label_map,
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


def _fetch_evaluation_from_subrequest(
    db: Session,
    subrequest: schemas.EvaluationRequest,
) -> models.Evaluation:
    """
    Fetch the row for an evaluation that matches the provided `EvaluationRequest` attributes.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    subrequest : schemas.EvaluationRequest
        Evaluation subrequest. Should only have one model name defined.

    Returns
    -------
    models.Evaluation
        The evaluation row.

    Raises
    ------
    RuntimeError
        If subrequest defines no model names or more than one.
    """
    if len(subrequest.model_names) != 1:
        raise RuntimeError(
            "Subrequests should only reference a single model name."
        )

    evaluation = (
        db.query(models.Evaluation)
        .where(
            and_(
                models.Evaluation.model_name == subrequest.model_names[0],
                models.Evaluation.datum_filter
                == subrequest.datum_filter.model_dump(),
                models.Evaluation.parameters
                == subrequest.parameters.model_dump(),
            )
        )
        .one_or_none()
    )
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
        if len(subrequest.model_names) != 1:
            raise RuntimeError(
                "Subrequests should only reference a single model name."
            )

        # check if evaluation exists
        if evaluation := _fetch_evaluation_from_subrequest(
            db=db,
            subrequest=subrequest,
        ):
            existing_rows.append(evaluation)

        # create evaluation row
        else:
            evaluation = models.Evaluation(
                model_name=subrequest.model_names[0],
                datum_filter=subrequest.datum_filter.model_dump(),
                parameters=subrequest.parameters.model_dump(),
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


def get_evaluation_requests_from_model(
    db: Session, model_name: str
) -> list[schemas.EvaluationResponse]:
    """
    Returns all evaluation settings for a given model.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    model_name : str
        The model name to find evaluations of

    Returns
    ----------
    list[schemas.EvaluationResponse]
        A list of evaluations.
    """
    evaluations = (
        db.query(models.Evaluation)
        .where(models.Evaluation.model_name == model_name)
        .all()
    )
    return [
        schemas.EvaluationResponse(
            id=eval_.id,
            model_name=model_name,
            datum_filter=eval_.datum_filter,
            parameters=eval_.parameters,
            status=eval_.status,
        )
        for eval_ in evaluations
    ]


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
