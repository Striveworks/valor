from datetime import timezone
from typing import Sequence

from sqlalchemy import (
    ColumnElement,
    and_,
    asc,
    case,
    desc,
    func,
    nulls_last,
    or_,
    select,
    update,
)
from sqlalchemy.dialects.postgresql import aggregate_order_by
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import BinaryExpression

from valor_api import api_utils, enums, exceptions, schemas
from valor_api.backend import core, models
from valor_api.backend.query import generate_query


def _create_dataset_expr_from_list(
    dataset_names: list[str],
) -> ColumnElement[bool] | BinaryExpression[bool] | None:
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
) -> BinaryExpression[bool] | ColumnElement[bool] | None:
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


def _create_eval_expr_from_list(
    ids: list[int],
) -> BinaryExpression[bool] | ColumnElement[bool] | None:
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
        return or_(*[(models.Evaluation.id == id_) for id_ in ids])


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
        generate_query(
            models.Dataset,
            db=db,
            filter_=job_request.datum_filter,
        )
        .distinct()
        .all()
    )
    if not datasets_to_evaluate:
        raise exceptions.EvaluationRequestError(
            "No datasets meet the requirements of this request."
        )
    elif any(
        [
            dataset.status != enums.TableStatus.FINALIZED
            for dataset in datasets_to_evaluate
        ]
    ):
        unfinalized_datasets = [
            dataset.name
            for dataset in datasets_to_evaluate
            if dataset.status != enums.TableStatus.FINALIZED
        ]
        raise exceptions.EvaluationRequestError(
            f"The following datasets have not been finalized and are required for this evaluation. {unfinalized_datasets}`"
        )

    # 1.b - get all models, note this just looks up the models by name and does not apply the filter.
    models_to_evaluate = (
        db.query(models.Model)
        .where(models.Model.name.in_(job_request.model_names))
        .distinct()
        .all()
    )
    if not models_to_evaluate:
        raise exceptions.EvaluationRequestError(
            "No models meet the requirements of this request."
        )
    elif any(
        [
            core.get_model_status(
                db=db, dataset_name=dataset.name, model_name=model.name
            )
            != enums.TableStatus.FINALIZED
            for model in models_to_evaluate
            for dataset in datasets_to_evaluate
        ]
    ):
        pairings = [
            (model.name, dataset.name)
            for model in models_to_evaluate
            for dataset in datasets_to_evaluate
            if core.get_model_status(
                db=db, dataset_name=dataset.name, model_name=model.name
            )
            != enums.TableStatus.FINALIZED
        ]
        raise exceptions.EvaluationRequestError(
            f"The following model-dataset pairings have not been finalized. {pairings}"
        )

    # 2 - verify all models and datasets are ready for evaluation
    _verify_ready_to_evaluate(
        db=db,
        dataset_list=datasets_to_evaluate,
        model_list=models_to_evaluate,
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
            model_names=[model.name],
            datum_filter=job_request.datum_filter,
            parameters=job_request.parameters,
        )
        for model in models_to_evaluate
    ]


def _create_response(
    db: Session,
    evaluation: models.Evaluation,
    **kwargs,
) -> schemas.EvaluationResponse:
    """Converts a evaluation row into a response schema."""
    metrics = db.query(
        select(models.Metric)
        .where(
            and_(
                models.Metric.evaluation_id == evaluation.id,
                models.Metric.type.in_(
                    evaluation.parameters["metrics_to_return"]
                ),
            )
        )
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
        status=enums.EvaluationStatus(evaluation.status),
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
        created_at=evaluation.created_at.replace(tzinfo=timezone.utc),
        meta=evaluation.meta,
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
                kwargs = {}
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


def _validate_create_or_get_evaluations(
    db: Session,
    job_request: schemas.EvaluationRequest,
    evaluation: models.Evaluation,
) -> models.Evaluation:
    """
    Validates whether a new or failed should proceed to a computation.

    Parameters
    ----------
    db : Session
        The database session.
    job_request : schemas.EvaluationRequest
        The evaluations to create.
    evaluation : models.Evaluation
        The evaluation row to validate.

    Returns
    -------
    model.Evaluation
        The row that was passed as input.
    """

    # unpack filters and params
    groundtruth_filter = job_request.datum_filter
    prediction_filter = groundtruth_filter.model_copy()
    prediction_filter.model_names = [evaluation.model_name]
    parameters = job_request.parameters

    datasets = (
        generate_query(
            models.Dataset,
            db=db,
            filter_=groundtruth_filter,
            label_source=models.GroundTruth,
        )
        .distinct()
        .all()
    )

    model = (
        generate_query(
            models.Model,
            db=db,
            filter_=prediction_filter,
            label_source=models.Prediction,
        )
        .distinct()
        .one_or_none()
    )

    # verify model and datasets have data for this evaluation
    if not datasets:
        raise exceptions.EvaluationRequestError(
            "No finalized datasets were found that met the filter criteria."
        )
    elif model is None:
        raise exceptions.EvaluationRequestError(
            f"The model '{evaluation.model_name}' did not meet the filter criteria."
        )

    # check that prediction label keys match ground truth label keys
    if job_request.parameters.task_type == enums.TaskType.CLASSIFICATION:
        core.validate_matching_label_keys(
            db=db,
            label_map=parameters.label_map,
            groundtruth_filter=groundtruth_filter,
            prediction_filter=prediction_filter,
        )

    return evaluation


def create_or_get_evaluations(
    db: Session,
    job_request: schemas.EvaluationRequest,
    allow_retries: bool = False,
) -> list[schemas.EvaluationResponse]:
    """
    Creates evaluations from evaluation request.

    If an evaluation already exists, it will be returned with its existing status.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    job_request : schemas.EvaluationRequest
        The evaluations to create.

    Returns
    -------
    list[schemas.EvaluationResponse]
        A list of evaluation responses.
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
            if (
                allow_retries
                and evaluation.status == enums.EvaluationStatus.FAILED
            ):
                evaluation = _validate_create_or_get_evaluations(
                    db=db,
                    job_request=subrequest,
                    evaluation=evaluation,
                )
                try:
                    evaluation.status = enums.EvaluationStatus.PENDING
                    db.commit()
                except IntegrityError:
                    db.rollback()
                    raise exceptions.EvaluationStateError(
                        evaluation_id=evaluation.id,
                        current_state=enums.EvaluationStatus.FAILED,
                        requested_state=enums.EvaluationStatus.PENDING,
                    )

            existing_rows.append(evaluation)

        # create evaluation row
        else:
            evaluation = models.Evaluation(
                model_name=subrequest.model_names[0],
                datum_filter=subrequest.datum_filter.model_dump(),
                parameters=subrequest.parameters.model_dump(),
                status=enums.EvaluationStatus.PENDING,
                meta={},
            )
            evaluation = _validate_create_or_get_evaluations(
                db=db,
                job_request=subrequest,
                evaluation=evaluation,
            )
            created_rows.append(evaluation)

    try:
        db.add_all(created_rows)
        db.commit()
    except IntegrityError:
        db.rollback()
        raise exceptions.EvaluationAlreadyExistsError()

    return _create_responses(db, created_rows + existing_rows)


def _fetch_evaluations_and_mark_for_deletion(
    db: Session,
    evaluation_ids: list[int] | None = None,
    dataset_names: list[str] | None = None,
    model_names: list[str] | None = None,
) -> Sequence[models.Evaluation]:
    """
    Gets all evaluations that conform to user-supplied constraints and that are not already marked
    for deletion. Then marks them for deletion and returns them.

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

    stmt = (
        update(models.Evaluation)
        .returning(models.Evaluation)
        .where(
            and_(
                *expr,
                models.Evaluation.status != enums.EvaluationStatus.DELETING,
            )
        )
        .values(status=enums.EvaluationStatus.DELETING)
        .execution_options(synchronize_session="fetch")
    )

    return db.execute(stmt).scalars().all()


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
        The ID of the evaluation.

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
        .where(
            and_(
                models.Evaluation.id == evaluation_id,
                models.Evaluation.status != enums.EvaluationStatus.DELETING,
            )
        )
        .one_or_none()
    )
    if evaluation is None:
        raise exceptions.EvaluationDoesNotExistError
    return evaluation


def get_paginated_evaluations(
    db: Session,
    evaluation_ids: list[int] | None = None,
    dataset_names: list[str] | None = None,
    model_names: list[str] | None = None,
    offset: int = 0,
    limit: int = -1,
    metrics_to_sort_by: dict[str, dict[str, str] | str] | None = None,
) -> tuple[list[schemas.EvaluationResponse], dict[str, str]]:
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
    offset : int, optional
        The start index of the items to return.
    limit : int, optional
        The number of items to return. Returns all items when set to -1.
    metrics_to_sort_by: dict[str, dict[str, str] | str], optional
        An optional dict of metric types to sort the evaluations by.

    Returns
    ----------
    tuple[list[schemas.EvaluationResponse], dict[str, str]]
        A tuple containing the evaluations and response headers to return to the user.
    """
    if offset < 0 or limit < -1:
        raise ValueError(
            "Offset should be an int greater than or equal to zero. Limit should be an int greater than or equal to -1."
        )

    expr = _create_bulk_expression(
        evaluation_ids=evaluation_ids,
        dataset_names=dataset_names,
        model_names=model_names,
    )

    count = (
        db.query(func.count(models.Evaluation.id))
        .where(
            and_(
                *expr,
                models.Evaluation.status != enums.EvaluationStatus.DELETING,
            )
        )
        .scalar()
    )

    if offset > count:
        raise ValueError(
            "Offset is greater than the total number of items returned in the query."
        )

    # return all rows when limit is -1
    if limit == -1:
        limit = count

    if metrics_to_sort_by is not None:
        conditions = []
        order_case = []

        for i, (metric_type, label) in enumerate(metrics_to_sort_by.items()):
            # if the value represents a label_key
            if isinstance(label, str):
                order_case.append(
                    (
                        and_(
                            models.Metric.type == metric_type,
                            models.Metric.parameters["label_key"].astext
                            == label,
                        ),
                        i + 1,
                    ),
                )
                conditions.append(
                    and_(
                        models.Metric.type == metric_type,
                        models.Metric.parameters["label_key"].astext == label,
                    )
                )
            # if the value represents a label
            else:
                order_case.append(
                    (
                        and_(
                            models.Metric.type == metric_type,
                            models.Label.key == label["key"],
                            models.Label.value == label["value"],
                        ),
                        i + 1,
                    ),
                )
                conditions.append(
                    and_(
                        models.Metric.type == metric_type,
                        models.Label.key == label["key"],
                        models.Label.value == label["value"],
                    )
                )

        aggregated_sorting_field = (
            select(
                models.Metric.evaluation_id,
                func.array_agg(
                    aggregate_order_by(
                        models.Metric.value, case(*order_case, else_=0)
                    )
                ).label("sort_array"),
            )
            .select_from(models.Metric)
            .group_by(models.Metric.evaluation_id)
            .filter(or_(*conditions))
            .alias()
        )

        evaluations = db.query(
            select(
                models.Evaluation.parameters["task_type"],
                aggregated_sorting_field.c.sort_array,
                models.Evaluation,
            )
            .select_from(models.Evaluation)
            .join(
                aggregated_sorting_field,
                aggregated_sorting_field.c.evaluation_id
                == models.Evaluation.id,
                isouter=True,
            )
            .where(
                and_(
                    *expr,
                    models.Evaluation.status
                    != enums.EvaluationStatus.DELETING,
                )
            )
            .order_by(
                asc(models.Evaluation.parameters["task_type"]),
                nulls_last(aggregated_sorting_field.c.sort_array.desc()),
                desc(models.Evaluation.created_at),
            )
            .offset(offset)
            .limit(limit)
            .alias()
        ).all()

    else:
        evaluations = (
            db.query(
                models.Evaluation,
            )
            .where(
                and_(
                    *expr,
                    models.Evaluation.status
                    != enums.EvaluationStatus.DELETING,
                )
            )
            .order_by(
                asc(models.Evaluation.parameters["task_type"]),
                desc(models.Evaluation.created_at),
            )
            .offset(offset)
            .limit(limit)
            .all()
        )

    content = _create_responses(db, evaluations)

    headers = api_utils._get_pagination_header(
        offset=offset,
        number_of_returned_items=len(evaluations),
        total_number_of_items=count,
    )

    return (content, headers)


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
        .where(
            and_(
                models.Evaluation.model_name == model_name,
                models.Evaluation.status != enums.EvaluationStatus.DELETING,
            )
        )
        .all()
    )
    return [
        schemas.EvaluationResponse(
            id=eval_.id,
            model_name=model_name,
            datum_filter=eval_.datum_filter,
            parameters=eval_.parameters,
            status=enums.EvaluationStatus(eval_.status),
            created_at=eval_.created_at.replace(tzinfo=timezone.utc),
            meta=eval_.meta,
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
        The ID of the evaluation.

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
        The ID of the evaluation.
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
    retval = db.scalar(
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

    if retval is None:
        raise RuntimeError("psql didn't return any active evaluations.")

    return retval


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

    evaluations = _fetch_evaluations_and_mark_for_deletion(
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
