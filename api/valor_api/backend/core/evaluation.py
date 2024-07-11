import warnings
from datetime import timezone

from pydantic import ValidationError
from sqlalchemy import (
    ColumnElement,
    and_,
    asc,
    case,
    delete,
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
from valor_api.backend.metrics.metric_utils import (
    prepare_filter_for_evaluation,
)
from valor_api.backend.query import generate_query
from valor_api.schemas import migrations


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
        return models.Evaluation.dataset_names.op("?")(dataset_names[0])
    else:
        return or_(
            *[
                models.Evaluation.dataset_names.op("?")(name)
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


def validate_request(
    db: Session,
    job_request: schemas.EvaluationRequest,
):
    """
    Gets and validates that all datasets and models are ready for evaluation.

    Parameters
    ----------
    db : Session
        The database session.
    job_request : EvaluationRequest
        The evaluation request to validate.

    Raises
    ------
    EvaluationRequestError
        If any of the datasets or models are in an illegal state.
    """
    if not job_request.dataset_names:
        raise exceptions.EvaluationRequestError(
            msg="At least one dataset is required to start an evaluation."
        )
    if not job_request.model_names:
        raise exceptions.EvaluationRequestError(
            msg="At least one model is required to start an evaluation."
        )

    errors = []
    for dataset_name in job_request.dataset_names:

        # verify dataset status
        try:
            dataset_status = core.get_dataset_status(db=db, name=dataset_name)
        except exceptions.DatasetDoesNotExistError as e:
            errors.append(e)
            continue

        match enums.TableStatus(dataset_status):
            case enums.TableStatus.CREATING:
                errors.append(
                    exceptions.DatasetNotFinalizedError(dataset_name)
                )
            case enums.TableStatus.DELETING | None:
                errors.append(
                    exceptions.DatasetDoesNotExistError(dataset_name)
                )
            case enums.TableStatus.FINALIZED:
                pass
            case _:
                raise NotImplementedError(
                    f"A case for `{dataset_status}` is not a supported status."
                )

        for model_name in job_request.model_names:

            # verify model status
            try:
                model_status = core.get_model_status(
                    db=db,
                    dataset_name=dataset_name,
                    model_name=model_name,
                )
            except exceptions.ModelDoesNotExistError as e:
                errors.append(e)
                continue

            match model_status:
                case enums.TableStatus.CREATING:
                    errors.append(
                        exceptions.ModelNotFinalizedError(
                            dataset_name=dataset_name,
                            model_name=model_name,
                        )
                    )
                case enums.TableStatus.DELETING | None:
                    errors.append(
                        exceptions.ModelDoesNotExistError(model_name)
                    )
                case enums.TableStatus.FINALIZED:
                    pass
                case _:
                    raise NotImplementedError(
                        f"A case for `{model_status}` has not been implemented."
                    )

    if errors:
        raise exceptions.EvaluationRequestError(
            msg="Failed request validation.", errors=errors
        )


def _validate_evaluation_filter(
    db: Session,
    evaluation: models.Evaluation,
):
    """
    Validates whether a new evaluation should proceed to a computation.

    Parameters
    ----------
    db : Session
        The database session.
    evaluation : models.Evaluation
        The evaluation row to validate.
    """

    # unpack filters and params
    filters = schemas.Filter(**evaluation.filters)
    parameters = schemas.EvaluationParameters(**evaluation.parameters)

    # generate filters
    _, groundtruth_filter, prediction_filter = prepare_filter_for_evaluation(
        db=db,
        filters=filters,
        dataset_names=evaluation.dataset_names,
        model_name=evaluation.model_name,
        task_type=parameters.task_type,
        label_map=parameters.label_map,
    )

    if parameters.task_type == enums.TaskType.TEXT_GENERATION:
        datasets = (
            generate_query(
                models.Dataset.name,
                db=db,
                filters=groundtruth_filter,
                label_source=models.Prediction,
            )
            .distinct()
            .all()
        )
    else:
        datasets = (
            generate_query(
                models.Dataset.name,
                db=db,
                filters=groundtruth_filter,
                label_source=models.GroundTruth,
            )
            .distinct()
            .all()
        )

    # verify datasets have data for this evaluation
    if not datasets:
        raise exceptions.EvaluationRequestError(
            msg="No datasets were found that met the filter criteria."
        )

    # check that prediction label keys match ground truth label keys
    if parameters.task_type == enums.TaskType.CLASSIFICATION:
        core.validate_matching_label_keys(
            db=db,
            groundtruth_filter=groundtruth_filter,
            prediction_filter=prediction_filter,
            label_map=parameters.label_map,
        )


def _create_response(
    db: Session,
    evaluation: models.Evaluation,
    **kwargs,
) -> schemas.EvaluationResponse:
    """Converts a evaluation row into a response schema."""

    metrics = [
        schemas.Metric(
            type=mtype,
            value=mvalue,
            label=(
                schemas.Label(key=lkey, value=lvalue)
                if lkey and lvalue
                else None
            ),
            parameters=mparam,
        )
        for mtype, mvalue, mparam, lkey, lvalue in (
            db.query(
                models.Metric.type,
                models.Metric.value,
                models.Metric.parameters,
                models.Label.key,
                models.Label.value,
            )
            .select_from(models.Metric)
            .join(
                models.Label,
                models.Label.id == models.Metric.label_id,
                isouter=True,
            )
            .where(
                and_(
                    models.Metric.evaluation_id == evaluation.id,
                    models.Metric.type.in_(
                        evaluation.parameters["metrics_to_return"]
                    ),
                )
            )
            .all()
        )
    ]

    confusion_matrices = [
        schemas.ConfusionMatrixResponse(
            label_key=matrix.label_key,
            entries=[
                schemas.ConfusionMatrixEntry(**entry) for entry in matrix.value
            ],
        )
        for matrix in (
            db.query(models.ConfusionMatrix)
            .where(models.ConfusionMatrix.evaluation_id == evaluation.id)
            .all()
        )
    ]

    return schemas.EvaluationResponse(
        id=evaluation.id,
        dataset_names=evaluation.dataset_names,
        model_name=evaluation.model_name,
        filters=evaluation.filters,
        parameters=evaluation.parameters,
        status=enums.EvaluationStatus(evaluation.status),
        metrics=metrics,
        confusion_matrices=confusion_matrices,
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

        parameters = schemas.EvaluationParameters(**evaluation.parameters)
        kwargs = dict()
        try:
            filters = schemas.Filter(**evaluation.filters)

            groundtruth_filter = filters.model_copy()
            groundtruth_filter.predictions = None

            prediction_filter = filters.model_copy()
            prediction_filter.groundtruths = None

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
                        groundtruth_filter,
                        prediction_filter,
                        label_map=parameters.label_map,
                    )
                    kwargs = {
                        "missing_pred_labels": missing_pred_labels,
                        "ignored_pred_labels": ignored_pred_labels,
                    }
                case enums.TaskType.TEXT_GENERATION:
                    kwargs = {}
                case _:
                    raise NotImplementedError
        except ValidationError as e:
            try:
                migrations.DeprecatedFilter(**evaluation.filters)
                warnings.warn(
                    "Evaluation response is using a deprecated filter format.",
                    DeprecationWarning,
                )
            except ValidationError:
                raise e

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
                models.Evaluation.dataset_names == subrequest.dataset_names,
                models.Evaluation.model_name == subrequest.model_names[0],
                models.Evaluation.filters == subrequest.filters.model_dump(),
                models.Evaluation.parameters
                == subrequest.parameters.model_dump(),
            )
        )
        .one_or_none()
    )
    return evaluation


def _split_request(
    job_request: schemas.EvaluationRequest,
) -> list[schemas.EvaluationRequest]:
    """
    Splits a job request into component requests by model.

    Parameters
    ----------
    job_request : EvaluationRequest
        The job request to split (if multiple model names exist).
    """

    return [
        schemas.EvaluationRequest(
            dataset_names=job_request.dataset_names,
            model_names=[model_name],
            filters=job_request.filters,
            parameters=job_request.parameters,
        )
        for model_name in job_request.model_names
    ]


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

    # verify that all datasets and models are ready to be evaluated
    validate_request(db=db, job_request=job_request)

    created_rows = []
    existing_rows = []
    for subrequest in _split_request(job_request):
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
                dataset_names=subrequest.dataset_names,
                model_name=subrequest.model_names[0],
                filters=subrequest.filters.model_dump(),
                parameters=subrequest.parameters.model_dump(),
                status=enums.EvaluationStatus.PENDING,
                meta=dict(),
            )
            _validate_evaluation_filter(
                db=db,
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
            dataset_names=eval_.dataset_names,
            model_name=model_name,
            filters=eval_.filters,
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

    # verify no active evaluations
    if count_active_evaluations(
        db=db,
        evaluation_ids=evaluation_ids,
        dataset_names=dataset_names,
        model_names=model_names,
    ):
        raise exceptions.EvaluationRunningError

    expr = _create_bulk_expression(
        evaluation_ids=evaluation_ids,
        dataset_names=dataset_names,
        model_names=model_names,
    )

    # mark evaluations for deletion
    mark_for_deletion = (
        update(models.Evaluation)
        .returning(models.Evaluation.id)
        .where(
            and_(
                *expr,
                models.Evaluation.status != enums.EvaluationStatus.DELETING,
            )
        )
        .values(status=enums.EvaluationStatus.DELETING)
        .execution_options(synchronize_session="fetch")
    )
    try:
        marked_evaluation_ids = db.execute(mark_for_deletion).scalars().all()
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise e

    # delete metrics
    try:
        db.execute(
            delete(models.Metric).where(
                models.Metric.evaluation_id.in_(marked_evaluation_ids)
            )
        )
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise e

    # delete confusion matrices
    try:
        db.execute(
            delete(models.ConfusionMatrix).where(
                models.ConfusionMatrix.evaluation_id.in_(marked_evaluation_ids)
            )
        )
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise e

    # delete evaluations
    try:
        db.execute(
            delete(models.Evaluation).where(
                models.Evaluation.id.in_(marked_evaluation_ids)
            )
        )
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise e


def delete_evaluation_from_id(db: Session, evaluation_id: int):
    """
    Delete a evaluation by id.

    Parameters
    ----------
    db : Session
        The database session.
    evaluation_id : int
        The evaluation identifer.

    Raises
    ------
    EvaluationRunningError
        If the evaluation is currently running.
    EvaluationDoesNotExistError
        If the evaluation does not exist.
    """
    evaluation = fetch_evaluation_from_id(db=db, evaluation_id=evaluation_id)
    if evaluation.status in {
        enums.EvaluationStatus.PENDING,
        enums.EvaluationStatus.RUNNING,
    }:
        raise exceptions.EvaluationRunningError
    elif evaluation.status in enums.EvaluationStatus.DELETING:
        return

    try:
        db.delete(evaluation)
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise e
