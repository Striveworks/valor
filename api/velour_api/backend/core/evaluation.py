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
    """Validates that the requested dependencies exist and are valid for evaluation."""

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


def _fetch_evaluation(
    db: Session,
    model: models.Model,
    model_filter: schemas.Filter,
    evaluation_filter: schemas.Filter,
    parameters: dict | None = None,
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
                models.Evaluation.model_id == model.id,
                models.Evaluation.model_filter == model_filter.model_dump(),
                models.Evaluation.evaluation_filter
                == evaluation_filter.model_dump(),
                (
                    models.Evaluation.parameters == parameters
                    if parameters
                    else models.Evaluation.parameters.is_(None)
                ),
            )
        )
        .one_or_none()
    )
    return evaluation


def _create_responses(
    db: Session,
    evaluations: list[models.Evaluation],
) -> list[schemas.CreateEvaluationResponse]:
    results = []
    for evaluation in evaluations:
        if evaluation.id is None:
            raise exceptions.EvaluationDoesNotExistError()

        model_filter = schemas.Filter(**evaluation.model_filter)
        evaluation_filter = schemas.Filter(**evaluation.evaluation_filter)
        match evaluation.evaluation_filter["task_type"]:
            case enums.TaskType.CLASSIFICATION:
                missing_pred_keys, ignored_pred_keys = core.get_disjoint_keys(
                    db, evaluation_filter, model_filter
                )
                results.append(
                    schemas.CreateClfEvaluationResponse(
                        missing_pred_keys=missing_pred_keys,
                        ignored_pred_keys=ignored_pred_keys,
                        evaluation_id=evaluation.id,
                    )
                )
            case enums.TaskType.DETECTION:
                (
                    missing_pred_labels,
                    ignored_pred_labels,
                ) = core.get_disjoint_labels(
                    db, evaluation_filter, model_filter
                )
                results.append(
                    schemas.CreateDetectionEvaluationResponse(
                        missing_pred_labels=missing_pred_labels,
                        ignored_pred_labels=ignored_pred_labels,
                        evaluation_id=evaluation.id,
                    )
                )
            case enums.TaskType.SEGMENTATION:
                (
                    missing_pred_labels,
                    ignored_pred_labels,
                ) = core.get_disjoint_labels(
                    db, evaluation_filter, model_filter
                )
                results.append(
                    schemas.CreateSemanticSegmentationEvaluationResponse(
                        missing_pred_labels=missing_pred_labels,
                        ignored_pred_labels=ignored_pred_labels,
                        evaluation_id=evaluation.id,
                    )
                )
            case _:
                raise NotImplementedError
    return results


def create_evaluations(
    db: Session,
    job_request: schemas.EvaluationRequest,
) -> tuple[
    list[schemas.CreateEvaluationResponse],
    list[schemas.CreateEvaluationResponse],
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
    tuple[list[schemas.CreateEvaluationResponse], list[schemas.CreateEvaluationResponse]]
        A tuple of evaluation response lists following the pattern (list[created_evaluations], list[existing_evaluations])
    """

    models_to_evaluate = db.query(
        Query(models.Model).filter(job_request.model_filter).any()
    ).all()
    datasets_to_evaluate = db.query(
        Query(models.Dataset).filter(job_request.evaluation_filter).any()
    ).all()

    # verify all models and datasets are ready for evaluation
    _verify_ready_to_evaluate(
        db=db,
        dataset_list=datasets_to_evaluate,
        model_list=models_to_evaluate,
    )

    existing_evaluation_rows = []
    created_evaluation_rows = []
    for model in models_to_evaluate:
        for task_type in job_request.evaluation_filter.task_types:
            if job_request.evaluation_filter.annotation_types:
                annotation_types = [
                    annotation_type
                    for annotation_type 
                    in job_request.evaluation_filter.annotation_types
                ]

                    
                    if job_request.evaluation_filter.annotation_types
                    else []
                )
                for annotation_type in annotation_types:
                    # clean model filter
                    model_filter = job_request.model_filter.model_copy()
                    model_filter.models_names = [model.name]
                    model_filter.models_metadata = None
                    model_filter.models_geospatial = None

                    # clean evaluation filter
                    evaluation_filter = job_request.evaluation_filter.model_copy()
                    evaluation_filter.task_types = [task_type]
                    evaluation_filter.annotation_types = annotation_types

                    # dump parameters (if they exist)
                    match task_type:
                        case enums.TaskType.CLASSIFICATION:
                            parameters = None
                        case enums.TaskType.DETECTION:
                            parameters = (
                                job_request.parameters.detection.model_dump()
                            )
                        case enums.TaskType.SEGMENTATION:
                            parameters = None
                        case _:
                            raise NotImplementedError

                    # check if evaluation exists
                    if evaluation := _fetch_evaluation(
                        db=db,
                        model=model,
                        model_filter=model_filter,
                        evaluation_filter=evaluation_filter,
                        parameters=parameters,
                    ):
                        existing_evaluation_rows.append(evaluation)

                    # create evaluation row
                    else:
                        evaluation = models.Evaluation(
                            model_id=model.id,
                            model_filter=model_filter.model_dump(),
                            evaluation_filter=evaluation_filter.model_dump(),
                            parameters=parameters,
                            status=enums.EvaluationStatus.PENDING,
                        )
                        created_evaluation_rows.append(evaluation)

                

    try:
        db.add_all(created_evaluation_rows)
        db.commit()
    except IntegrityError:
        db.rollback()
        raise exceptions.EvaluationAlreadyExistsError()

    return _create_responses(db, created_evaluation_rows), _create_responses(
        existing_evaluation_rows
    )


def fetch_evaluation_from_id(
    db: Session,
    evaluation_id: int,
) -> models.Evaluation:
    """
    Fetch evaluation from database.
    """
    evaluation = db.scalar(
        select(models.Evaluation).where(models.Evaluation.id == evaluation_id)
    )
    if evaluation is None:
        raise exceptions.EvaluationDoesNotExistError
    return evaluation


def get_evaluation_ids_from_request(
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
    models_to_evaluate = db.query(
        Query(models.Model).filter(job_request.model_filter).any()
    ).all()

    evaluation_ids = []
    for model in models_to_evaluate:
        for task_type in job_request.evaluation_filter.task_types:
            for (
                annotation_type
            ) in job_request.evaluation_filter.annotation_types:

                # clean model filter
                model_filter = job_request.model_filter.model_copy()
                model_filter.models_names = [model.name]
                model_filter.models_metadata = None
                model_filter.models_geospatial = None

                # clean evaluation filter
                evaluation_filter = job_request.evaluation_filter.model_copy()
                evaluation_filter.task_types = [task_type]
                evaluation_filter.annotation_types = [annotation_type]

                match task_type:
                    case enums.TaskType.CLASSIFICATION:
                        parameters = None
                    case enums.TaskType.DETECTION:
                        parameters = (
                            job_request.parameters.detection.model_dump()
                        )
                    case enums.TaskType.SEGMENTATION:
                        parameters = None
                    case _:
                        raise NotImplementedError

                # if exists, append to existing evaluations list
                evaluation = _fetch_evaluation(
                    db=db,
                    model=model,
                    model_filter=job_request.model_filter,
                    evaluation_filter=evaluation_filter,
                    parameters=parameters,
                )
                evaluation_ids.append(evaluation.id)
    return evaluation_ids


def get_evaluations(
    db: Session,
    evaluation_ids: list[int] | None = None,
    dataset_names: list[str] | None = None,
    model_names: list[str] | None = None,
    settings: list[schemas.Filter] | None = None,
) -> list[schemas.Evaluation]:
    """
    Get evaluations that conform to input arguments.

    Parameters
    ----------
    evaluation_ids: list[int] | None
        A list of job ids to get evaluations for.
    dataset_names: list[str] | None
        A list of dataset names to get evaluations for.
    model_names: list[str] | None
        A list of model names to get evaluations for.
    settings: list[schemas.Filter] | None
        A list of evaluation settings to get evaluations for.

    Returns
    ----------
    List[schemas.Evaluation]
        A list of evaluations.
    """

    # argument expressions
    expr_evaluation_ids = (
        models.Evaluation.id.in_(evaluation_ids) if evaluation_ids else None
    )
    expr_datasets = (
        models.Dataset.name.in_(dataset_names) if dataset_names else None
    )
    expr_models = models.Model.name.in_(model_names) if model_names else None
    expr_settings = (
        models.Evaluation.settings.in_(
            [setting.model_dump() for setting in settings]
        )
        if settings
        else None
    )

    # aggregate valid expressions
    expressions = [
        expr
        for expr in [
            expr_evaluation_ids,
            expr_datasets,
            expr_models,
            expr_settings,
        ]
        if expr is not None
    ]

    # query evaluations
    evaluation_rows = db.scalars(
        select(models.Evaluation)
        .join(
            models.Dataset,
            models.Dataset.id == models.Evaluation.dataset_id,
        )
        .join(
            models.Model,
            models.Model.id == models.Evaluation.model_id,
        )
        .where(*expressions)
    ).all()

    return [
        schemas.Evaluation(
            dataset=db.scalar(
                select(models.Dataset.name).where(
                    models.Dataset.id == evaluation.dataset_id
                )
            ),
            model=db.scalar(
                select(models.Model.name).where(
                    models.Model.id == evaluation.model_id
                )
            ),
            settings=evaluation.settings,
            evaluation_id=evaluation.id,
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
            task_type=evaluation.task_type,
        )
        for evaluation in evaluation_rows
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
        raise exceptions.EvaluationDoesNotExistError


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
        expr.append(models.Dataset.name == dataset_name)
    if model_name:
        expr.append(models.Model.name == model_name)

    return db.scalar(
        select(func.count())
        .select_from(models.Evaluation)
        .join(
            models.Dataset, models.Dataset.id == models.Evaluation.dataset_id
        )
        .join(models.Model, models.Model.id == models.Evaluation.model_id)
        .where(
            or_(
                models.Evaluation.status == enums.EvaluationStatus.PENDING,
                models.Evaluation.status == enums.EvaluationStatus.RUNNING,
            ),
            *expr,
        )
    )
