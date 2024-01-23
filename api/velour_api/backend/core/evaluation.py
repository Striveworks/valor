from sqlalchemy import and_, func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from velour_api import enums, exceptions, schemas
from velour_api.backend import core, models


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


def _validate_filters(job_request: schemas.EvaluationJob):
    """Validates that the filter object is properly configured for evaluation."""
    if (
        job_request.settings.filters.dataset_names is not None
        or job_request.settings.filters.dataset_metadata is not None
        or job_request.settings.filters.dataset_geospatial is not None
        or job_request.settings.filters.models_names is not None
        or job_request.settings.filters.models_metadata is not None
        or job_request.settings.filters.models_geospatial is not None
        or job_request.settings.filters.prediction_scores is not None
        or job_request.settings.filters.task_types is not None
    ):
        raise ValueError(
            "Evaluation filter objects should not include any dataset, model, prediction score or task type filters."
        )
    elif (
        job_request.task_type == enums.TaskType.SEGMENTATION
        and job_request.settings.filters.annotation_types is not None
    ):
        raise ValueError(
            "Segmentation evaluation should not include any annotation type filters."
        )


def _validate_request(
    db: Session,
    job_request: schemas.EvaluationJob,
    dataset: models.Dataset,
    model: models.Model,
):
    """Validates that the requested dependencies exist and are valid for evaluation."""

    # validate type
    if not isinstance(job_request, schemas.EvaluationJob):
        raise TypeError(
            f"Expected `schemas.EvaluationJob`, received `{type(job_request)}`"
        )
    if job_request.dataset != dataset.name or job_request.model != model.name:
        raise ValueError("Name mismatch.")

    # get statuses
    dataset_status = enums.TableStatus(dataset.status)
    model_status = core.get_model_status(
        db=db, dataset_name=dataset.name, model_name=model.name
    )

    # verify dataset status
    match dataset_status:
        case enums.TableStatus.CREATING:
            raise exceptions.DatasetNotFinalizedError(job_request.dataset)
        case enums.TableStatus.DELETING | None:
            raise exceptions.DatasetDoesNotExistError(job_request.dataset)
        case enums.TableStatus.FINALIZED:
            pass
        case _:
            raise RuntimeError

    # verify model status
    match model_status:
        case enums.TableStatus.CREATING:
            raise exceptions.ModelNotFinalizedError(
                dataset_name=job_request.dataset, model_name=job_request.model
            )
        case enums.TableStatus.DELETING | None:
            raise exceptions.ModelDoesNotExistError(job_request.model)
        case enums.TableStatus.FINALIZED:
            pass
        case _:
            raise RuntimeError

    # validate parameters
    match job_request.task_type:
        case enums.TaskType.DETECTION:
            if not job_request.settings.parameters:
                job_request.settings.parameters = schemas.DetectionParameters()
        case _:
            if job_request.settings.parameters:
                raise ValueError(
                    f"Evaluations with task type `{job_request.task_type}` do not take parametric input."
                )

    # validate filters
    if job_request.settings.filters:
        _validate_filters(job_request=job_request)


def create_evaluation(
    db: Session,
    job_request: schemas.EvaluationJob,
) -> int:
    """
    Creates an evaluation.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    job_request : schemas.EvaluationJob
        The evaluation job to create.

    Returns
    -------
    int
        The id of the new evaluation.
    """
    if fetch_evaluation_from_job_request(db, job_request) is not None:
        raise exceptions.EvaluationAlreadyExistsError()

    dataset = core.fetch_dataset(db, job_request.dataset)
    model = core.fetch_model(db, job_request.model)

    _validate_request(
        db=db, job_request=job_request, dataset=dataset, model=model
    )

    try:
        evaluation = models.Evaluation(
            dataset_id=dataset.id,
            model_id=model.id,
            task_type=job_request.task_type,
            settings=job_request.settings.model_dump(),
            status=enums.EvaluationStatus.PENDING,
        )
        db.add(evaluation)
        db.commit()
        return evaluation.id
    except IntegrityError:
        db.rollback()
        raise exceptions.EvaluationAlreadyExistsError()


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


def fetch_evaluation_from_job_request(
    db: Session,
    job_request: schemas.EvaluationJob,
):
    """
    Get the row model for an evaluation that matches the provided `EvaluationJob`.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    job_request : schemas.EvaluationJob
        The evaluation job to create.

    Returns
    -------
    models.Evaluation
        The evaluation row.
    """
    dataset = core.fetch_dataset(db, job_request.dataset)
    model = core.fetch_model(db, job_request.model)

    _validate_request(
        db=db, job_request=job_request, dataset=dataset, model=model
    )

    evaluation = (
        db.query(models.Evaluation)
        .where(
            and_(
                models.Evaluation.dataset_id == dataset.id,
                models.Evaluation.model_id == model.id,
                models.Evaluation.task_type == job_request.task_type,
                models.Evaluation.settings
                == job_request.settings.model_dump(),
            )
        )
        .one_or_none()
    )
    return evaluation


def get_evaluation_id_from_job_request(
    db: Session,
    job_request: schemas.EvaluationJob,
) -> int | None:
    """
    Get the id for an evaluation that matches the provided `EvaluationJob`.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    job_request : schemas.EvaluationJob
        The evaluation job to create.

    Returns
    -------
    int | None
        The id of the matching evaluation. Returns None if one does not exist.
    """
    evaluation = fetch_evaluation_from_job_request(db, job_request)
    return evaluation.id if evaluation else None


def get_evaluations(
    db: Session,
    evaluation_ids: list[int] | None = None,
    dataset_names: list[str] | None = None,
    model_names: list[str] | None = None,
    settings: list[schemas.EvaluationSettings] | None = None,
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
    settings: list[schemas.EvaluationSettings] | None
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


def _get_annotation_types_for_computation(
    db: Session,
    dataset: models.Dataset,
    model: models.Model,
    job_filter: schemas.Filter | None = None,
) -> enums.AnnotationType:
    """Fetch the groundtruth and prediction annotation types for a given dataset / model combination."""
    # get dominant type
    groundtruth_type = core.get_annotation_type(db, dataset, None)
    prediction_type = core.get_annotation_type(db, dataset, model)
    greatest_common_type = (
        groundtruth_type
        if groundtruth_type < prediction_type
        else prediction_type
    )
    if job_filter.annotation_types:
        if greatest_common_type not in job_filter.annotation_types:
            sorted_types = sorted(
                job_filter.annotation_types,
                key=lambda x: x,
                reverse=True,
            )
            for annotation_type in sorted_types:
                if greatest_common_type >= annotation_type:
                    return annotation_type, annotation_type
            raise RuntimeError(
                f"Annotation type filter is too restrictive. Attempted filter `{greatest_common_type}` over `{groundtruth_type, prediction_type}`."
            )
    return groundtruth_type, prediction_type


def get_disjoint_labels_from_evaluation(
    db: Session,
    job_request: schemas.EvaluationJob,
) -> tuple[list[schemas.Label], list[schemas.Label]]:
    """
    Return a tuple containing the unique labels associated with the groundtruths and predictions stored in a database.

    Parameters
    ----------
    db : Session
        The database session.
    job_request : schemas.EvaluationJob

    Returns
    -------
    tuple[list[schemas.Label], list[schemas.Label]]
        A tuple of the disjoint label sets. The tuple follows the form (GroundTruth, Prediction).
    """

    # load sql objects
    dataset = core.fetch_dataset(db, job_request.dataset)
    model = core.fetch_model(db, job_request.model)

    # get filter object
    if not job_request.settings.filters:
        filters = schemas.Filter()
    else:
        _validate_request(
            db=db,
            job_request=job_request,
            dataset=dataset,
            model=model,
        )
        filters = job_request.settings.filters.model_copy()

    # determine annotation types
    (
        groundtruth_type,
        prediction_type,
    ) = _get_annotation_types_for_computation(db, dataset, model, filters)

    # create groundtruth label filter
    groundtruth_label_filter = filters.model_copy()
    groundtruth_label_filter.dataset_names = [job_request.dataset]
    groundtruth_label_filter.annotation_types = [groundtruth_type]

    # create prediction label filter
    prediction_label_filter = filters.model_copy()
    prediction_label_filter.dataset_names = [job_request.dataset]
    prediction_label_filter.models_names = [model.name]
    prediction_label_filter.annotation_types = [prediction_type]

    groundtruth_labels = core.get_labels(
        db, groundtruth_label_filter, ignore_predictions=True
    )
    prediction_labels = core.get_labels(
        db, prediction_label_filter, ignore_groundtruths=True
    )

    # don't count user-mapped labels as disjoint
    mapped_labels = set()
    if job_request.settings.label_map:
        for map_from, map_to in job_request.settings.label_map:
            mapped_labels.add(
                schemas.Label(key=map_from[0], value=map_from[1])
            )
            mapped_labels.add(schemas.Label(key=map_to[0], value=map_to[1]))

    groundtruth_unique = list(
        groundtruth_labels - prediction_labels - mapped_labels
    )
    prediction_unique = list(
        prediction_labels - groundtruth_labels - mapped_labels
    )

    return groundtruth_unique, prediction_unique


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
