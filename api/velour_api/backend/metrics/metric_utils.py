from sqlalchemy import select
from sqlalchemy.orm import Session

from velour_api import schemas
from velour_api.backend import core, models
from velour_api.enums import JobStatus


def get_or_create_row(
    db: Session,
    model_class: type,
    mapping: dict,
    columns_to_ignore: list[str] = None,
) -> any:
    """
    Tries to get the row defined by mapping. If that exists then its mapped object is returned. Otherwise a row is created by `mapping` and the newly created object is returned.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    model_class : type
        The type of model.
    mapping : dict
        The mapping to use when creating the row.
    columns_to_ignore : List[str]
        Specifies any columns to ignore in forming the WHERE expression. This can be used for numerical columns that might slightly differ but are essentially the same.

    Returns
    ----------
    any
        A model class object.
    """
    columns_to_ignore = columns_to_ignore or []

    # create the query from the mapping
    where_expressions = [
        (getattr(model_class, k) == v)
        for k, v in mapping.items()
        if k not in columns_to_ignore
    ]
    where_expression = where_expressions[0]
    for exp in where_expressions[1:]:
        where_expression = where_expression & exp

    db_element = db.scalar(select(model_class).where(where_expression))

    if not db_element:
        db_element = model_class(**mapping)
        db.add(db_element)
        db.flush()
        db.commit()

    return db_element


def create_metric_mappings(
    db: Session,
    metrics: list[
        schemas.APMetric
        | schemas.APMetricAveragedOverIOUs
        | schemas.mAPMetric
        | schemas.mAPMetricAveragedOverIOUs
    ],
    evaluation_id: int,
) -> list[dict]:
    """
    Create metric mappings from a list of metrics.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    metrics : List
        A list of metrics to create mappings for.
    evaluation_id : int
        The id of the evaluation job.

    Returns
    ----------
    List[Dict]
        A list of metric mappings.
    """
    labels = set(
        [
            (metric.label.key, metric.label.value)
            for metric in metrics
            if hasattr(metric, "label")
        ]
    )
    label_map = {
        (label[0], label[1]): core.get_label(
            db, label=schemas.Label(key=label[0], value=label[1])
        ).id
        for label in labels
    }

    ret = []
    for metric in metrics:
        if hasattr(metric, "label"):
            ret.append(
                metric.db_mapping(
                    label_id=label_map[(metric.label.key, metric.label.value)],
                    evaluation_id=evaluation_id,
                )
            )
        else:
            ret.append(metric.db_mapping(evaluation_id=evaluation_id))

    return ret


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


def get_evaluations(
    db: Session,
    job_ids: list[int] | None = None,
    dataset_names: list[str] | None = None,
    model_names: list[str] | None = None,
    settings: list[schemas.EvaluationSettings] | None = None,
) -> list[schemas.Evaluation]:
    """
    Get evaluations that conform to input arguments.

    Parameters
    ----------
    job_ids: list[int] | None
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
    expr_job_ids = models.Evaluation.id.in_(job_ids) if job_ids else None
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
        for expr in [expr_job_ids, expr_datasets, expr_models, expr_settings]
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
            job_id=evaluation.id,
            status=JobStatus.PENDING,
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
