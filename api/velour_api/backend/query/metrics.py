import json

from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from velour_api import schemas
from velour_api.backend import core, jobs, models


def _db_metric_to_pydantic_metric(db, metric: models.Metric) -> schemas.Metric:
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


def _db_evaluation_settings_to_pydantic_evaluation_settings(
    evaluation_settings: models.Evaluation,
) -> schemas.EvaluationSettings:
    return schemas.EvaluationSettings(
        model=evaluation_settings.model.name,
        dataset=evaluation_settings.dataset.name,
        parameters=evaluation_settings.parameters,
        id=evaluation_settings.id,
    )


def _get_metrics_from_evaluation_settings(
    db: Session,
    evaluation_settings: list[models.Evaluation],
) -> list[schemas.Evaluations]:
    """Return a list of unnested Evaluations from a list of evaluation settings"""
    output = []

    for ms in evaluation_settings:
        job_id = ms.id
        status = jobs.get_stateflow().get_job_status(job_id=ms.id).value
        confusion_matrices = [
            schemas.ConfusionMatrix(
                label_key=matrix.label_key,
                entries=[
                    schemas.ConfusionMatrixEntry(**entry)
                    for entry in matrix.value
                ],
            )
            for matrix in ms.confusion_matrices
        ]

        # shared across evaluation settings, so just pick the first one
        dataset = ms.dataset.name
        model = ms.model.name
        filter_ = json.dumps(ms.parameters)

        metrics = [
            _db_metric_to_pydantic_metric(db, metric) for metric in ms.metrics
        ]

        output.append(
            {
                "dataset": dataset,
                "model": model,
                "filter": filter_,
                "job_id": job_id,
                "status": status,
                "metrics": metrics,
                "confusion_matrices": confusion_matrices,
            }
        )

    return output


def get_metrics_from_evaluation_ids(
    db: Session, evaluation_ids: list[int]
) -> list[schemas.Metric]:
    """Return all metrics for a list of evaluation ids"""

    eval_settings = db.scalars(
        select(models.Evaluation).where(
            models.Evaluation.id.in_(evaluation_ids)
        )
    ).all()

    return _get_metrics_from_evaluation_settings(db, eval_settings)


def get_evaluation_settings_from_id(
    db: Session, evaluation_id: int
) -> schemas.EvaluationSettings:
    ms = db.scalar(
        select(models.Evaluation).where(models.Evaluation.id == evaluation_id)
    )
    return _db_evaluation_settings_to_pydantic_evaluation_settings(ms)


def get_model_metrics(
    db: Session, model_name: str, evaluation_id: int
) -> list[schemas.Metric]:
    # TODO: may return multiple types of metrics
    # use get_model so exception get's raised if model does
    # not exist
    model = core.get_model(db, model_name)
    evaluation_settings = db.scalars(
        select(models.Evaluation)
        .join(models.Model)
        .where(
            and_(
                models.Model.id == model.id,
                models.Evaluation.id == evaluation_id,
            )
        )
    )

    return _get_metrics_from_evaluation_settings(db, evaluation_settings)


def get_model_evaluation_settings(
    db: Session, model_name: str
) -> list[schemas.EvaluationSettings]:
    model = core.get_model(db, model_name)
    all_eval_settings = db.scalars(
        select(models.Evaluation).where(models.Evaluation.model_id == model.id)
    ).all()
    return [
        _db_evaluation_settings_to_pydantic_evaluation_settings(eval_settings)
        for eval_settings in all_eval_settings
    ]
