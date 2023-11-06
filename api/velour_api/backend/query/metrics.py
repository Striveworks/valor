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


def _db_evaluation_job_to_pydantic_evaluation_job(
    evaluation_job: models.Evaluation,
) -> schemas.EvaluationJob:
    return schemas.EvaluationJob(
        model=evaluation_job.model.name,
        dataset=evaluation_job.dataset.name,
        settings=evaluation_job.settings,
        id=evaluation_job.id,
    )


def _get_metrics_from_evaluation_settings(
    db: Session,
    evaluation_jobs: list[models.Evaluation],
) -> list[schemas.Evaluation]:
    """Return a list of unnested Evaluations from a list of evaluation settings"""
    output = []

    for ms in evaluation_jobs:
        job_id = ms.id
        status = jobs.get_stateflow().get_job_status(job_id=ms.id).value
        confusion_matrices = [
            schemas.ConfusionMatrixResponse(
                label_key=matrix.label_key,
                entries=[
                    schemas.ConfusionMatrixEntry(**entry)
                    for entry in matrix.value
                ],
            )
            for matrix in ms.confusion_matrices
        ]

        metrics = [
            _db_metric_to_pydantic_metric(db, metric) for metric in ms.metrics
        ]

        output.append(
            schemas.Evaluation(
                dataset=ms.dataset.name,
                model=ms.model.name,
                settings=ms.settings,
                job_id=job_id,
                status=status,
                metrics=metrics,
                confusion_matrices=confusion_matrices,
            )
        )

    return output


def get_metrics_from_evaluation_ids(
    db: Session, evaluation_ids: list[int]
) -> list[schemas.Evaluation]:
    """Return all metrics for a list of evaluation ids"""

    evaluation_jobs = db.scalars(
        select(models.Evaluation).where(
            models.Evaluation.id.in_(evaluation_ids)
        )
    ).all()

    return _get_metrics_from_evaluation_settings(db, evaluation_jobs)


def get_evaluation_job_from_id(
    db: Session, evaluation_id: int
) -> schemas.EvaluationJob:
    evaluation_job = db.scalar(
        select(models.Evaluation).where(models.Evaluation.id == evaluation_id)
    )
    return _db_evaluation_job_to_pydantic_evaluation_job(evaluation_job)


def get_model_metrics(
    db: Session, model_name: str, evaluation_id: int
) -> list[schemas.Evaluation]:
    # TODO: may return multiple types of metrics
    # use get_model so exception get's raised if model does
    # not exist
    model = core.get_model(db, model_name)
    evaluation_jobs = db.scalars(
        select(models.Evaluation)
        .join(models.Model)
        .where(
            and_(
                models.Model.id == model.id,
                models.Evaluation.id == evaluation_id,
            )
        )
    )

    return _get_metrics_from_evaluation_settings(db, evaluation_jobs)


def get_model_evaluation_jobs(
    db: Session, model_name: str
) -> list[schemas.EvaluationJob]:
    model = core.get_model(db, model_name)
    evaluation_jobs = db.scalars(
        select(models.Evaluation).where(models.Evaluation.model_id == model.id)
    ).all()
    return [
        _db_evaluation_job_to_pydantic_evaluation_job(job)
        for job in evaluation_jobs
    ]
