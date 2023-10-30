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


def _db_evaluation_job_to_pydantic_evaluation_job(
    evaluation: models.Evaluation,
) -> schemas.EvaluationJob:
    return schemas.EvaluationJob(
        model=evaluation.model.name,
        dataset=evaluation.dataset.name,
        settings=evaluation.settings,
        id=evaluation.id,
    )


def get_metrics_from_evaluation_job(
    db: Session,
    evaluation: list[models.Evaluation],
) -> list[schemas.Metric]:
    return [
        _db_metric_to_pydantic_metric(db, m)
        for ms in evaluation
        for m in ms.metrics
    ]


def _get_bulk_metrics_from_evaluation_job(
    db: Session,
    evaluation: list[models.Evaluation],
) -> list[schemas.BulkEvaluations]:
    """Return a list of unnested Evaluations from a list of evaluation settings"""
    output = []

    for ms in evaluation:
        job_id = ms.id
        status = (
            jobs.get_stateflow()
            .get_job_status(job_id=ms.id)
            .name.replace('"', "")
        )
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
        dataset = ms.metrics[0].settings.dataset.name
        model = ms.metrics[0].settings.model.name
        filter_ = json.dumps(ms.metrics[0].settings.parameters)

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


def get_metrics_from_evaluation_id(
    db: Session, evaluation_id: int
) -> list[schemas.Metric]:
    """Return metrics for a specific evaluation id"""
    eval_settings = db.scalar(
        select(models.Evaluation).where(models.Evaluation.id == evaluation_id)
    )
    return get_metrics_from_evaluation_job(db, [eval_settings])


def get_metrics_from_evaluation_ids(
    db: Session, evaluation_ids: list[int]
) -> list[schemas.Metric]:
    """Return all metrics for a list of evaluation ids"""
    eval_settings = db.scalars(
        select(models.Evaluation).where(
            models.Evaluation.id.in_(evaluation_ids)
        )
    ).all()

    return _get_bulk_metrics_from_evaluation_job(db, eval_settings)


def get_confusion_matrices_from_evaluation_id(
    db: Session, evaluation_id: int
) -> list[schemas.ConfusionMatrix]:
    eval_settings = db.scalar(
        select(models.Evaluation).where(models.Evaluation.id == evaluation_id)
    )
    db_cms = eval_settings.confusion_matrices

    return [
        schemas.ConfusionMatrix(
            label_key=db_cm.label_key,
            entries=[
                schemas.ConfusionMatrixEntry(**entry) for entry in db_cm.value
            ],
        )
        for db_cm in db_cms
    ]


def get_evaluation_job_from_id(
    db: Session, evaluation_id: int
) -> schemas.EvaluationJob:
    evaluation_job = db.scalar(
        select(models.Evaluation).where(models.Evaluation.id == evaluation_id)
    )
    return _db_evaluation_job_to_pydantic_evaluation_job(evaluation_job)


def get_model_metrics(
    db: Session, model_name: str, evaluation_id: int
) -> list[schemas.Metric]:
    # TODO: may return multiple types of metrics
    # use get_model so exception get's raised if model does
    # not exist
    model = core.get_model(db, model_name)
    evaluation_job = db.scalars(
        select(models.Evaluation)
        .join(models.Model)
        .where(
            and_(
                models.Model.id == model.id,
                models.Evaluation.id == evaluation_id,
            )
        )
    )

    return get_metrics_from_evaluation_job(db, evaluation_job)


def get_model_evaluation_jobs(
    db: Session, model_name: str
) -> list[schemas.EvaluationJob]:
    model = core.get_model(db, model_name)
    all_evals = db.scalars(
        select(models.Evaluation).where(models.Evaluation.model_id == model.id)
    ).all()
    return [
        _db_evaluation_job_to_pydantic_evaluation_job(job) for job in all_evals
    ]
