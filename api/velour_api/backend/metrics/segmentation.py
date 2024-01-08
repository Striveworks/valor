from geoalchemy2.functions import ST_Count, ST_MapAlgebra
from sqlalchemy.orm import Session, aliased
from sqlalchemy.sql import Select, func, select

from velour_api import enums, schemas
from velour_api.backend import models
from velour_api.backend.metrics.metric_utils import (
    computation_wrapper,
    create_metric_mappings,
    get_or_create_row,
)
from velour_api.backend.ops import Query
from velour_api.schemas.metrics import EvaluationJob, IOUMetric, mIOUMetric


def _generate_groundtruth_query(groundtruth_filter: schemas.Filter) -> Select:
    """Generate a sqlalchemy query to fetch a groundtruth."""
    return (
        Query(
            models.Annotation.id.label("annotation_id"),
            models.Annotation.datum_id.label("datum_id"),
        )
        .filter(groundtruth_filter)
        .groundtruths("gt")
    )


def _generate_prediction_query(prediction_filter: schemas.Filter) -> Select:
    """Generate a sqlalchemy query to fetch a prediction."""

    return (
        Query(
            models.Annotation.id.label("annotation_id"),
            models.Annotation.datum_id.label("datum_id"),
        )
        .filter(prediction_filter)
        .predictions("pd")
    )


def _count_true_positives(
    db: Session,
    groundtruth_subquery: Select,
    prediction_subquery: Select,
) -> int:
    """Computes the pixelwise true positives for the given dataset, model, and label"""
    gt_annotation = aliased(models.Annotation)
    pd_annotation = aliased(models.Annotation)
    ret = db.scalar(
        select(
            func.sum(
                ST_Count(
                    ST_MapAlgebra(
                        gt_annotation.raster,
                        pd_annotation.raster,
                        "[rast1]*[rast2]",  # https://postgis.net/docs/RT_ST_MapAlgebra_expr.html
                    )
                )
            )
        )
        .select_from(groundtruth_subquery)
        .join(
            prediction_subquery,
            prediction_subquery.c.datum_id == groundtruth_subquery.c.datum_id,
        )
        .join(
            gt_annotation,
            gt_annotation.id == groundtruth_subquery.c.annotation_id,
        )
        .join(
            pd_annotation,
            pd_annotation.id == prediction_subquery.c.annotation_id,
        )
    )
    if ret is None:
        return 0
    return int(ret)


def _count_groundtruths(
    db: Session,
    groundtruth_subquery: schemas.Filter,
    dataset_name: str,
    label_id: int,
) -> int:
    """Total number of groundtruth pixels for the given dataset and label"""
    ret = db.scalar(
        select(func.sum(ST_Count(models.Annotation.raster)))
        .select_from(groundtruth_subquery)
        .join(
            models.Annotation,
            models.Annotation.id == groundtruth_subquery.c.annotation_id,
        )
    )
    if ret is None:
        raise RuntimeError(
            f"No groundtruth pixels for label id '{label_id}' found in dataset '{dataset_name}'"
        )
    return int(ret)


def _count_predictions(
    db: Session,
    prediction_subquery: Select,
) -> int:
    """Total number of predicted pixels for the given dataset, model, and label"""
    ret = db.scalar(
        select(func.sum(ST_Count(models.Annotation.raster)))
        .select_from(prediction_subquery)
        .join(
            models.Annotation,
            models.Annotation.id == prediction_subquery.c.annotation_id,
        )
    )
    if ret is None:
        return 0
    return int(ret)


def _compute_iou(
    db: Session,
    groundtruth_filter: schemas.Filter,
    prediction_filter: schemas.Filter,
    dataset_name: str,
    label_id: int,
) -> float:
    """Computes the pixelwise intersection over union for the given dataset, model, and label"""

    groundtruth_subquery = _generate_groundtruth_query(groundtruth_filter)
    prediction_subquery = _generate_prediction_query(prediction_filter)

    tp = _count_true_positives(db, groundtruth_subquery, prediction_subquery)
    gt = _count_groundtruths(db, groundtruth_subquery, dataset_name, label_id)
    pred = _count_predictions(db, prediction_subquery)

    return tp / (gt + pred - tp)


def _get_groundtruth_labels(
    db: Session, groundtruth_filter: schemas.Filter
) -> list[models.Label]:
    """Fetch groundtruth labels from the database."""
    return db.scalars(
        Query(models.Label)
        .filter(groundtruth_filter)
        .groundtruths(as_subquery=False)
        .distinct()
    ).all()


def _compute_segmentation_metrics(
    db: Session,
    job_request: schemas.EvaluationJob,
) -> list[IOUMetric | mIOUMetric]:
    """
    Computes the _compute_IOU metrics. The return is one `IOUMetric` for each label in groundtruth
    and one `mIOUMetric` for the mean _compute_IOU over all labels.
    """

    # create groundtruth + prediction filters
    groundtruth_filter = job_request.settings.filters.model_copy()
    prediction_filter = job_request.settings.filters.model_copy()
    prediction_filter.models_names = [job_request.model]

    labels = _get_groundtruth_labels(db, groundtruth_filter)

    ret = []
    for label in labels:
        # set filter
        groundtruth_filter.label_ids = [label.id]
        prediction_filter.label_ids = [label.id]

        _compute_iou_score = _compute_iou(
            db,
            groundtruth_filter,
            prediction_filter,
            job_request.dataset,
            label.id,
        )

        ret.append(
            IOUMetric(
                label=schemas.Label(key=label.key, value=label.value),
                value=_compute_iou_score,
            )
        )

    ret.append(
        mIOUMetric(
            value=(
                sum([metric.value for metric in ret]) / len(ret)
                if len(ret) != 0
                else -1
            )
        )
    )

    return ret


@computation_wrapper
def compute_semantic_segmentation_metrics(
    *,
    db: Session,
    evaluation_id: int,
    job_request: EvaluationJob,
) -> int:
    """
    Create semantic segmentation metrics. This function is intended to be run using FastAPI's `BackgroundTasks`.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    evaluation_id : int
        The job ID to create metrics for.
    job_request : EvaluationJob
        The evaluation job.
    """
    evaluation = db.scalar(
        select(models.Evaluation).where(models.Evaluation.id == evaluation_id)
    )

    # unpack job request
    job_request = schemas.EvaluationJob(
        dataset=evaluation.dataset.name,
        model=evaluation.model.name,
        task_type=evaluation.task_type,
        settings=schemas.EvaluationSettings(**evaluation.settings),
        id=evaluation.id,
    )

    # check evaluation type
    if job_request.task_type != enums.TaskType.SEGMENTATION:
        raise ValueError(
            f"Cannot run segmentation evaluation on task with type `{job_request.task_type}`."
        )

    # configure filters
    if not job_request.settings.filters:
        job_request.settings.filters = schemas.Filter()
    job_request.settings.filters.task_types = [enums.TaskType.SEGMENTATION]
    job_request.settings.filters.dataset_names = [job_request.dataset]
    job_request.settings.filters.annotation_types = [
        enums.AnnotationType.RASTER
    ]

    metrics = _compute_segmentation_metrics(
        db,
        job_request,
    )
    metric_mappings = create_metric_mappings(db, metrics, evaluation_id)
    for mapping in metric_mappings:
        # ignore value since the other columns are unique identifiers
        # and have empirically noticed value can slightly change due to floating
        # point errors
        get_or_create_row(
            db,
            models.Metric,
            mapping,
            columns_to_ignore=["value"],
        )

    return evaluation_id
