from geoalchemy2.functions import ST_Count, ST_MapAlgebra
from sqlalchemy.orm import Session
from sqlalchemy.sql import Select, func, join, select

from velour_api import enums, schemas
from velour_api.backend import core, models
from velour_api.backend.metrics.metrics import (
    create_metric_mappings,
    get_or_create_row,
)
from velour_api.backend.ops import Query
from velour_api.schemas.metrics import EvaluationJob, IOUMetric, mIOUMetric


def _generate_groundtruth_query(groundtruth_filter: schemas.Filter) -> Select:
    return (
        Query(
            models.Annotation.raster.label("raster"),
            models.Annotation.datum_id.label("datum_id"),
        )
        .filter(groundtruth_filter)
        .groundtruths("gt")
    )


def _generate_prediction_query(prediction_filter: schemas.Filter) -> Select:
    return (
        Query(
            models.Annotation.raster.label("raster"),
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
    ret = db.scalar(
        select(
            func.sum(
                ST_Count(
                    ST_MapAlgebra(
                        groundtruth_subquery.c.raster,
                        prediction_subquery.c.raster,
                        "[rast1]*[rast2]",  # https://postgis.net/docs/RT_ST_MapAlgebra_expr.html
                    )
                )
            )
        ).select_from(
            join(
                groundtruth_subquery,
                prediction_subquery,
                groundtruth_subquery.c.datum_id
                == prediction_subquery.c.datum_id,
            )
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
    ret = db.scalar(select(func.sum(ST_Count(groundtruth_subquery.c.raster))))
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
    ret = db.scalar(select(func.sum(ST_Count(prediction_subquery.c.raster))))
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
    """Computes the _compute_IOU metrics. The return is one `IOUMetric` for each label in groundtruth
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
        mIOUMetric(value=sum([metric.value for metric in ret]) / len(ret))
    )

    return ret


def create_semantic_segmentation_evaluation(
    db: Session, job_request: EvaluationJob
) -> int:
    """
    Create semantic segmentation evaluation job.
    """
    # check matching task_type
    if job_request.task_type != enums.TaskType.SEGMENTATION:
        raise TypeError(
            "Invalid task_type, please choose an evaluation method that supports semantic segmentation"
        )

    # validate parameters
    if job_request.settings.parameters:
        raise ValueError(
            "Semantic segmentation evaluations do not take parametric input."
        )

    # validate filters
    if not job_request.settings.filters:
        job_request.settings.filters = schemas.Filter()
    else:
        if (
            job_request.settings.filters.dataset_names is not None
            or job_request.settings.filters.dataset_metadata is not None
            or job_request.settings.filters.dataset_geospatial is not None
            or job_request.settings.filters.models_names is not None
            or job_request.settings.filters.models_metadata is not None
            or job_request.settings.filters.models_geospatial is not None
            or job_request.settings.filters.prediction_scores is not None
            or job_request.settings.filters.task_types is not None
            or job_request.settings.filters.annotation_types is not None
        ):
            raise ValueError(
                "Evaluation filter objects should not include any dataset, model, prediction score or task type filters."
            )

    dataset = core.get_dataset(db, job_request.dataset)
    model = core.get_model(db, job_request.model)

    es = get_or_create_row(
        db,
        models.Evaluation,
        mapping={
            "dataset_id": dataset.id,
            "model_id": model.id,
            "task_type": enums.TaskType.SEGMENTATION,
            "settings": job_request.settings.model_dump(),
        },
    )

    return es.id


def create_semantic_segmentation_metrics(
    db: Session,
    job_request: EvaluationJob,
    job_id: int,
) -> int:
    """
    Compute semantic segmentation evaluation.
    """
    evaluation = db.scalar(
        select(models.Evaluation).where(models.Evaluation.id == job_id)
    )

    # unpack job request
    job_request = schemas.EvaluationJob(
        dataset=evaluation.dataset.name,
        model=evaluation.model.name,
        task_type=evaluation.task_type,
        settings=schemas.EvaluationSettings(**evaluation.settings),
        id=evaluation.id,
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
    metric_mappings = create_metric_mappings(db, metrics, job_id)
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

    return job_id
