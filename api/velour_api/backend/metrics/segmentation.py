from geoalchemy2.functions import ST_Count, ST_MapAlgebra
from sqlalchemy.orm import Session
from sqlalchemy.sql import Select, and_, func, join, select

from velour_api.backend import core, models
from velour_api.backend.metrics.core import (
    create_metric_mappings,
    get_or_create_row,
)
from velour_api.backend.query.label import get_dataset_labels_query
from velour_api.enums import AnnotationType, TaskType
from velour_api.schemas import Label
from velour_api.schemas.metrics import (
    IOUMetric,
    SemanticSegmentationMetricsRequest,
    mIOUMetric,
)


def _gt_query(dataset_name: str, label_id: int) -> Select:
    return (
        select(
            models.Annotation.raster.label("raster"),
            models.Annotation.datum_id.label("datum_id"),
        )
        .join(
            models.GroundTruth,
            and_(
                models.GroundTruth.label_id == label_id,
                models.GroundTruth.annotation_id == models.Annotation.id,
            ),
        )
        .join(models.Dataset, models.Dataset.name == dataset_name)
        .join(
            models.Datum,
            and_(
                models.Datum.dataset_id == models.Dataset.id,
                models.Datum.id == models.Annotation.datum_id,
            ),
        )
        .where(models.Annotation.task_type == TaskType.SEMANTIC_SEGMENTATION)
    )


def _pred_query(dataset_name: str, label_id: int, model_name: str) -> Select:
    return (
        select(
            models.Annotation.raster.label("raster"),
            models.Annotation.datum_id.label("datum_id"),
        )
        .join(
            models.Prediction,
            and_(
                models.Prediction.label_id == label_id,
                models.Prediction.annotation_id == models.Annotation.id,
            ),
        )
        .join(models.Dataset, models.Dataset.name == dataset_name)
        .join(models.Model, models.Model.name == model_name)
        .join(
            models.Datum,
            and_(
                models.Datum.dataset_id == models.Dataset.id,
                models.Datum.id == models.Annotation.datum_id,
            ),
        )
        .where(
            and_(
                models.Annotation.task_type == TaskType.SEMANTIC_SEGMENTATION,
                models.Model.id == models.Annotation.model_id,
            )
        )
    )


def tp_count(
    db: Session, dataset_name: str, model_name: str, label_id: int
) -> int:
    """Computes the pixelwise true positives for the given dataset, model, and label"""

    gt = _gt_query(dataset_name, label_id).subquery()
    pred = _pred_query(
        dataset_name=dataset_name, label_id=label_id, model_name=model_name
    ).subquery()

    ret = db.scalar(
        select(
            func.sum(
                ST_Count(
                    ST_MapAlgebra(
                        gt.c.raster,
                        pred.c.raster,
                        "[rast1]*[rast2]",  # https://postgis.net/docs/RT_ST_MapAlgebra_expr.html
                    )
                )
            )
        ).select_from(join(gt, pred, gt.c.datum_id == pred.c.datum_id))
    )

    if ret is None:
        return 0

    return int(ret)


def gt_count(db: Session, dataset_name: str, label_id: int) -> int:
    """Total number of groundtruth pixels for the given dataset and label"""
    gt = _gt_query(dataset_name, label_id).subquery()
    ret = db.scalar(select(func.sum(ST_Count(gt.c.raster))))
    if ret is None:
        raise RuntimeError(
            f"No groundtruth pixels for label id '{label_id}' found in dataset '{dataset_name}'"
        )

    return int(ret)


def pred_count(
    db: Session, dataset_name: str, model_name: str, label_id: int
) -> int:
    """Total number of predicted pixels for the given dataset, model, and label"""
    pred = _pred_query(
        dataset_name=dataset_name, label_id=label_id, model_name=model_name
    ).subquery()
    ret = db.scalar(select(func.sum(ST_Count(pred.c.raster))))
    if ret is None:
        return 0
    return int(ret)


def iou(
    db: Session, dataset_name: str, model_name: str, label_id: int
) -> float:
    """Computes the pixelwise intersection over union for the given dataset, model, and label"""
    tp = tp_count(db, dataset_name, model_name, label_id)
    gt = gt_count(db, dataset_name, label_id)
    pred = pred_count(db, dataset_name, model_name, label_id)

    return tp / (gt + pred - tp)


def get_groundtruth_labels(
    db: Session, dataset_name: str
) -> list[tuple[str, str, int]]:
    """Gets all unique groundtruth labels for semenatic segmentations
    in the dataset. Return is list of tuples (label key, label value, label id)
    """
    return [
        (label.key, label.value, label.id)
        for label in db.scalars(
            get_dataset_labels_query(
                dataset_name=dataset_name,
                annotation_type=AnnotationType.RASTER,
                task_types=[TaskType.SEMANTIC_SEGMENTATION],
            )
        )
    ]


def compute_segmentation_metrics(
    db: Session, dataset_name: str, model_name: str
) -> list[IOUMetric | mIOUMetric]:
    """Computes the IOU metrics. The return is one `IOUMetric` for each label in groundtruth
    and one `mIOUMetric` for the mean IOU over all labels.
    """
    labels = get_groundtruth_labels(db, dataset_name)
    ret = []
    for label in labels:
        iou_score = iou(db, dataset_name, model_name, label[2])

        ret.append(
            IOUMetric(
                label=Label(key=label[0], value=label[1]), value=iou_score
            )
        )

    ret.append(
        mIOUMetric(value=sum([metric.value for metric in ret]) / len(ret))
    )

    return ret


def create_semantic_segmentation_evaluation(
    db: Session, request_info: SemanticSegmentationMetricsRequest
) -> int:
    dataset = core.get_dataset(db, request_info.settings.dataset)
    model = core.get_model(db, request_info.settings.model)

    es = get_or_create_row(
        db,
        models.EvaluationSettings,
        mapping={
            "dataset_id": dataset.id,
            "model_id": model.id,
            "task_type": TaskType.SEMANTIC_SEGMENTATION,
            "target_type": AnnotationType.NONE,
        },
    )

    return es.id


def create_semantic_segmentation_metrics(
    db: Session,
    request_info: SemanticSegmentationMetricsRequest,
    evaluation_settings_id: int,
) -> int:
    metrics = compute_segmentation_metrics(
        db,
        dataset_name=request_info.settings.dataset,
        model_name=request_info.settings.model,
    )
    metric_mappings = create_metric_mappings(
        db, metrics, evaluation_settings_id
    )
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

    return evaluation_settings_id
