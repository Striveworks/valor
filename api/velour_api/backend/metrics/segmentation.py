from geoalchemy2.functions import ST_Count, ST_MapAlgebra
from sqlalchemy.orm import Session
from sqlalchemy.sql import Select, and_, func, join, select

from velour_api.backend import models
from velour_api.enums import TaskType

# iterate through al the dataset and accumate:
# tp, fp, fn, tn


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
    return db.execute(
        select(models.Label.key, models.Label.value, models.Label.id)
        .join(models.GroundTruth)
        .join(
            models.Annotation,
            models.GroundTruth.annotation_id == models.Annotation.id,
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
        .distinct()
    ).all()


def compute_segmentation_metrics(
    db: Session, dataset_name: str, model_name: str
):
    pass
