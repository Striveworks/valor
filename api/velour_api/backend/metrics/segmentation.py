from sqlalchemy.sql import and_, select

from velour_api.backend import models
from velour_api.enums import TaskType

# iterate through al the dataset and accumate:
# tp, fp, fn, tn


def _gt_query(dataset_name: str, label_id: int):
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


def _pred_query(dataset_name: str, label_id: int, model_name: str):
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
