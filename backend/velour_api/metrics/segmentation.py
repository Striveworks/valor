from geoalchemy2.functions import ST_Count, ST_MapAlgebra
from sqlalchemy.orm import Session
from sqlalchemy.sql import and_, join, select

from velour_api import models
from velour_api.ops import iou_two_segs

iou = iou_two_segs


# iterate through al the dataset and accumate:
# tp, fp, fn, tn


def tp_count(
    db: Session, dataset_name: str, model_name: str, label_id: int
) -> int:
    # see https://postgis.net/docs/RT_ST_MapAlgebra_expr.html

    return db.execute(
        select(
            ST_Count(
                ST_MapAlgebra(
                    models.GroundTruthSegmentation.shape,
                    models.PredictedSegmentation.shape,
                    "[rast1]*[rast2]",
                )
            )
        )
        .select_from(
            join(
                models.GroundTruthSegmentation,
                models.PredictedSegmentation,
                models.GroundTruthSegmentation.datum_id
                == models.PredictedSegmentation.datum_id,
            )
        )
        .join(
            models.LabeledGroundTruthSegmentation,
            and_(
                models.LabeledGroundTruthSegmentation.label_id == label_id,
                models.LabeledGroundTruthSegmentation.segmentation_id
                == models.GroundTruthSegmentation.id,
            ),
        )
        .join(
            models.LabeledPredictedSegmentation,
            and_(
                models.LabeledPredictedSegmentation.label_id == label_id,
                models.LabeledPredictedSegmentation.segmentation_id
                == models.PredictedSegmentation.id,
            ),
        )
        .join(models.Datum, models.Datum.id == models.PredictedSegmentation.id)
        .join(models.Dataset, models.Dataset.name == dataset_name)
        .join(models.Model, models.Model.name == model_name)
        .where(
            and_(
                models.GroundTruthSegmentation.datum_id
                == models.PredictedSegmentation.datum_id,
                models.GroundTruthSegmentation.is_instance
                == False,  # noqa: E712
                models.PredictedSegmentation.is_instance
                == False,  # noqa: E712
            )
        )
    )
