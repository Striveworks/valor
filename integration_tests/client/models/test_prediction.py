""" These integration tests should be run with a back end at http://localhost:8000
that is no auth
"""

from geoalchemy2.functions import ST_AsText
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.valor_api.backend import models
from client.valor import Annotation, Client, Dataset, Label, Model, Prediction
from client.valor.coretypes import GroundTruth
from client.valor.enums import TaskType
from client.valor.metatypes import Datum
from client.valor.schemas import BasicPolygon, BoundingBox, Point, Polygon


def test_create_pred_detections_as_bbox_or_poly(
    db: Session,
    client: Client,
    dataset_name: str,
    model_name: str,
    gt_dets1: list[GroundTruth],
    img1: Datum,
):
    """Test that a predicted detection can be created as either a bounding box
    or a polygon
    """
    xmin, ymin, xmax, ymax = 10, 25, 30, 50

    dataset = Dataset.create(dataset_name)
    for gt in gt_dets1:
        dataset.add_groundtruth(gt)
    dataset.finalize()

    model = Model.create(model_name)
    pd = Prediction(
        datum=img1,
        annotations=[
            Annotation(
                task_type=TaskType.OBJECT_DETECTION,
                labels=[Label(key="k", value="v", score=0.6)],
                bounding_box=BoundingBox.from_extrema(
                    xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax
                ),
            ),
            Annotation(
                task_type=TaskType.OBJECT_DETECTION,
                labels=[Label(key="k", value="v", score=0.4)],
                polygon=Polygon(
                    boundary=BasicPolygon(
                        points=[
                            Point(x=xmin, y=ymin),
                            Point(x=xmax, y=ymin),
                            Point(x=xmax, y=ymax),
                            Point(x=xmin, y=ymax),
                        ]
                    )
                ),
            ),
        ],
    )
    model.add_prediction(dataset, pd)
    model.finalize_inferences(dataset)

    db_dets = db.scalars(
        select(models.Annotation).where(models.Annotation.model_id.isnot(None))
    ).all()
    assert len(db_dets) == 3
    boxes = [det.box for det in db_dets if det.box is not None]
    assert len(boxes) == 1
    polygons = [det.polygon for det in db_dets if det.polygon is not None]
    assert len(polygons) == 1
    assert (
        db.scalar(ST_AsText(boxes[0]))
        == "POLYGON((10 25,30 25,30 50,10 50,10 25))"
        == db.scalar(ST_AsText(polygons[0]))
    )
