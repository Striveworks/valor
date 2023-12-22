""" These integration tests should be run with a backend at http://localhost:8000
that is no auth
"""
from geoalchemy2.functions import ST_AsText
from sqlalchemy import select
from sqlalchemy.orm import Session

from velour import Annotation, Dataset, Label, Model, Prediction
from velour.client import Client
from velour.enums import TaskType
from velour.metatypes import ImageMetadata
from velour.schemas import BasicPolygon, BoundingBox, Point, Polygon
from velour_api.backend import models


def test_create_pred_detections_as_bbox_or_poly(
    db: Session,
    client: Client,
    dataset_name: str,
    model_name: str,
    gt_dets1: list[Annotation],
    img1: ImageMetadata,
):
    """Test that a predicted detection can be created as either a bounding box
    or a polygon
    """
    xmin, ymin, xmax, ymax = 10, 25, 30, 50

    dataset = Dataset(client, dataset_name)
    for gt in gt_dets1:
        dataset.add_groundtruth(gt)
    dataset.finalize()

    model = Model(client, model_name)
    pd = Prediction(
        model=model_name,
        datum=img1.to_datum(),
        annotations=[
            Annotation(
                task_type=TaskType.DETECTION,
                labels=[Label(key="k", value="v", score=0.6)],
                bounding_box=BoundingBox.from_extrema(
                    xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax
                ),
            ),
            Annotation(
                task_type=TaskType.DETECTION,
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
    model.add_prediction(pd)
    model.finalize_inferences(dataset)

    db_dets = db.scalars(
        select(models.Annotation).where(models.Annotation.model_id.isnot(None))
    ).all()
    assert len(db_dets) == 2
    assert set([db_det.box is not None for db_det in db_dets]) == {True, False}
    assert (
        db.scalar(ST_AsText(db_dets[0].box))
        == "POLYGON((10 25,30 25,30 50,10 50,10 25))"
        == db.scalar(ST_AsText(db_dets[1].polygon))
    )
