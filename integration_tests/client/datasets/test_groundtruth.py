""" These integration tests should be run with a back end at http://localhost:8000
that is no auth
"""

import numpy as np
import pytest
from geoalchemy2.functions import ST_AsText, ST_Polygon
from sqlalchemy import select
from sqlalchemy.orm import Session

from valor import Annotation, Client, Dataset, Datum, GroundTruth, Label
from valor.enums import TaskType
from valor.exceptions import ClientException
from valor.schemas import Box, MultiPolygon, Polygon, Raster
from valor_api.backend import models


def test_create_gt_detections_as_bbox_or_poly(
    db: Session,
    client: Client,
    dataset_name: str,
):
    """Test that a ground truth detection can be created as either a bounding box
    or a polygon
    """
    xmin, ymin, xmax, ymax = 10, 25, 30, 50
    image = Datum(
        uid="uid",
        metadata={
            "height": 200,
            "width": 150,
        },
    )

    dataset = Dataset.create(dataset_name)
    gt = GroundTruth(
        datum=image,
        annotations=[
            Annotation(
                task_type=TaskType.OBJECT_DETECTION,
                labels=[Label(key="k", value="v")],
                bounding_box=Box.from_extrema(
                    xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax
                ),
            ),
            Annotation(
                task_type=TaskType.OBJECT_DETECTION,
                labels=[Label(key="k", value="v")],
                polygon=Polygon(
                    [
                        [
                            (xmin, ymin),
                            (xmax, ymin),
                            (xmax, ymax),
                            (xmin, ymax),
                            (xmin, ymin),
                        ]
                    ]
                ),
            ),
        ],
    )
    dataset.add_groundtruth(gt)

    db_dets = db.scalars(
        select(models.Annotation).where(models.Annotation.model_id.is_(None))
    ).all()
    assert len(db_dets) == 2
    assert set([db_det.box is not None for db_det in db_dets]) == {
        True,
        False,
    }

    assert (
        str(db.scalar(ST_AsText(db_dets[0].box)))
        == "POLYGON((10 25,30 25,30 50,10 50,10 25))"
        == str(db.scalar(ST_AsText(db_dets[1].polygon)))
    )

    # check that they can be recovered by the client
    detections = dataset.get_groundtruth("uid")
    assert detections
    assert len(detections.annotations) == 2
    assert (
        len(
            [
                det
                for det in detections.annotations
                if det.box.get_value() is not None
            ]
        )
        == 1
    )
    for det in detections.annotations:
        if det.box.get_value():
            assert det.to_dict() == gt.annotations[0].to_dict()
        else:
            assert det.to_dict() == gt.annotations[1].to_dict()


def test_create_gt_segs_as_polys_or_masks(
    db: Session,
    client: Client,
    dataset_name: str,
    img1: Datum,
    image_height: int,
    image_width: int,
):
    """Test that we can create a dataset with ground truth segmentations that are defined
    both my polygons and mask arrays
    """
    xmin, xmax, ymin, ymax = 11, 45, 37, 102
    h, w = image_height, image_width
    mask = np.zeros((h, w), dtype=bool)
    mask[ymin:ymax, xmin:xmax] = True

    pts = [
        (xmin, ymin),
        (xmin, ymax),
        (xmax, ymax),
        (xmax, ymin),
        (xmin, ymin),
    ]
    poly = Polygon([pts])
    multipoly = MultiPolygon([[pts]])

    dataset = Dataset.create(dataset_name)

    # check we get an error for adding semantic segmentation with duplicate labels
    with pytest.raises(ClientException) as exc_info:
        gts = GroundTruth(
            datum=img1,
            annotations=[
                Annotation(
                    task_type=TaskType.SEMANTIC_SEGMENTATION,
                    labels=[Label(key="k1", value="v1")],
                    raster=Raster.from_numpy(mask),
                ),
                Annotation(
                    task_type=TaskType.SEMANTIC_SEGMENTATION,
                    labels=[Label(key="k1", value="v1")],
                    raster=Raster.from_geometry(
                        poly,
                        height=image_height,
                        width=image_width,
                    ),
                ),
            ],
        )

        dataset.add_groundtruth(gts)

    assert "one annotation per label" in str(exc_info.value)

    # fine with instance segmentation though
    gts = GroundTruth(
        datum=img1,
        annotations=[
            Annotation(
                task_type=TaskType.SEMANTIC_SEGMENTATION,
                labels=[Label(key="k1", value="v1")],
                raster=Raster.from_numpy(mask),
            ),
            Annotation(
                task_type=TaskType.OBJECT_DETECTION,
                labels=[Label(key="k1", value="v1")],
                raster=Raster.from_geometry(
                    multipoly,
                    height=image_height,
                    width=image_width,
                ),
            ),
        ],
    )

    dataset.add_groundtruth(gts)

    wkts = db.scalars(
        select(ST_AsText(ST_Polygon(models.Annotation.raster)))
    ).all()

    for wkt in wkts:
        assert (
            wkt
            == f"MULTIPOLYGON((({xmin} {ymin},{xmin} {ymax},{xmax} {ymax},{xmax} {ymin},{xmin} {ymin})))"
        )


def test_add_groundtruth(
    client: Client,
    dataset_name: str,
    gt_semantic_segs_mismatch: GroundTruth,
):
    dataset = Dataset.create(dataset_name)

    # make sure we get an error when passing a non-ground truth object to add_groundtruth
    with pytest.raises(TypeError):
        dataset.add_groundtruth("not_a_gt")  # type: ignore

    # make sure we get a warning when passing a ground truth without annotations
    with pytest.warns(UserWarning):
        dataset.add_groundtruth(
            GroundTruth(
                datum=Datum(
                    uid="uid",
                    metadata={
                        "height": 200,
                        "width": 150,
                    },
                ),
                annotations=[],
            )
        )

    # make sure raster is not dependent on datum metadata
    dataset.add_groundtruth(gt_semantic_segs_mismatch)

    client.delete_dataset(dataset_name, timeout=30)


def test_get_groundtruth(
    client: Client,
    dataset_name: str,
    gt_semantic_segs1_mask: GroundTruth,
    gt_semantic_segs2_mask: GroundTruth,
):
    dataset = Dataset.create(dataset_name)
    dataset.add_groundtruth(gt_semantic_segs1_mask)
    dataset.add_groundtruth(gt_semantic_segs2_mask)

    try:
        dataset.get_groundtruth("uid1")
        dataset.get_groundtruth("uid2")
    except Exception as e:
        raise AssertionError(e)

    client.delete_dataset(dataset_name, timeout=30)
