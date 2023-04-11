import io

import numpy as np
import pytest
from PIL import Image
from sqlalchemy import insert
from sqlalchemy.orm import Session

from velour_api import models, ops, schemas
from velour_api.crud._create import (
    _boundary_points_to_str,
    _select_statement_from_poly,
)


def bytes_to_pil(b: bytes) -> Image.Image:
    f = io.BytesIO(b)
    img = Image.open(f)
    return img


def pil_to_bytes(img: Image.Image) -> bytes:
    f = io.BytesIO()
    img.save(f, format="PNG")
    f.seek(0)
    return f.read()


@pytest.fixture
def model(db: Session) -> models.Model:
    model = models.Model(name="model")
    db.add(model)
    db.commit()

    return model


@pytest.fixture
def mask_bytes_poly_intersection():
    """Returns a bytes mask, a polygon, and the area of intersection of them."""
    h, w = 100, 200
    y_min, y_max, x_min, x_max = 50, 80, 20, 30
    mask = np.zeros(shape=(h, w), dtype=bool)
    mask[y_min:y_max, x_min:x_max] = True
    mask_bytes = pil_to_bytes(Image.fromarray(mask))

    poly_y_min, poly_y_max, poly_x_min, poly_x_max = 60, 90, 10, 25
    poly = schemas.PolygonWithHole(
        polygon=[
            (poly_x_min, poly_y_min),
            (poly_x_max, poly_y_min),
            (poly_x_max, poly_y_max),
            (poly_x_min, poly_y_max),
        ]
    )

    inter_xmin = max(x_min, poly_x_min)
    inter_xmax = min(x_max, poly_x_max)
    inter_ymin = max(y_min, poly_y_min)
    inter_ymax = min(y_max, poly_y_max)

    intersection_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)

    mask_area = (y_max - y_min) * (x_max - x_min)
    poly_area = (poly_y_max - poly_y_min) * (poly_x_max - poly_x_min)

    return mask_bytes, poly, mask_area, poly_area, intersection_area


def _pred_seg_from_bytes(
    db: Session, mask_bytes: bytes, model: models.Model, img: models.Image
) -> models.PredictedSegmentation:
    pred_seg = models.PredictedSegmentation(
        shape=mask_bytes, image_id=img.id, model_id=model.id, is_instance=True
    )
    db.add(pred_seg)
    db.commit()
    return pred_seg


def _gt_seg_from_polys(
    db: Session, polys: list[schemas.PolygonWithHole], img: models.Image
) -> models.GroundTruthSegmentation:
    mapping = {
        "shape": _select_statement_from_poly(polys),
        "image_id": img.id,
        "is_instance": False,
    }
    gt_seg = db.scalar(
        insert(models.GroundTruthSegmentation)
        .values([mapping])
        .returning(models.GroundTruthSegmentation)
    )

    db.commit()
    return gt_seg


def test__raster_area(
    db: Session, mask_bytes1: bytes, model: models.Model, img: models.Image
):
    pred_seg = _pred_seg_from_bytes(
        db=db, mask_bytes=mask_bytes1, model=model, img=img
    )

    mask = bytes_to_pil(mask_bytes1)
    assert ops._raster_area(db, pred_seg.shape) == np.array(mask).sum()


def test_intersection_area_of_segs(
    db: Session, model: models.Model, img: models.Image
):
    h, w = 100, 200
    y_min, y_max, x_min, x_max = 50, 80, 20, 30
    mask = np.zeros(shape=(h, w), dtype=bool)
    mask[y_min:y_max, x_min:x_max] = True
    mask_bytes = pil_to_bytes(Image.fromarray(mask))

    poly_y_min, poly_y_max, poly_x_min, poly_x_max = 60, 90, 10, 25
    poly = schemas.PolygonWithHole(
        polygon=[
            (poly_x_min, poly_y_min),
            (poly_x_max, poly_y_min),
            (poly_x_max, poly_y_max),
            (poly_x_min, poly_y_max),
        ]
    )

    inter_xmin = max(x_min, poly_x_min)
    inter_xmax = min(x_max, poly_x_max)
    inter_ymin = max(y_min, poly_y_min)
    inter_ymax = min(y_max, poly_y_max)

    pred_seg = _pred_seg_from_bytes(
        db=db, mask_bytes=mask_bytes, model=model, img=img
    )
    gt_seg = _gt_seg_from_polys(db=db, polys=[poly], img=img)

    assert ops.intersection_area_of_segs(
        db=db, seg1=gt_seg, seg2=pred_seg
    ) == (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)


def test_intersection_pred_seg_multi_poly_gt_seg(
    db: Session, model: models.Model, img: models.Image
):
    """Tests intersection of a prediction mask with a groundtruth
    that's comprised of two disjoint polygons, with one having a hole
    """

    h, w = 300, 800
    y_min, y_max, x_min, x_max = 7, 290, 108, 316
    mask = np.zeros(shape=(h, w), dtype=bool)
    mask[y_min:y_max, x_min:x_max] = True
    mask_bytes = pil_to_bytes(Image.fromarray(mask))

    poly_y_min, poly_y_max, poly_x_min, poly_x_max = 103, 200, 92, 330
    polygon = [
        (poly_x_min, poly_y_min),
        (poly_x_max, poly_y_min),
        (poly_x_max, poly_y_max),
        (poly_x_min, poly_y_max),
    ]
    # a hole inside the polygon that's completely inside the mask
    hole_y_min, hole_y_max, hole_x_min, hole_x_max = 110, 170, 124, 190
    hole = [
        (hole_x_min, hole_y_min),
        (hole_x_max, hole_y_min),
        (hole_x_max, hole_y_max),
        (hole_x_min, hole_y_max),
    ]
    poly1 = schemas.PolygonWithHole(polygon=polygon, hole=hole)
    # triangle contained in the mask
    poly2 = schemas.PolygonWithHole(
        polygon=[(200, 210), (200, 250), (265, 210)]
    )

    inter_xmin = max(x_min, poly_x_min)
    inter_xmax = min(x_max, poly_x_max)
    inter_ymin = max(y_min, poly_y_min)
    inter_ymax = min(y_max, poly_y_max)

    pred_seg = _pred_seg_from_bytes(
        db=db, mask_bytes=mask_bytes, model=model, img=img
    )
    gt_seg = _gt_seg_from_polys(db=db, polys=[poly1, poly2], img=img)

    area_int_mask_rect = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    area_hole = (hole_x_max - hole_x_min) * (hole_y_max - hole_y_min)
    area_triangle = (265 - 200) * (250 - 210) / 2

    assert (
        ops.intersection_area_of_segs(db=db, seg1=gt_seg, seg2=pred_seg)
        == area_int_mask_rect + area_triangle - area_hole
    )


def test_iou_and_intersection_area_of_two_segs(
    db: Session,
    model: models.Model,
    img: models.Image,
    mask_bytes_poly_intersection: tuple[bytes, schemas.PolygonWithHole, float],
):
    (
        mask_bytes,
        poly,
        mask_area,
        poly_area,
        intersection_area,
    ) = mask_bytes_poly_intersection

    pred_seg = _pred_seg_from_bytes(
        db=db, mask_bytes=mask_bytes, model=model, img=img
    )
    gt_seg = _gt_seg_from_polys(db=db, polys=[poly], img=img)

    assert (
        ops.intersection_area_of_segs(db=db, seg1=gt_seg, seg2=pred_seg)
        == intersection_area
    )

    assert ops.iou_two_segs(db, gt_seg, pred_seg) == intersection_area / (
        mask_area + poly_area - intersection_area
    )


def test_iou_det_and_seg(
    db: Session,
    model: models.Model,
    img: models.Image,
    mask_bytes_poly_intersection: tuple[bytes, schemas.PolygonWithHole, float],
):
    (
        mask_bytes,
        poly,
        mask_area,
        poly_area,
        intersection_area,
    ) = mask_bytes_poly_intersection
    seg = _pred_seg_from_bytes(
        db=db, mask_bytes=mask_bytes, model=model, img=img
    )
    det = models.GroundTruthDetection(
        boundary=f"POLYGON({_boundary_points_to_str(poly.polygon)})",
        image_id=img.id,
        is_bbox=True,
    )
    db.add(det)
    db.commit()

    assert ops.iou_det_and_seg(db, det, seg) == intersection_area / (
        mask_area + poly_area - intersection_area
    )

    # now create a mask that's a triangle and check intersection is correct
    # when computed against a bounding box detection and a polygon detection
    # that contains the triangle
    mask = np.zeros((10, 30), dtype=bool)
    for i in range(10):
        for j in range(30):
            if i + j < 10:
                mask[i, j] = True
    mask_bytes = pil_to_bytes(Image.fromarray(mask))
    seg = _pred_seg_from_bytes(
        db=db, mask_bytes=mask_bytes, model=model, img=img
    )

    ymin, ymax, xmin, xmax = 0, 10, 0, 20
    poly = f"POLYGON({_boundary_points_to_str([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])})"
    bbox_det = models.GroundTruthDetection(
        boundary=poly,
        image_id=img.id,
        is_bbox=True,
    )
    poly_det = models.GroundTruthDetection(
        boundary=poly,
        image_id=img.id,
        is_bbox=False,
    )

    db.add(bbox_det)
    db.add(poly_det)
    db.commit()

    # the segmentation gets changed to its convex hull. this turns out to have
    # area 59.5 (honestly not exactly sure why, probably some aliasing thing?)
    # since the triangle is contained in the detection, that area is also the area
    # of its intersection
    area_convex_hull_triangle = 59.5
    assert ops.iou_det_and_seg(
        db, poly_det, seg
    ) == area_convex_hull_triangle / (
        10 * 20 + area_convex_hull_triangle - area_convex_hull_triangle
    )

    # this should be the IOU of the reactangle that circumscribes the triangle,
    # which is a 10x10 square that's contained in the 10x20 bbox_det so
    # should get .5
    assert ops.iou_det_and_seg(db, bbox_det, seg) == 0.5


def test_iou_two_dets(db: Session, img: models.Image):
    ymin1, ymax1, xmin1, xmax1 = 50, 80, 20, 30
    ymin2, ymax2, xmin2, xmax2 = 60, 90, 10, 25

    # poly1 and poly2 are bounding boxes
    poly1 = f"POLYGON({_boundary_points_to_str([(xmin1, ymin1), (xmax1, ymin1), (xmax1, ymax1), (xmin1, ymax1)])})"
    poly2 = f"POLYGON({_boundary_points_to_str([(xmin2, ymin2), (xmax2, ymin2), (xmax2, ymax2), (xmin2, ymax2)])})"
    # triangle thats half of the bounding box poly1
    poly3 = f"POLYGON({_boundary_points_to_str([(xmin1, ymin1), (xmax1, ymin1), (xmax1, ymax1)])})"

    # intersection area of poly1 and poly2
    poly1_poly2_intersection_area = (25 - 20) * (80 - 60)
    area_poly1 = (80 - 50) * (30 - 20)
    area_poly2 = (90 - 60) * (25 - 10)
    poly1_poly2_iou = poly1_poly2_intersection_area / (
        area_poly1 + area_poly2 - poly1_poly2_intersection_area
    )

    bbox_det1 = models.GroundTruthDetection(
        boundary=poly1,
        image_id=img.id,
        is_bbox=True,
    )
    bbox_det2 = models.GroundTruthDetection(
        boundary=poly2,
        image_id=img.id,
        is_bbox=True,
    )
    poly_det1 = models.GroundTruthDetection(
        boundary=poly1,
        image_id=img.id,
        is_bbox=False,
    )
    poly_det2 = models.GroundTruthDetection(
        boundary=poly2,
        image_id=img.id,
        is_bbox=False,
    )
    poly_det3 = models.GroundTruthDetection(
        boundary=poly3,
        image_id=img.id,
        is_bbox=False,
    )

    db.add(bbox_det1)
    db.add(bbox_det2)
    db.add(poly_det1)
    db.add(poly_det2)
    db.add(poly_det3)
    db.commit()

    # check IOU of two bounding boxes is usual IOU
    assert ops.iou_two_dets(db, bbox_det1, bbox_det2) == poly1_poly2_iou
    # check that we still get the same thing if iou is computed with the second bounding
    # box as a polygon type
    assert ops.iou_two_dets(db, bbox_det1, poly_det2) == poly1_poly2_iou

    # check that we still get the same thing if iou is computed with both bounding boxes
    # as polygon types
    assert ops.iou_two_dets(db, poly_det1, poly_det2) == poly1_poly2_iou

    # doing intersection of rectangle det as a polygon with triangle should give half the area
    # of the rectangle for the intersection. area of poly_det3 is half the area of poly_det1
    poly1_poly3_intersection_area = 0.5 * area_poly1
    area_poly3 = 0.5 * area_poly1
    assert ops.iou_two_dets(
        db, poly_det1, poly_det3
    ) == poly1_poly3_intersection_area / (
        area_poly1 + area_poly3 - poly1_poly3_intersection_area
    )

    # doing intersection of rectangle det as a bounding box with triangle should give an IOU 1
    # since the triangle gets converted to the circumsribing bounding box (which is bbox_det1)
    assert ops.iou_two_dets(db, bbox_det1, poly_det3) == 1.0
