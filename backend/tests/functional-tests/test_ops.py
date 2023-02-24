import io

import numpy as np
import pytest
from PIL import Image
from sqlalchemy.orm import Session

from velour_api import models, ops, schemas
from velour_api.crud import _wkt_multipolygon_from_polygons_with_hole


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
def dset(db: Session) -> models.Dataset:
    dset = models.Dataset(name="dset")
    db.add(dset)
    db.commit()

    return dset


@pytest.fixture
def model(db: Session) -> models.Model:
    model = models.Model(name="model")
    db.add(model)
    db.commit()

    return model


@pytest.fixture
def img(db: Session, dset: models.Dataset) -> models.Image:
    img = models.Image(uri="uri", dataset_id=dset.id)
    db.add(img)
    db.commit()

    return img


def _pred_seg_from_bytes(
    db: Session, mask_bytes: bytes, model: models.Model, img: models.Image
) -> models.PredictedSegmentation:
    pred_seg = models.PredictedSegmentation(
        shape=mask_bytes, image_id=img.id, model_id=model.id
    )
    db.add(pred_seg)
    db.commit()
    return pred_seg


def _gt_seg_from_polys(
    db: Session, polys: schemas.PolygonWithHole, img: models.Image
) -> models.GroundTruthSegmentation:
    wkt_poly = _wkt_multipolygon_from_polygons_with_hole(polys)
    gt_seg = models.GroundTruthSegmentation(shape=wkt_poly, image_id=img.id)
    db.add(gt_seg)
    db.commit()
    return gt_seg


def test_area_pred_seg(
    db: Session, mask_bytes1: bytes, model: models.Model, img: models.Image
):
    pred_seg = _pred_seg_from_bytes(
        db=db, mask_bytes=mask_bytes1, model=model, img=img
    )

    mask = bytes_to_pil(mask_bytes1)
    assert ops.pred_seg_area(db, pred_seg) == np.array(mask).sum()


def test_intersection_pred_seg_gt_seg(
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

    assert ops.intersection_area_of_gt_seg_and_pred_seg(
        db=db, gt_seg=gt_seg, pred_seg=pred_seg
    ) == (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)


def test_intersection_pred_seg_multi_poly_gt_seg(
    db: Session, model: models.Model, img: models.Image
):
    """Tests intersection of a predictino mask with a groundtruth
    that's comprised of two disjoint polygons
    """

    h, w = 300, 800
    y_min, y_max, x_min, x_max = 7, 290, 108, 316
    mask = np.zeros(shape=(h, w), dtype=bool)
    mask[y_min:y_max, x_min:x_max] = True
    mask_bytes = pil_to_bytes(Image.fromarray(mask))

    poly_y_min, poly_y_max, poly_x_min, poly_x_max = 103, 200, 92, 330
    poly1 = schemas.PolygonWithHole(
        polygon=[
            (poly_x_min, poly_y_min),
            (poly_x_max, poly_y_min),
            (poly_x_max, poly_y_max),
            (poly_x_min, poly_y_max),
        ]
    )
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
    area_triangle = (265 - 200) * (250 - 210) / 2

    assert (
        ops.intersection_area_of_gt_seg_and_pred_seg(
            db=db, gt_seg=gt_seg, pred_seg=pred_seg
        )
        == area_int_mask_rect + area_triangle
    )
