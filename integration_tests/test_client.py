""" These integration tests should be run with a backend at http://localhost:8000
that is no auth
"""

import io
import json
import time
from typing import Any

import numpy as np
import pytest
from geoalchemy2.functions import ST_Area, ST_AsPNG, ST_AsText, ST_Polygon
from PIL import Image as PILImage
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.sql import text

from velour.client import Client, ClientException, Dataset, Model
from velour.data_types import (
    BoundingBox,
    BoundingPolygon,
    GroundTruthDetection,
    GroundTruthImageClassification,
    GroundTruthInstanceSegmentation,
    GroundTruthSemanticSegmentation,
    Image,
    Label,
    Point,
    PolygonWithHole,
    PredictedDetection,
    PredictedImageClassification,
    PredictedInstanceSegmentation,
    ScoredLabel,
)
from velour.metrics import Task
from velour_api import models, ops

dset_name = "test dataset"
model_name = "test model"


def bbox_to_poly(bbox: BoundingBox) -> BoundingPolygon:
    return BoundingPolygon(
        points=[
            Point(x=bbox.xmin, y=bbox.ymin),
            Point(x=bbox.xmin, y=bbox.ymax),
            Point(x=bbox.xmax, y=bbox.ymax),
            Point(x=bbox.xmax, y=bbox.ymin),
        ]
    )


def _list_of_points_from_wkt_polygon(
    db: Session, det: GroundTruthDetection | PredictedDetection
) -> list[tuple[int, int]]:
    geo = json.loads(db.scalar(det.boundary.ST_AsGeoJSON()))
    assert len(geo["coordinates"]) == 1
    return [tuple(p) for p in geo["coordinates"][0]]


def area(rect: BoundingPolygon) -> float:
    """Computes the area of a rectangle"""
    assert len(rect.points) == 4
    xs = [pt.x for pt in rect.points]
    ys = [pt.y for pt in rect.points]

    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def intersection_area(rect1: BoundingPolygon, rect2: BoundingPolygon) -> float:
    """Computes the intersection area of two rectangles"""
    assert len(rect1.points) == len(rect2.points) == 4

    xs1 = [pt.x for pt in rect1.points]
    xs2 = [pt.x for pt in rect2.points]

    ys1 = [pt.y for pt in rect1.points]
    ys2 = [pt.y for pt in rect2.points]

    inter_xmin = max(min(xs1), min(xs2))
    inter_xmax = min(max(xs1), max(xs2))

    inter_ymin = max(min(ys1), min(ys2))
    inter_ymax = min(max(ys1), max(ys2))

    inter_width = max(inter_xmax - inter_xmin, 0)
    inter_height = max(inter_ymax - inter_ymin, 0)

    return inter_width * inter_height


def iou(rect1: BoundingPolygon, rect2: BoundingPolygon) -> float:
    """Computes the "intersection over union" of two rectangles"""
    inter_area = intersection_area(rect1, rect2)
    return inter_area / (area(rect1) + area(rect2) - inter_area)


@pytest.fixture
def client():
    return Client(host="http://localhost:8000")


@pytest.fixture
def img1():
    return Image(uid="uid1", height=900, width=300)


@pytest.fixture
def img2():
    return Image(uid="uid2", height=400, width=300)


@pytest.fixture
def img5():
    return Image(uid="uid5", height=400, width=300)


@pytest.fixture
def img6():
    return Image(uid="uid6", height=400, width=300)


@pytest.fixture
def img8():
    return Image(uid="uid8", height=400, width=300)


@pytest.fixture
def img9():
    return Image(uid="uid9", height=400, width=300)


@pytest.fixture
def db(client: Client) -> Session:
    """This fixture makes sure there's not datasets, models, or labels in the backend
    (raising a RuntimeError if there are). It returns a db session and as cleanup
    clears out all datasets, models, and labels from the backend.
    """
    if len(client.get_datasets()) > 0:
        raise RuntimeError(
            "Tests should be run on an empty velour backend but found existing datasets."
        )

    if len(client.get_models()) > 0:
        raise RuntimeError(
            "Tests should be run on an empty velour backend but found existing models."
        )

    if len(client.get_all_labels()) > 0:
        raise RuntimeError(
            "Tests should be run on an empty velour backend but found existing labels."
        )

    engine = create_engine("postgresql://postgres:password@localhost/postgres")
    sess = Session(engine)
    sess.execute(text("SET postgis.gdal_enabled_drivers = 'ENABLE_ALL';"))
    sess.execute(text("SET postgis.enable_outdb_rasters = True;"))

    yield sess

    # cleanup by deleting all datasets, models, and labels
    for dataset in client.get_datasets():
        client.delete_dataset(name=dataset["name"])

    for model in client.get_models():
        client.delete_model(name=model["name"])

    labels = sess.scalars(select(models.Label))
    for label in labels:
        sess.delete(label)

    sess.commit()


@pytest.fixture
def rect1():
    return BoundingBox(xmin=10, ymin=10, xmax=60, ymax=40)


@pytest.fixture
def rect2():
    return BoundingBox(xmin=15, ymin=0, xmax=70, ymax=20)


@pytest.fixture
def rect3():
    return BoundingBox(xmin=87, ymin=10, xmax=158, ymax=820)


@pytest.fixture
def gt_dets1(
    rect1: BoundingBox, rect2: BoundingBox, img1: Image, img2: Image
) -> list[GroundTruthDetection]:
    return [
        GroundTruthDetection(
            bbox=rect1,
            labels=[Label(key="k1", value="v1")],
            image=img1,
        ),
        GroundTruthDetection(
            bbox=rect2,
            labels=[Label(key="k1", value="v1")],
            image=img2,
        ),
    ]


@pytest.fixture
def gt_poly_dets1(
    gt_dets1: list[GroundTruthDetection],
) -> list[GroundTruthDetection]:
    """Same thing as gt_dets1 but represented as a polygon instead of bounding box"""

    return [
        GroundTruthDetection(
            image=det.image, labels=det.labels, boundary=bbox_to_poly(det.bbox)
        )
        for det in gt_dets1
    ]


@pytest.fixture
def gt_dets2(rect3: BoundingBox, img1: Image) -> list[GroundTruthDetection]:
    return [
        GroundTruthDetection(
            bbox=rect3,
            labels=[Label(key="k2", value="v2")],
            image=img1,
        )
    ]


@pytest.fixture
def gt_dets3(rect3: BoundingBox, img8: Image) -> list[GroundTruthDetection]:
    return [
        GroundTruthDetection(
            bbox=rect3,
            labels=[Label(key="k3", value="v3")],
            image=img8,
        )
    ]


@pytest.fixture
def gt_segs1(
    rect1: BoundingBox, rect2: BoundingBox, img1: Image, img2: Image
) -> list[GroundTruthInstanceSegmentation]:
    return [
        GroundTruthInstanceSegmentation(
            shape=[PolygonWithHole(polygon=bbox_to_poly(rect1))],
            labels=[Label(key="k1", value="v1")],
            image=img1,
        ),
        GroundTruthInstanceSegmentation(
            shape=[
                PolygonWithHole(
                    polygon=bbox_to_poly(rect2), hole=bbox_to_poly(rect1)
                )
            ],
            labels=[Label(key="k1", value="v1")],
            image=img2,
        ),
    ]


@pytest.fixture
def gt_segs2(
    rect1: BoundingBox, rect3: BoundingBox, img1: Image
) -> list[GroundTruthSemanticSegmentation]:
    return [
        GroundTruthSemanticSegmentation(
            shape=[
                PolygonWithHole(polygon=bbox_to_poly(rect3)),
                PolygonWithHole(polygon=bbox_to_poly(rect1)),
            ],
            labels=[Label(key="k2", value="v2")],
            image=img1,
        )
    ]


@pytest.fixture
def gt_segs3(
    rect3: BoundingBox, img9: Image
) -> list[GroundTruthSemanticSegmentation]:
    return [
        GroundTruthSemanticSegmentation(
            shape=[PolygonWithHole(polygon=bbox_to_poly(rect3))],
            labels=[Label(key="k3", value="v3")],
            image=img9,
        )
    ]


@pytest.fixture
def gt_clfs1(img5: Image, img6: Image) -> list[GroundTruthImageClassification]:
    return [
        GroundTruthImageClassification(
            image=img5, labels=[Label(key="k5", value="v5")]
        ),
        GroundTruthImageClassification(
            image=img6, labels=[Label(key="k4", value="v4")]
        ),
    ]


@pytest.fixture
def gt_clfs2(img5: Image) -> list[GroundTruthImageClassification]:
    return [
        GroundTruthImageClassification(
            image=img5, labels=[Label(key="k4", value="v4")]
        )
    ]


@pytest.fixture
def gt_clfs3(img8: Image) -> list[GroundTruthImageClassification]:
    return [
        GroundTruthImageClassification(
            labels=[Label(key="k3", value="v3")],
            image=img8,
        )
    ]


@pytest.fixture
def pred_dets(
    rect1: BoundingBox, rect2: BoundingBox, img1: Image, img2: Image
) -> list[PredictedDetection]:
    return [
        PredictedDetection(
            bbox=rect1,
            scored_labels=[
                ScoredLabel(label=Label(key="k1", value="v1"), score=0.3)
            ],
            image=img1,
        ),
        PredictedDetection(
            bbox=rect2,
            scored_labels=[
                ScoredLabel(label=Label(key="k2", value="v2"), score=0.98)
            ],
            image=img2,
        ),
    ]


@pytest.fixture
def pred_poly_dets(
    pred_dets: list[PredictedDetection],
) -> list[PredictedDetection]:
    return [
        PredictedDetection(
            image=det.image,
            scored_labels=det.scored_labels,
            boundary=bbox_to_poly(det.bbox),
        )
        for det in pred_dets
    ]


@pytest.fixture
def pred_segs(img1: Image, img2: Image) -> list[PredictedInstanceSegmentation]:
    mask_1 = np.random.randint(0, 2, size=(64, 32), dtype=bool)
    mask_2 = np.random.randint(0, 2, size=(12, 23), dtype=bool)
    return [
        PredictedInstanceSegmentation(
            mask=mask_1,
            scored_labels=[
                ScoredLabel(label=Label(key="k1", value="v1"), score=0.87)
            ],
            image=img1,
        ),
        PredictedInstanceSegmentation(
            mask=mask_2,
            scored_labels=[
                ScoredLabel(label=Label(key="k2", value="v2"), score=0.92)
            ],
            image=img2,
        ),
    ]


@pytest.fixture
def pred_clfs(img5: Image, img6: Image) -> list[PredictedImageClassification]:
    return [
        PredictedImageClassification(
            image=img5,
            scored_labels=[
                ScoredLabel(label=Label(key="k12", value="v12"), score=0.47),
                ScoredLabel(label=Label(key="k12", value="v16"), score=0.53),
                ScoredLabel(label=Label(key="k13", value="v13"), score=1.0),
            ],
        ),
        PredictedImageClassification(
            image=img6,
            scored_labels=[
                ScoredLabel(label=Label(key="k4", value="v4"), score=0.71),
                ScoredLabel(label=Label(key="k4", value="v5"), score=0.29),
            ],
        ),
    ]


def _test_create_dataset_with_gts(
    client: Client,
    gts1: list[Any],
    gts2: list[Any],
    gts3: list[Any],
    add_method_name: str,
    expected_labels_tuples: set[tuple[str, str]],
    expected_image_uids: list[str],
) -> Dataset:
    """This test does the following
    - Creates a dataset
    - Adds groundtruth data to it in two batches
    - Verifies the images and labels have actually been added
    - Finalizes dataset
    - Tries to add more data and verifies an error is thrown

    Parameters
    ----------
    client
    gts1
        list of groundtruth objects (from `velour.data_types`)
    gts2
        list of groundtruth objects (from `velour.data_types`)
    gts3
        list of groundtruth objects (from `velour.data_types`)
    add_method_name
        method name of `velour.client.Dataset` to add groundtruth objects
    expected_labels_tuples
        set of tuples of key/value labels to check were added to the database
    expected_image_uids
        set of image uids to check were added to the database
    """
    dataset = client.create_dataset(dset_name)

    with pytest.raises(ClientException) as exc_info:
        client.create_dataset(dset_name)
    assert "already exists" in str(exc_info)

    add_method = getattr(dataset, add_method_name)
    add_method(gts1)
    add_method(gts2)

    # check that the dataset has two images
    images = dataset.get_images()
    assert len(images) == len(expected_image_uids)
    assert set([image.uid for image in images]) == expected_image_uids

    # check that there are two labels
    labels = dataset.get_labels()
    assert len(labels) == len(expected_labels_tuples)
    assert (
        set([(label.key, label.value) for label in labels])
        == expected_labels_tuples
    )

    dataset.finalize()
    # check that we get an error when trying to add more images
    # to the dataset since it is finalized
    with pytest.raises(ClientException) as exc_info:
        add_method(gts3)
    assert "since it is finalized" in str(exc_info)

    return dataset


def _test_create_model_with_preds(
    client: Client,
    gts: list[Any],
    preds: list[Any],
    add_gts_method_name: str,
    add_preds_method_name: str,
    preds_model_class: type,
    preds_expected_number: int,
    expected_labels_tuples: set[tuple[str, str]],
    expected_scores: set[float],
    db: Session,
):
    """Tests that the client can be used to add predictions.

    Parameters
    ----------
    client
    gts
        list of groundtruth objects (from `velour.data_types`)
    preds
        list of prediction objects (from `velour.data_types`)
    add_gts_method_name
        method name of `velour.client.Dataset` to add groundtruth objects
    add_preds_method_name
        method name of `velour.client.Model` to add prediction objects
    preds_model_class
        class in `velour_api.models` that specifies the labeled predictions
    preds_expected_number
        expected number of (labeled) predictions added to the database
    expected_labels_tuples
        set of tuples of key/value labels to check were added to the database
    expected_scores
        set of the scores of hte predictions
    db

    Returns
    -------
    the sqlalchemy objects for the created predictions
    """
    model = client.create_model(model_name)
    dataset = client.create_dataset(dset_name)

    add_preds_method = getattr(model, add_preds_method_name)

    # verify we get an error if we try to create another model
    # with the same name
    with pytest.raises(ClientException) as exc_info:
        client.create_model(model_name)
    assert "already exists" in str(exc_info)

    # check that if we try to add detections we get an error
    # since we haven't added any images yet
    with pytest.raises(ClientException) as exc_info:
        add_preds_method(dataset, preds)
    assert "Image with uid" in str(exc_info)

    add_gts_method = getattr(dataset, add_gts_method_name)

    add_gts_method(gts)
    add_preds_method(dataset, preds)

    # check predictions have been added
    db_preds = db.scalars(select(preds_model_class)).all()
    assert len(db_preds) == preds_expected_number

    # check labels
    assert (
        set([(p.label.key, p.label.value) for p in db_preds])
        == expected_labels_tuples
    )

    # check scores
    assert set([p.score for p in db_preds]) == expected_scores

    # check that the get_model method works
    retrieved_model = client.get_model(model_name)
    assert isinstance(retrieved_model, Model)
    assert retrieved_model.name == model_name

    return db_preds


def test_create_dataset_with_href_and_description(client: Client, db: Session):
    href = "http://a.com/b"
    description = "a description"
    client.create_dataset(dset_name, href=href, description=description)
    db_dataset = db.scalar(select(models.Dataset))
    assert db_dataset.href == href
    assert db_dataset.description == description


def test_create_model_with_href_and_description(client: Client, db: Session):
    href = "http://a.com/b"
    description = "a description"
    client.create_model(model_name, href=href, description=description)
    db_model = db.scalar(select(models.Model))
    assert db_model.href == href
    assert db_model.description == description


def test_create_dataset_with_detections(
    client: Client,
    gt_dets1: list[GroundTruthDetection],
    gt_dets2: list[GroundTruthDetection],
    gt_dets3: list[GroundTruthDetection],
    db: Session,  # this is unused but putting it here since the teardown of the fixture does cleanup
):
    dataset = _test_create_dataset_with_gts(
        client=client,
        gts1=gt_dets1,
        gts2=gt_dets2,
        gts3=gt_dets3,
        add_method_name="add_groundtruth",
        expected_image_uids={"uid1", "uid2"},
        expected_labels_tuples={
            ("k1", "v1"),
            ("k2", "v2"),
        },
    )

    dets1 = dataset.get_groundtruth_detections("uid1")
    dets2 = dataset.get_groundtruth_detections("uid2")

    # check we get back what we inserted
    gt_dets_uid1 = [
        gt for gt in gt_dets1 + gt_dets2 + gt_dets3 if gt.image.uid == "uid1"
    ]
    assert dets1 == gt_dets_uid1

    gt_dets_uid2 = [
        gt for gt in gt_dets1 + gt_dets2 + gt_dets3 if gt.image.uid == "uid2"
    ]
    assert dets2 == gt_dets_uid2


def test_create_model_with_predicted_detections(
    client: Client,
    gt_poly_dets1: list[GroundTruthDetection],
    pred_poly_dets: list[PredictedDetection],
    db: Session,
):
    labeled_pred_dets = _test_create_model_with_preds(
        client=client,
        gts=gt_poly_dets1,
        preds=pred_poly_dets,
        add_gts_method_name="add_groundtruth",
        add_preds_method_name="add_predicted_detections",
        preds_model_class=models.LabeledPredictedDetection,
        preds_expected_number=2,
        expected_labels_tuples={("k1", "v1"), ("k2", "v2")},
        expected_scores={0.3, 0.98},
        db=db,
    )

    # check boundary
    db_pred = [
        p for p in labeled_pred_dets if p.detection.image.uid == "uid1"
    ][0]
    points = _list_of_points_from_wkt_polygon(db, db_pred.detection)
    pred = pred_poly_dets[0]

    assert set(points) == set([(pt.x, pt.y) for pt in pred.boundary.points])


def test_create_gt_detections_as_bbox_or_poly(db: Session, client: Client):
    """Test that a groundtruth detection can be created as either a bounding box
    or a polygon
    """
    xmin, ymin, xmax, ymax = 10, 25, 30, 50
    image = Image(uid="uid", height=200, width=150)
    dataset = client.create_dataset(dset_name)

    gt_bbox = GroundTruthDetection(
        image=image,
        labels=[Label(key="k", value="v")],
        bbox=BoundingBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax),
    )
    gt_poly = GroundTruthDetection(
        image=image,
        labels=[Label(key="k", value="v")],
        boundary=BoundingPolygon(
            points=[
                Point(x=xmin, y=ymin),
                Point(x=xmin, y=ymax),
                Point(x=xmax, y=ymax),
                Point(x=xmax, y=ymin),
            ]
        ),
    )

    dataset.add_groundtruth([gt_bbox, gt_poly])

    db_dets = db.scalars(select(models.GroundTruthDetection)).all()
    assert len(db_dets) == 2
    assert set([db_det.is_bbox for db_det in db_dets]) == {True, False}
    assert (
        db.scalar(ST_AsText(db_dets[0].boundary))
        == "POLYGON((10 25,10 50,30 50,30 25,10 25))"
        == db.scalar(ST_AsText(db_dets[1].boundary))
    )

    # check that they can be recovered by the client
    detections = dataset.get_groundtruth_detections("uid")
    assert len(detections) == 2
    assert len([det for det in detections if det.is_bbox]) == 1
    for det in detections:
        if det.bbox:
            assert det == gt_bbox
        else:
            assert det == gt_poly


def test_create_pred_detections_as_bbox_or_poly(
    db: Session,
    client: Client,
    gt_dets1: list[GroundTruthDetection],
    img1: Image,
):
    """Test that a predicted detection can be created as either a bounding box
    or a polygon
    """
    xmin, ymin, xmax, ymax = 10, 25, 30, 50
    dataset = client.create_dataset(dset_name)
    model = client.create_model(model_name)

    dataset.add_groundtruth(gt_dets1)

    pred_bbox = PredictedDetection(
        image=img1,
        scored_labels=[
            ScoredLabel(label=Label(key="k", value="v"), score=0.6)
        ],
        bbox=BoundingBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax),
    )
    pred_poly = PredictedDetection(
        image=img1,
        scored_labels=[
            ScoredLabel(label=Label(key="k", value="v"), score=0.4)
        ],
        boundary=BoundingPolygon(
            points=[
                Point(x=xmin, y=ymin),
                Point(x=xmin, y=ymax),
                Point(x=xmax, y=ymax),
                Point(x=xmax, y=ymin),
            ]
        ),
    )

    model.add_predicted_detections(
        dataset=dataset, dets=[pred_bbox, pred_poly]
    )

    db_dets = db.scalars(select(models.PredictedDetection)).all()
    assert len(db_dets) == 2
    assert set([db_det.is_bbox for db_det in db_dets]) == {True, False}
    assert (
        db.scalar(ST_AsText(db_dets[0].boundary))
        == "POLYGON((10 25,10 50,30 50,30 25,10 25))"
        == db.scalar(ST_AsText(db_dets[1].boundary))
    )


def test_create_dataset_with_segmentations(
    client: Client,
    gt_segs1: list[GroundTruthSemanticSegmentation],
    gt_segs2: list[GroundTruthSemanticSegmentation],
    gt_segs3: list[GroundTruthSemanticSegmentation],
    db: Session,  # this is unused but putting it here since the teardown of the fixture does cleanup
):
    dataset = _test_create_dataset_with_gts(
        client=client,
        gts1=gt_segs1,
        gts2=gt_segs2,
        gts3=gt_segs3,
        add_method_name="add_groundtruth",
        expected_image_uids={"uid1", "uid2"},
        expected_labels_tuples={
            ("k1", "v1"),
            ("k2", "v2"),
        },
    )

    # should have one instance segmentation that's a rectangle
    # with xmin, ymin, xmax, ymax = 10, 10, 60, 40
    instance_segs = dataset.get_groundtruth_instance_segmentations("uid1")
    for seg in instance_segs:
        assert isinstance(seg, GroundTruthInstanceSegmentation)
    assert len(instance_segs) == 1
    mask = instance_segs[0].shape
    # check get all True in the box
    assert mask[10:40, 10:60].all()
    # check that outside the box is all False
    assert mask.sum() == (40 - 10) * (60 - 10)
    # check shape agrees with image
    assert mask.shape == (gt_segs1[0].image.height, gt_segs1[0].image.width)

    # should have one semantic segmentation that's a rectangle
    # with xmin, ymin, xmax, ymax = 10, 10, 60, 40 plus a rectangle
    # with xmin, ymin, xmax, ymax = 87, 10, 158, 820
    semantic_segs = dataset.get_groundtruth_semantic_segmentations("uid1")
    for seg in semantic_segs:
        assert isinstance(seg, GroundTruthSemanticSegmentation)
    mask = semantic_segs[0].shape
    assert mask[10:40, 10:60].all()
    assert mask[10:820, 87:158].all()
    assert mask.sum() == (40 - 10) * (60 - 10) + (820 - 10) * (158 - 87)
    assert mask.shape == (gt_segs1[0].image.height, gt_segs1[0].image.width)


def test_create_gt_segs_as_polys_or_masks(
    client: Client, img1: Image, db: Session
):
    """Test that we can create a dataset with groundtruth segmentations that are defined
    both my polygons and mask arrays
    """
    dataset = client.create_dataset(dset_name)

    xmin, xmax, ymin, ymax = 11, 45, 37, 102
    h, w = 900, 300
    mask = np.zeros((h, w), dtype=bool)
    mask[ymin:ymax, xmin:xmax] = True

    poly = PolygonWithHole(
        polygon=BoundingPolygon(
            [
                Point(x=xmin, y=ymin),
                Point(x=xmin, y=ymax),
                Point(x=xmax, y=ymax),
                Point(x=xmax, y=ymin),
            ]
        )
    )

    gt1 = GroundTruthSemanticSegmentation(
        shape=mask, labels=[Label(key="k1", value="v1")], image=img1
    )
    gt2 = GroundTruthSemanticSegmentation(
        shape=[poly], labels=[Label(key="k1", value="v1")], image=img1
    )

    dataset.add_groundtruth([gt1, gt2])
    wkts = db.scalars(
        select(ST_AsText(ST_Polygon(models.GroundTruthSegmentation.shape)))
    ).all()

    for wkt in wkts:
        assert (
            wkt
            == f"MULTIPOLYGON((({xmin} {ymin},{xmin} {ymax},{xmax} {ymax},{xmax} {ymin},{xmin} {ymin})))"
        )


def test_create_model_with_predicted_segmentations(
    client: Client,
    gt_segs1: list[GroundTruthInstanceSegmentation],
    pred_segs: list[PredictedInstanceSegmentation],
    db: Session,
):
    """Tests that we can create a predicted segmentation from a mask array"""
    labeled_pred_segs = _test_create_model_with_preds(
        client=client,
        gts=gt_segs1,
        preds=pred_segs,
        add_gts_method_name="add_groundtruth",
        add_preds_method_name="add_predicted_segmentations",
        preds_model_class=models.LabeledPredictedSegmentation,
        preds_expected_number=2,
        expected_labels_tuples={("k1", "v1"), ("k2", "v2")},
        expected_scores={0.87, 0.92},
        db=db,
    )

    # grab the segmentation from the db, recover the mask, and check
    # its equal to the mask the client sent over
    db_pred = [
        p for p in labeled_pred_segs if p.segmentation.image.uid == "uid1"
    ][0]
    png_from_db = db.scalar(ST_AsPNG(db_pred.segmentation.shape))
    f = io.BytesIO(png_from_db.tobytes())
    mask_array = np.array(PILImage.open(f))

    np.testing.assert_equal(mask_array, pred_segs[0].mask)


def test_create_dataset_with_classifications(
    client: Client,
    gt_clfs1: list[GroundTruthImageClassification],
    gt_clfs2: list[GroundTruthImageClassification],
    gt_clfs3: list[GroundTruthImageClassification],
    db: Session,  # this is unused but putting it here since the teardown of the fixture does cleanup
):
    _test_create_dataset_with_gts(
        client=client,
        gts1=gt_clfs1,
        gts2=gt_clfs2,
        gts3=gt_clfs3,
        add_method_name="add_groundtruth",
        expected_image_uids={"uid5", "uid6"},
        expected_labels_tuples={
            ("k5", "v5"),
            ("k4", "v4"),
        },
    )


def test_create_model_with_predicted_classifications(
    client: Client,
    gt_clfs1: list[GroundTruthDetection],
    pred_clfs: list[PredictedDetection],
    db: Session,
):
    _test_create_model_with_preds(
        client=client,
        gts=gt_clfs1,
        preds=pred_clfs,
        add_gts_method_name="add_groundtruth",
        add_preds_method_name="add_predicted_classifications",
        preds_model_class=models.PredictedImageClassification,
        preds_expected_number=5,
        expected_labels_tuples={
            ("k12", "v12"),
            ("k12", "v16"),
            ("k13", "v13"),
            ("k4", "v4"),
            ("k4", "v5"),
        },
        expected_scores={0.47, 0.53, 1.0, 0.71, 0.29},
        db=db,
    )


def test_boundary(
    client: Client, db: Session, rect1: BoundingPolygon, img1: Image
):
    """Test consistency of boundary in backend and client"""
    dataset = client.create_dataset(dset_name)
    rect1_poly = bbox_to_poly(rect1)
    dataset.add_groundtruth(
        [
            GroundTruthDetection(
                boundary=rect1_poly,
                labels=[Label(key="k1", value="v1")],
                image=img1,
            )
        ]
    )

    # get the one detection that exists
    db_det = db.scalar(select(models.GroundTruthDetection))

    # check boundary
    points = _list_of_points_from_wkt_polygon(db, db_det)

    assert set(points) == set([(pt.x, pt.y) for pt in rect1_poly.points])


def test_iou(
    client: Client,
    db: Session,
    rect1: BoundingPolygon,
    rect2: BoundingPolygon,
    img1: Image,
):
    dataset = client.create_dataset(dset_name)
    model = client.create_model(model_name)

    rect1_poly = bbox_to_poly(rect1)
    rect2_poly = bbox_to_poly(rect2)

    dataset.add_groundtruth(
        [
            GroundTruthDetection(
                boundary=rect1_poly, labels=[Label("k", "v")], image=img1
            )
        ]
    )
    db_gt = db.scalar(select(models.GroundTruthDetection))

    model.add_predicted_detections(
        dataset,
        [
            PredictedDetection(
                boundary=rect2_poly,
                scored_labels=[ScoredLabel(label=Label("k", "v"), score=0.6)],
                image=img1,
            )
        ],
    )
    db_pred = db.scalar(select(models.PredictedDetection))

    assert ops.iou_two_dets(db, db_gt, db_pred) == iou(rect1_poly, rect2_poly)


def test_delete_dataset_exception(client: Client):
    with pytest.raises(ClientException) as exc_info:
        client.delete_dataset("non-existent dataset")
    assert "does not exist" in str(exc_info)


def test_evaluate_ap(
    client: Client,
    gt_dets1: list[GroundTruthDetection],
    pred_dets: list[PredictedDetection],
    db: Session,
):
    dataset = client.create_dataset(dset_name)
    dataset.add_groundtruth(gt_dets1)
    dataset.finalize()

    model = client.create_model(model_name)
    model.add_predicted_detections(dataset, pred_dets)
    model.finalize_inferences(dataset)

    eval_job = model.evaluate_ap(
        dataset=dataset,
        iou_thresholds=[0.1, 0.6],
        ious_to_keep=[0.1, 0.6],
    )

    assert eval_job.ignored_pred_labels == [Label(key="k2", value="v2")]
    assert eval_job.missing_pred_labels == []
    assert isinstance(eval_job._id, str)

    # sleep to give the backend time to compute
    time.sleep(1)
    assert eval_job.status() == "Done"

    settings = eval_job.settings()
    settings.pop("id")
    assert settings == {
        "model_name": "test model",
        "dataset_name": "test dataset",
        "model_pred_task_type": "Bounding Box Object Detection",
        "dataset_gt_task_type": "Bounding Box Object Detection",
        "min_area": None,
        "max_area": None,
    }

    expected_metrics = [
        {
            "type": "AP",
            "value": 0.504950495049505,
            "label": {"key": "k1", "value": "v1"},
            "parameters": {
                "iou": 0.1,
            },
        },
        {
            "type": "AP",
            "value": 0.504950495049505,
            "label": {"key": "k1", "value": "v1"},
            "parameters": {
                "iou": 0.6,
            },
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.1},
            "value": 0.504950495049505,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.6},
            "value": 0.504950495049505,
        },
        {
            "type": "APAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.504950495049505,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "mAPAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.504950495049505,
        },
    ]

    assert eval_job.metrics() == expected_metrics

    # now test if we set min_area and/or max_area
    areas = db.scalars(ST_Area(models.GroundTruthDetection.boundary)).all()
    assert sorted(areas) == [1100.0, 1500.0]

    # sanity check this should give us the same thing excpet min_area and max_area
    # are not None
    eval_job = model.evaluate_ap(
        dataset=dataset,
        model_pred_task_type=Task.BBOX_OBJECT_DETECTION,
        dataset_gt_task_type=Task.BBOX_OBJECT_DETECTION,
        iou_thresholds=[0.1, 0.6],
        ious_to_keep=[0.1, 0.6],
        min_area=10,
        max_area=2000,
    )
    time.sleep(1)
    settings = eval_job.settings()
    settings.pop("id")
    assert settings == {
        "model_name": "test model",
        "dataset_name": "test dataset",
        "model_pred_task_type": "Bounding Box Object Detection",
        "dataset_gt_task_type": "Bounding Box Object Detection",
        "min_area": 10,
        "max_area": 2000,
    }
    assert eval_job.metrics() == expected_metrics

    # now check we get different things by setting the thresholds accordingly
    eval_job = model.evaluate_ap(
        dataset=dataset,
        model_pred_task_type=Task.BBOX_OBJECT_DETECTION,
        dataset_gt_task_type=Task.BBOX_OBJECT_DETECTION,
        iou_thresholds=[0.1, 0.6],
        ious_to_keep=[0.1, 0.6],
        min_area=1200,
    )
    time.sleep(1)

    settings = eval_job.settings()
    settings.pop("id")
    assert settings == {
        "model_name": "test model",
        "dataset_name": "test dataset",
        "model_pred_task_type": "Bounding Box Object Detection",
        "dataset_gt_task_type": "Bounding Box Object Detection",
        "min_area": 1200,
        "max_area": None,
    }

    assert eval_job.metrics() != expected_metrics

    eval_job = model.evaluate_ap(
        dataset=dataset,
        model_pred_task_type=Task.BBOX_OBJECT_DETECTION,
        dataset_gt_task_type=Task.BBOX_OBJECT_DETECTION,
        iou_thresholds=[0.1, 0.6],
        ious_to_keep=[0.1, 0.6],
        max_area=1200,
    )
    time.sleep(1)
    settings = eval_job.settings()
    settings.pop("id")
    assert settings == {
        "model_name": "test model",
        "dataset_name": "test dataset",
        "model_pred_task_type": "Bounding Box Object Detection",
        "dataset_gt_task_type": "Bounding Box Object Detection",
        "min_area": None,
        "max_area": 1200,
    }
    assert eval_job.metrics() != expected_metrics

    eval_job = model.evaluate_ap(
        dataset=dataset,
        model_pred_task_type=Task.BBOX_OBJECT_DETECTION,
        dataset_gt_task_type=Task.BBOX_OBJECT_DETECTION,
        iou_thresholds=[0.1, 0.6],
        ious_to_keep=[0.1, 0.6],
        min_area=1200,
        max_area=1800,
    )
    time.sleep(1)
    settings = eval_job.settings()
    settings.pop("id")
    assert settings == {
        "model_name": "test model",
        "dataset_name": "test dataset",
        "model_pred_task_type": "Bounding Box Object Detection",
        "dataset_gt_task_type": "Bounding Box Object Detection",
        "min_area": 1200,
        "max_area": 1800,
    }
    assert eval_job.metrics() != expected_metrics


def test_evaluate_clf(
    client: Client,
    gt_clfs1: list[GroundTruthImageClassification],
    pred_clfs: list[PredictedImageClassification],
    db: Session,  # this is unused but putting it here since the teardown of the fixture does cleanup
):
    dataset = client.create_dataset(dset_name)
    dataset.add_groundtruth(gt_clfs1)
    dataset.finalize()

    model = client.create_model(model_name)
    model.add_predicted_classifications(dataset, pred_clfs)
    model.finalize_inferences(dataset)

    eval_job = model.evaluate_classification(dataset=dataset)

    assert set(eval_job.ignored_pred_keys) == {"k12", "k13"}
    assert set(eval_job.missing_pred_keys) == {"k5"}

    # sleep to give the backend time to compute
    time.sleep(1)
    assert eval_job.status() == "Done"

    metrics = eval_job.metrics()

    expected_metrics = [
        {"type": "Accuracy", "parameters": {"label_key": "k4"}, "value": 1.0},
        {"type": "ROCAUC", "parameters": {"label_key": "k4"}, "value": 1.0},
        {
            "type": "Precision",
            "value": 1.0,
            "label": {"key": "k4", "value": "v4"},
        },
        {
            "type": "Recall",
            "value": 1.0,
            "label": {"key": "k4", "value": "v4"},
        },
        {"type": "F1", "value": 1.0, "label": {"key": "k4", "value": "v4"}},
        {"type": "Precision", "label": {"key": "k4", "value": "v5"}},
        {"type": "Recall", "label": {"key": "k4", "value": "v5"}},
        {"type": "F1", "label": {"key": "k4", "value": "v5"}},
    ]
    for m in metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in metrics

    confusion_matrices = eval_job.confusion_matrices()
    assert confusion_matrices == [
        {
            "label_key": "k4",
            "entries": [{"prediction": "v4", "groundtruth": "v4", "count": 1}],
        }
    ]
