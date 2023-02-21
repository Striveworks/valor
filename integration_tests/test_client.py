import json
from typing import List, Union

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from velour.client import Client, ClientException
from velour.data_types import (
    BoundingPolygon,
    GroundTruthDetection,
    GroundTruthImageClassification,
    GroundTruthSegmentation,
    Image,
    Label,
    Point,
    PolygonWithHole,
    PredictedDetection,
    PredictedImageClassification,
    ScoredLabel,
)

from velour_api import models, ops

dset_name = "test dataset"
model_name = "test model"


def _list_of_points_from_wkt_polygon(
    db: Session, det: Union[GroundTruthDetection, PredictedDetection]
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
    return BoundingPolygon(
        [
            Point(x=10, y=10),
            Point(x=10, y=40),
            Point(x=60, y=40),
            Point(x=60, y=10),
        ]
    )


@pytest.fixture
def rect2():
    return BoundingPolygon(
        [
            Point(x=15, y=0),
            Point(x=70, y=0),
            Point(x=70, y=20),
            Point(x=15, y=20),
        ]
    )


@pytest.fixture
def rect3():
    return BoundingPolygon(
        [
            Point(x=158, y=10),
            Point(x=87, y=10),
            Point(x=87, y=820),
            Point(x=158, y=820),
        ]
    )


@pytest.fixture
def gt_dets1(
    rect1: BoundingPolygon, rect2: BoundingPolygon
) -> List[GroundTruthDetection]:
    return [
        GroundTruthDetection(
            boundary=rect1,
            labels=[Label(key="k1", value="v1")],
            image=Image(uri="uri1"),
        ),
        GroundTruthDetection(
            boundary=rect2,
            labels=[Label(key="k1", value="v1")],
            image=Image(uri="uri2"),
        ),
    ]


@pytest.fixture
def gt_dets2(rect3: BoundingPolygon) -> List[GroundTruthDetection]:
    return [
        GroundTruthDetection(
            boundary=rect3,
            labels=[Label(key="k2", value="v2")],
            image=Image(uri="uri1"),
        )
    ]


@pytest.fixture
def gt_dets3(rect3: BoundingPolygon) -> List[GroundTruthDetection]:
    return [
        GroundTruthDetection(
            boundary=rect3,
            labels=[Label(key="k3", value="v3")],
            image=Image(uri="uri8"),
        )
    ]


@pytest.fixture
def gt_segs1(
    rect1: BoundingPolygon, rect2: BoundingPolygon
) -> List[GroundTruthSegmentation]:
    return [
        GroundTruthSegmentation(
            shape=[PolygonWithHole(polygon=rect1)],
            labels=[Label(key="k1", value="v1")],
            image=Image(uri="uri1"),
        ),
        GroundTruthSegmentation(
            shape=[PolygonWithHole(polygon=rect2, hole=rect1)],
            labels=[Label(key="k1", value="v1")],
            image=Image(uri="uri2"),
        ),
    ]


@pytest.fixture
def gt_segs2(
    rect1: BoundingPolygon, rect3: BoundingPolygon
) -> List[GroundTruthDetection]:
    return [
        GroundTruthSegmentation(
            shape=[
                PolygonWithHole(polygon=rect3),
                PolygonWithHole(polygon=rect1),
            ],
            labels=[Label(key="k2", value="v2")],
            image=Image(uri="uri1"),
        )
    ]


@pytest.fixture
def gt_segs3(rect3: BoundingPolygon) -> List[GroundTruthDetection]:
    return [
        GroundTruthSegmentation(
            shape=[PolygonWithHole(polygon=rect3)],
            labels=[Label(key="k3", value="v3")],
            image=Image(uri="uri9"),
        )
    ]


@pytest.fixture
def gt_clfs1() -> List[GroundTruthImageClassification]:
    return [
        GroundTruthImageClassification(
            image=Image(uri="uri5"), labels=[Label(key="k5", value="v5")]
        ),
        GroundTruthImageClassification(
            image=Image(uri="uri6"), labels=[Label(key="k4", value="v4")]
        ),
    ]


@pytest.fixture
def gt_clfs2() -> List[GroundTruthImageClassification]:
    return [
        GroundTruthImageClassification(
            image=Image(uri="uri5"), labels=[Label(key="k4", value="v4")]
        )
    ]


@pytest.fixture
def gt_clfs3() -> List[GroundTruthImageClassification]:
    return [
        GroundTruthImageClassification(
            labels=[Label(key="k3", value="v3")],
            image=Image(uri="uri8"),
        )
    ]


@pytest.fixture
def pred_dets(
    rect1: BoundingPolygon, rect2: BoundingPolygon
) -> List[PredictedDetection]:
    return [
        PredictedDetection(
            boundary=rect1,
            scored_labels=[
                ScoredLabel(label=Label(key="k1", value="v1"), score=0.3)
            ],
            image=Image(uri="uri1"),
        ),
        PredictedDetection(
            boundary=rect2,
            scored_labels=[
                ScoredLabel(label=Label(key="k2", value="v2"), score=0.98)
            ],
            image=Image(uri="uri2"),
        ),
    ]


@pytest.fixture
def pred_clfs() -> List[PredictedImageClassification]:
    return [
        PredictedImageClassification(
            image=Image(uri="uri5"),
            scored_labels=[
                ScoredLabel(label=Label(key="k12", value="v12"), score=0.47),
                ScoredLabel(label=Label(key="k13", value="v13"), score=0.12),
            ],
        ),
        PredictedImageClassification(
            image=Image(uri="uri6"),
            scored_labels=[
                ScoredLabel(label=Label(key="k4", value="v4"), score=0.71),
            ],
        ),
    ]


def _test_create_dataset_with_gts(
    client: Client,
    gts1,
    gts2,
    gts3,
    add_method_name,
    expected_labels_tuples,
    expected_image_uris,
    db: Session,
):
    """This test does the following
    - Creates a dataset
    - Adds groundtruth data to it in two batches
    - Verifies the images and labels have actually been added
    - Finalizes dataset
    - Tries to add more data and verifies an error is thrown
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
    assert len(images) == len(expected_image_uris)
    assert set([image.uri for image in images]) == expected_image_uris

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


def test_create_dataset_with_detections(
    client: Client,
    gt_dets1: List[GroundTruthDetection],
    gt_dets2: List[GroundTruthDetection],
    gt_dets3: List[GroundTruthDetection],
    db: Session,  # this is unused but putting it here since the teardown of the fixture does cleanup
):
    _test_create_dataset_with_gts(
        client=client,
        gts1=gt_dets1,
        gts2=gt_dets2,
        gts3=gt_dets3,
        add_method_name="add_groundtruth_detections",
        expected_image_uris={"uri1", "uri2"},
        expected_labels_tuples={
            ("k1", "v1"),
            ("k2", "v2"),
        },
        db=db,
    )


def test_create_model_with_predicted_detections(
    client: Client,
    gt_dets1: List[GroundTruthDetection],
    pred_dets: List[PredictedDetection],
    db: Session,
):
    model = client.create_model(model_name)

    # verify we get an error if we try to create another model
    # with the same name
    with pytest.raises(ClientException) as exc_info:
        client.create_model(model_name)
    assert "already exists" in str(exc_info)

    # check that if we try to add detections we get an error
    # since we haven't added any images yet
    with pytest.raises(ClientException) as exc_info:
        model.add_predicted_detections(pred_dets)
    assert "Image with uri" in str(exc_info)

    # add groundtruth and predictions
    dataset = client.create_dataset(dset_name)
    dataset.add_groundtruth_detections(gt_dets1)
    model.add_predicted_detections(pred_dets)

    # check predictions have been added
    labeled_pred_dets = db.scalars(
        select(models.LabeledPredictedDetection)
    ).all()
    assert len(labeled_pred_dets) == 2

    # check labels
    assert set([(p.label.key, p.label.value) for p in labeled_pred_dets]) == {
        ("k1", "v1"),
        ("k2", "v2"),
    }

    # check scores
    assert set([p.score for p in labeled_pred_dets]) == {0.3, 0.98}

    # check boundary
    db_pred = [
        p for p in labeled_pred_dets if p.detection.image.uri == "uri1"
    ][0]
    points = _list_of_points_from_wkt_polygon(db, db_pred.detection)
    pred = pred_dets[0]

    assert set(points) == set([(pt.x, pt.y) for pt in pred.boundary.points])


def test_create_dataset_with_segmentations(
    client: Client,
    gt_segs1: List[GroundTruthSegmentation],
    gt_segs2: List[GroundTruthSegmentation],
    gt_segs3: List[GroundTruthSegmentation],
    db: Session,  # this is unused but putting it here since the teardown of the fixture does cleanup
):
    _test_create_dataset_with_gts(
        client=client,
        gts1=gt_segs1,
        gts2=gt_segs2,
        gts3=gt_segs3,
        add_method_name="add_groundtruth_segmentations",
        expected_image_uris={"uri1", "uri2"},
        expected_labels_tuples={
            ("k1", "v1"),
            ("k2", "v2"),
        },
        db=db,
    )


def test_create_dataset_with_classifications(
    client: Client,
    gt_clfs1: List[GroundTruthImageClassification],
    gt_clfs2: List[GroundTruthImageClassification],
    gt_clfs3: List[GroundTruthImageClassification],
    db: Session,  # this is unused but putting it here since the teardown of the fixture does cleanup
):
    _test_create_dataset_with_gts(
        client=client,
        gts1=gt_clfs1,
        gts2=gt_clfs2,
        gts3=gt_clfs3,
        add_method_name="add_groundtruth_classifications",
        expected_image_uris={"uri5", "uri6"},
        expected_labels_tuples={
            ("k5", "v5"),
            ("k4", "v4"),
        },
        db=db,
    )


def test_create_model_with_predicted_classifications(
    client: Client,
    gt_clfs1: List[GroundTruthDetection],
    pred_clfs: List[PredictedDetection],
    db: Session,
):
    model = client.create_model(model_name)

    # verify we get an error if we try to create another model
    # with the same name
    with pytest.raises(ClientException) as exc_info:
        client.create_model(model_name)
    assert "already exists" in str(exc_info)

    # check that if we try to add detections we get an error
    # since we haven't added any images yet
    with pytest.raises(ClientException) as exc_info:
        model.add_predicted_classifications(pred_clfs)
    assert "Image with uri" in str(exc_info)

    # add groundtruth and predictions
    dataset = client.create_dataset(dset_name)
    dataset.add_groundtruth_classifications(gt_clfs1)
    model.add_predicted_classifications(pred_clfs)

    # check predictions have been added
    pred_clfs = db.scalars(select(models.PredictedImageClassification)).all()
    assert len(pred_clfs) == 3

    # check labels
    assert set([(p.label.key, p.label.value) for p in pred_clfs]) == {
        ("k12", "v12"),
        ("k13", "v13"),
        ("k4", "v4"),
    }

    # check scores
    assert set([p.score for p in pred_clfs]) == {0.47, 0.12, 0.71}


def test_boundary(client: Client, db: Session, rect1: BoundingPolygon):
    """Test consistency of boundary in backend and client"""
    dataset = client.create_dataset(dset_name)
    dataset.add_groundtruth_detections(
        [
            GroundTruthDetection(
                boundary=rect1,
                labels=[Label(key="k1", value="v1")],
                image=Image(uri="uri1"),
            )
        ]
    )

    # get the one detection that exists
    db_det = db.scalar(select(models.GroundTruthDetection))

    # check boundary
    points = _list_of_points_from_wkt_polygon(db, db_det)

    assert set(points) == set([(pt.x, pt.y) for pt in rect1.points])


def test_iou(
    client: Client,
    db: Session,
    rect1: BoundingPolygon,
    rect2: BoundingPolygon,
):
    dataset = client.create_dataset(dset_name)
    model = client.create_model(model_name)

    dataset.add_groundtruth_detections(
        [
            GroundTruthDetection(
                boundary=rect1, labels=[Label("k", "v")], image=Image("uri")
            )
        ]
    )
    db_gt = db.scalar(select(models.GroundTruthDetection))

    model.add_predicted_detections(
        [
            PredictedDetection(
                boundary=rect2,
                scored_labels=[ScoredLabel(label=Label("k", "v"), score=0.6)],
                image=Image("uri"),
            )
        ]
    )
    db_pred = db.scalar(select(models.PredictedDetection))

    assert ops.iou(db, db_gt, db_pred) == iou(rect1, rect2)
