""" These integration tests should be run with a backend at http://localhost:8000
that is no auth
"""
import io
import json
import time
from typing import Any

import numpy as np
import PIL.Image
import pytest
from geoalchemy2.functions import (
    ST_Area,
    ST_AsPNG,
    ST_AsText,
    ST_Intersection,
    ST_Polygon,
    ST_Union,
)
from sqlalchemy import and_, create_engine, func, select, text
from sqlalchemy.orm import Session

from velour.client import Client, ClientException, Dataset, Model
from velour.data_generation import _generate_mask
from velour.enums import DataType, JobStatus, TaskType
from velour.schemas import (
    Annotation,
    BasicPolygon,
    BoundingBox,
    Datum,
    GroundTruth,
    ImageMetadata,
    Label,
    MetaDatum,
    MultiPolygon,
    Point,
    Polygon,
    Prediction,
    Raster,
)
from velour_api import crud, exceptions
from velour_api.backend import jobs, models

dset_name = "test_dataset"
model_name = "test_model"


def bbox_to_poly(bbox: BoundingBox) -> Polygon:
    return Polygon(
        boundary=bbox.polygon,
        holes=None,
    )


def _list_of_points_from_wkt_polygon(
    db: Session, det: models.Annotation
) -> list[Point]:
    geo = json.loads(db.scalar(det.polygon.ST_AsGeoJSON()))
    assert len(geo["coordinates"]) == 1
    return [Point(p[0], p[1]) for p in geo["coordinates"][0][:-1]]


def area(rect: Polygon) -> float:
    """Computes the area of a rectangle"""
    assert len(rect.boundary.points) == 4
    xs = [pt.x for pt in rect.boundary.points]
    ys = [pt.y for pt in rect.boundary.points]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def intersection_area(rect1: Polygon, rect2: Polygon) -> float:
    """Computes the intersection area of two rectangles"""
    assert len(rect1.boundary.points) == len(rect2.boundary.points) == 4

    xs1 = [pt.x for pt in rect1.boundary.points]
    xs2 = [pt.x for pt in rect2.boundary.points]

    ys1 = [pt.y for pt in rect1.boundary.points]
    ys2 = [pt.y for pt in rect2.boundary.points]

    inter_xmin = max(min(xs1), min(xs2))
    inter_xmax = min(max(xs1), max(xs2))

    inter_ymin = max(min(ys1), min(ys2))
    inter_ymax = min(max(ys1), max(ys2))

    inter_width = max(inter_xmax - inter_xmin, 0)
    inter_height = max(inter_ymax - inter_ymin, 0)

    return inter_width * inter_height


def iou(rect1: Polygon, rect2: Polygon) -> float:
    """Computes the "intersection over union" of two rectangles"""
    inter_area = intersection_area(rect1, rect2)
    return inter_area / (area(rect1) + area(rect2) - inter_area)


def random_mask(img: ImageMetadata) -> np.ndarray:
    return np.random.randint(0, 2, size=(img.height, img.width), dtype=bool)


# @TODO: Implement geospatial support
@pytest.fixture
def metadata():
    """Some sample metadata of different types"""
    return [
        # MetaDatum(
        #     key="metadatum name1",
        #     value=GeoJSON(
        #         type="Point",
        #         coordinates=[-48.23456, 20.12345],
        #     )
        # ),
        MetaDatum(key="metadatum1", value="temporary"),
        MetaDatum(key="metadatum2", value="a string"),
        MetaDatum(key="metadatum3", value=0.45),
    ]


@pytest.fixture
def client():
    return Client(host="http://localhost:8000")


@pytest.fixture
def img1() -> ImageMetadata:
    return ImageMetadata(dataset=dset_name, uid="uid1", height=900, width=300)


@pytest.fixture
def img2() -> ImageMetadata:
    return ImageMetadata(dataset=dset_name, uid="uid2", height=40, width=30)


@pytest.fixture
def img5() -> ImageMetadata:
    return ImageMetadata(dataset=dset_name, uid="uid5", height=40, width=30)


@pytest.fixture
def img6() -> ImageMetadata:
    return ImageMetadata(dataset=dset_name, uid="uid6", height=40, width=30)


@pytest.fixture
def img8() -> ImageMetadata:
    return ImageMetadata(dataset=dset_name, uid="uid8", height=40, width=30)


@pytest.fixture
def img9() -> ImageMetadata:
    return ImageMetadata(dataset=dset_name, uid="uid9", height=40, width=30)


@pytest.fixture
def db(client: Client) -> Session:
    """This fixture makes sure there's not datasets, models, or labels in the backend
    (raising a RuntimeError if there are). It returns a db session and as cleanup
    clears out all datasets, models, and labels from the backend.
    """
    if len(client.get_datasets()) > 0:
        raise RuntimeError(
            "Tests should be run on an empty velour backend but found existing datasets.",
            [ds["name"] for ds in client.get_datasets()],
        )

    if len(client.get_models()) > 0:
        raise RuntimeError(
            "Tests should be run on an empty velour backend but found existing models."
        )

    if len(client.get_labels()) > 0:
        raise RuntimeError(
            "Tests should be run on an empty velour backend but found existing labels."
        )

    engine = create_engine("postgresql://postgres:password@localhost/postgres")
    sess = Session(engine)
    sess.execute(text("SET postgis.gdal_enabled_drivers = 'ENABLE_ALL';"))
    sess.execute(text("SET postgis.enable_outdb_rasters = True;"))

    yield sess

    for model in client.get_models():
        try:
            crud.delete(db=sess, model_name=model["name"])
        except exceptions.ModelDoesNotExistError:
            continue

    for dataset in client.get_datasets():
        try:
            crud.delete(db=sess, dataset_name=dataset["name"])
        except exceptions.DatasetDoesNotExistError:
            continue

    labels = sess.scalars(select(models.Label))
    for label in labels:
        sess.delete(label)

    sess.commit()

    # clean redis
    jobs.connect_to_redis()
    jobs.r.flushdb()


@pytest.fixture
def rect1():
    return BoundingBox.from_extrema(xmin=10, ymin=10, xmax=60, ymax=40)


@pytest.fixture
def rect2():
    return BoundingBox.from_extrema(xmin=15, ymin=0, xmax=70, ymax=20)


@pytest.fixture
def rect3():
    return BoundingBox.from_extrema(xmin=87, ymin=10, xmax=158, ymax=820)


@pytest.fixture
def gt_dets1(
    rect1: BoundingBox,
    rect2: BoundingBox,
    rect3: BoundingBox,
    img1: ImageMetadata,
    img2: ImageMetadata,
) -> list[GroundTruth]:
    return [
        GroundTruth(
            datum=img1.to_datum(),
            annotations=[
                Annotation(
                    task_type=TaskType.DETECTION,
                    labels=[Label(key="k1", value="v1")],
                    bounding_box=rect1,
                ),
                Annotation(
                    task_type=TaskType.DETECTION,
                    labels=[Label(key="k2", value="v2")],
                    bounding_box=rect3,
                ),
            ],
        ),
        GroundTruth(
            datum=img2.to_datum(),
            annotations=[
                Annotation(
                    task_type=TaskType.DETECTION,
                    labels=[Label(key="k1", value="v1")],
                    bounding_box=rect2,
                )
            ],
        ),
    ]


@pytest.fixture
def gt_dets2(
    rect1: BoundingBox,
    rect2: BoundingBox,
    rect3: BoundingBox,
    img5: ImageMetadata,
    img6: ImageMetadata,
    img8: ImageMetadata,
) -> list[GroundTruth]:
    return [
        GroundTruth(
            datum=img5.to_datum(),
            annotations=[
                Annotation(
                    task_type=TaskType.DETECTION,
                    labels=[Label(key="k1", value="v1")],
                    polygon=Polygon(boundary=rect1.polygon, holes=None),
                ),
                Annotation(
                    task_type=TaskType.DETECTION,
                    labels=[Label(key="k2", value="v2")],
                    bounding_box=rect3,
                ),
            ],
        ),
        GroundTruth(
            datum=img6.to_datum(),
            annotations=[
                Annotation(
                    task_type=TaskType.DETECTION,
                    labels=[Label(key="k1", value="v1")],
                    polygon=Polygon(boundary=rect2.polygon, holes=None),
                )
            ],
        ),
        GroundTruth(
            datum=img8.to_datum(),
            annotations=[
                Annotation(
                    task_type=TaskType.DETECTION,
                    labels=[Label(key="k3", value="v3")],
                    bounding_box=rect3,
                )
            ],
        ),
    ]


@pytest.fixture
def gt_poly_dets1(
    img1: ImageMetadata,
    img2: ImageMetadata,
    rect1: BoundingBox,
    rect2: BoundingBox,
):
    """Same thing as gt_dets1 but represented as a polygon instead of bounding box"""
    return [
        GroundTruth(
            datum=img1.to_datum(),
            annotations=[
                Annotation(
                    task_type=TaskType.DETECTION,
                    labels=[Label(key="k1", value="v1")],
                    polygon=Polygon(boundary=rect1.polygon, holes=None),
                ),
            ],
        ),
        GroundTruth(
            datum=img2.to_datum(),
            annotations=[
                Annotation(
                    task_type=TaskType.DETECTION,
                    labels=[Label(key="k1", value="v1")],
                    polygon=Polygon(boundary=rect2.polygon, holes=None),
                )
            ],
        ),
    ]


@pytest.fixture
def gt_segs(
    rect1: BoundingBox,
    rect2: BoundingBox,
    rect3: BoundingBox,
    img1: ImageMetadata,
    img2: ImageMetadata,
) -> list[Annotation]:
    return [
        GroundTruth(
            datum=img1.to_datum(),
            annotations=[
                Annotation(
                    task_type=TaskType.INSTANCE_SEGMENTATION,
                    labels=[Label(key="k1", value="v1")],
                    multipolygon=MultiPolygon(
                        polygons=[Polygon(boundary=rect1.polygon)]
                    ),
                ),
                Annotation(
                    task_type=TaskType.SEMANTIC_SEGMENTATION,
                    labels=[Label(key="k2", value="v2")],
                    multipolygon=MultiPolygon(
                        polygons=[
                            Polygon(boundary=rect3.polygon),
                            Polygon(boundary=rect1.polygon),
                        ]
                    ),
                ),
            ],
        ),
        GroundTruth(
            datum=img2.to_datum(),
            annotations=[
                Annotation(
                    task_type=TaskType.INSTANCE_SEGMENTATION,
                    labels=[Label(key="k1", value="v1")],
                    multipolygon=MultiPolygon(
                        polygons=[
                            Polygon(
                                boundary=rect2.polygon,
                                holes=[rect1.polygon],
                            )
                        ]
                    ),
                )
            ],
        ),
    ]


@pytest.fixture
def gt_semantic_segs1(
    rect1: BoundingBox, rect3: BoundingBox, img1: ImageMetadata
) -> list[GroundTruth]:
    return [
        GroundTruth(
            datum=img1.to_datum(),
            annotations=[
                Annotation(
                    task_type=TaskType.SEMANTIC_SEGMENTATION,
                    labels=[Label(key="k2", value="v2")],
                    multipolygon=MultiPolygon(
                        polygons=[
                            Polygon(boundary=rect3.polygon),
                            Polygon(boundary=rect1.polygon),
                        ]
                    ),
                )
            ],
        ),
    ]


@pytest.fixture
def gt_semantic_segs1_mask(img1: ImageMetadata) -> GroundTruth:
    mask = _generate_mask(height=900, width=300)
    raster = Raster.from_numpy(mask)

    return GroundTruth(
        datum=img1.to_datum(),
        annotations=[
            Annotation(
                task_type=TaskType.SEMANTIC_SEGMENTATION,
                labels=[Label(key="k2", value="v2")],
                raster=raster,
            )
        ],
    )


@pytest.fixture
def gt_semantic_segs2(rect3: BoundingBox, img2: ImageMetadata) -> GroundTruth:
    return [
        GroundTruth(
            datum=img2.to_datum(),
            annotations=[
                Annotation(
                    task_type=TaskType.SEMANTIC_SEGMENTATION,
                    labels=[Label(key="k3", value="v3")],
                    multipolygon=MultiPolygon(
                        polygons=[Polygon(boundary=rect3.polygon)],
                    ),
                )
            ],
        ),
    ]


@pytest.fixture
def gt_semantic_segs2_mask(img2: ImageMetadata) -> GroundTruth:
    mask = _generate_mask(height=40, width=30)
    raster = Raster.from_numpy(mask)

    return GroundTruth(
        datum=img2.to_datum(),
        annotations=[
            Annotation(
                task_type=TaskType.SEMANTIC_SEGMENTATION,
                labels=[Label(key="k2", value="v2")],
                raster=raster,
            )
        ],
    )


@pytest.fixture
def gt_semantic_segs_error(img1: ImageMetadata) -> GroundTruth:
    mask = _generate_mask(height=100, width=100)
    raster = Raster.from_numpy(mask)

    # expected to throw an error since the mask size differs from the image size
    return GroundTruth(
        datum=img1.to_datum(),
        annotations=[
            Annotation(
                task_type=TaskType.SEMANTIC_SEGMENTATION,
                labels=[Label(key="k3", value="v3")],
                raster=raster,
            )
        ],
    )


@pytest.fixture
def gt_clfs(
    img5: ImageMetadata,
    img6: ImageMetadata,
    img8: ImageMetadata,
) -> list[GroundTruth]:
    return [
        GroundTruth(
            datum=img5.to_datum(),
            annotations=[
                Annotation(
                    task_type=TaskType.CLASSIFICATION,
                    labels=[
                        Label(key="k4", value="v4"),
                        Label(key="k5", value="v5"),
                    ],
                ),
            ],
        ),
        GroundTruth(
            datum=img6.to_datum(),
            annotations=[
                Annotation(
                    task_type=TaskType.CLASSIFICATION,
                    labels=[Label(key="k4", value="v4")],
                )
            ],
        ),
        GroundTruth(
            datum=img8.to_datum(),
            annotations=[
                Annotation(
                    task_type=TaskType.CLASSIFICATION,
                    labels=[Label(key="k3", value="v3")],
                )
            ],
        ),
    ]


@pytest.fixture
def pred_dets(
    rect1: BoundingBox,
    rect2: BoundingBox,
    img1: ImageMetadata,
    img2: ImageMetadata,
) -> list[Prediction]:
    return [
        Prediction(
            model=model_name,
            datum=img1.to_datum(),
            annotations=[
                Annotation(
                    task_type=TaskType.DETECTION,
                    labels=[Label(key="k1", value="v1", score=0.3)],
                    bounding_box=rect1,
                )
            ],
        ),
        Prediction(
            model=model_name,
            datum=img2.to_datum(),
            annotations=[
                Annotation(
                    task_type=TaskType.DETECTION,
                    labels=[Label(key="k2", value="v2", score=0.98)],
                    bounding_box=rect2,
                )
            ],
        ),
    ]


@pytest.fixture
def pred_poly_dets(
    pred_dets: list[Prediction],
) -> list[Prediction]:
    return [
        Prediction(
            model=det.model,
            datum=det.datum,
            annotations=[
                Annotation(
                    task_type=TaskType.DETECTION,
                    labels=annotation.labels,
                    polygon=bbox_to_poly(annotation.bounding_box),
                )
                for annotation in det.annotations
            ],
        )
        for det in pred_dets
    ]


@pytest.fixture
def pred_instance_segs(
    img1: ImageMetadata, img2: ImageMetadata
) -> list[Prediction]:
    mask_1 = random_mask(img1)
    mask_2 = random_mask(img2)
    return [
        Prediction(
            model=model_name,
            datum=img1.to_datum(),
            annotations=[
                Annotation(
                    task_type=TaskType.INSTANCE_SEGMENTATION,
                    labels=[Label(key="k1", value="v1", score=0.87)],
                    raster=Raster.from_numpy(mask_1),
                )
            ],
        ),
        Prediction(
            model=model_name,
            datum=img2.to_datum(),
            annotations=[
                Annotation(
                    task_type=TaskType.INSTANCE_SEGMENTATION,
                    labels=[Label(key="k2", value="v2", score=0.92)],
                    raster=Raster.from_numpy(mask_2),
                )
            ],
        ),
    ]


@pytest.fixture
def pred_semantic_segs(
    img1: ImageMetadata, img2: ImageMetadata
) -> list[Prediction]:
    mask_1 = random_mask(img1)
    mask_2 = random_mask(img2)
    return [
        Prediction(
            model=model_name,
            datum=img1.to_datum(),
            annotations=[
                Annotation(
                    task_type=TaskType.SEMANTIC_SEGMENTATION,
                    labels=[Label(key="k2", value="v2")],
                    raster=Raster.from_numpy(mask_1),
                )
            ],
        ),
        Prediction(
            model=model_name,
            datum=img2.to_datum(),
            annotations=[
                Annotation(
                    task_type=TaskType.SEMANTIC_SEGMENTATION,
                    labels=[Label(key="k1", value="v1")],
                    raster=Raster.from_numpy(mask_2),
                )
            ],
        ),
    ]


@pytest.fixture
def pred_clfs(img5: ImageMetadata, img6: ImageMetadata) -> list[Prediction]:
    return [
        Prediction(
            model=model_name,
            datum=img5.to_datum(),
            annotations=[
                Annotation(
                    task_type=TaskType.CLASSIFICATION,
                    labels=[
                        Label(key="k12", value="v12", score=0.47),
                        Label(key="k12", value="v16", score=0.53),
                        Label(key="k13", value="v13", score=1.0),
                    ],
                )
            ],
        ),
        Prediction(
            model=model_name,
            datum=img6.to_datum(),
            annotations=[
                Annotation(
                    task_type=TaskType.CLASSIFICATION,
                    labels=[
                        Label(key="k4", value="v4", score=0.71),
                        Label(key="k4", value="v5", score=0.29),
                    ],
                )
            ],
        ),
    ]


@pytest.fixture
def y_true() -> list[int]:
    """groundtruth for a tabular classification task"""
    return [1, 1, 2, 0, 0, 0, 1, 1, 1, 1]


@pytest.fixture
def tabular_preds() -> list[list[float]]:
    """predictions for a tabular classification task"""
    return [
        [0.37, 0.35, 0.28],
        [0.24, 0.61, 0.15],
        [0.03, 0.88, 0.09],
        [0.97, 0.03, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.01, 0.96, 0.03],
        [0.28, 0.02, 0.7],
        [0.78, 0.21, 0.01],
        [0.45, 0.11, 0.44],
    ]


def _test_create_image_dataset_with_gts(
    client: Client,
    gts: list[Any],
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
    gts
        list of groundtruth objects (from `velour.data_types`)
    expected_labels_tuples
        set of tuples of key/value labels to check were added to the database
    expected_image_uids
        set of image uids to check were added to the database
    """

    dataset = Dataset.create(client, dset_name)

    with pytest.raises(ClientException) as exc_info:
        Dataset.create(client, dset_name)
    assert "already exists" in str(exc_info)

    for gt in gts:
        dataset.add_groundtruth(gt)
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
        dataset.add_groundtruth(gts[0])
    assert "has been finalized" in str(exc_info)

    return dataset


def _test_create_model_with_preds(
    client: Client,
    datum_type: DataType,
    gts: list[Any],
    preds: list[Any],
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
    dataset = Dataset.create(client, dset_name)
    model = Model.create(client, model_name)

    # verify we get an error if we try to create another model
    # with the same name
    with pytest.raises(ClientException) as exc_info:
        Model.create(client, model_name)
    assert "already exists" in str(exc_info)

    # add groundtruths
    for gt in gts:
        dataset.add_groundtruth(gt)

    # finalize dataset
    dataset.finalize()

    # add predictions
    for pd in preds:
        model.add_prediction(pd)

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
    retrieved_model = Model.get(client, model_name)
    assert isinstance(retrieved_model, type(model))
    assert retrieved_model.name == model_name

    return db_preds


def test_create_image_dataset_with_href_and_description(
    client: Client, db: Session
):
    href = "http://a.com/b"
    description = "a description"
    Dataset.create(client, dset_name, href=href, description=description)

    dataset_id = db.scalar(
        select(models.Dataset.id).where(models.Dataset.name == dset_name)
    )
    assert href == db.scalar(
        select(models.MetaDatum.string_value).where(
            and_(
                models.MetaDatum.dataset_id == dataset_id,
                models.MetaDatum.key == "href",
            )
        )
    )
    assert description == db.scalar(
        select(models.MetaDatum.string_value).where(
            and_(
                models.MetaDatum.dataset_id == dataset_id,
                models.MetaDatum.key == "description",
            )
        )
    )


def test_create_model_with_href_and_description(client: Client, db: Session):
    href = "http://a.com/b"
    description = "a description"
    Model.create(client, model_name, href=href, description=description)

    model_id = db.scalar(
        select(models.Model.id).where(models.Model.name == model_name)
    )
    assert href == db.scalar(
        select(models.MetaDatum.string_value).where(
            and_(
                models.MetaDatum.model_id == model_id,
                models.MetaDatum.key == "href",
            )
        )
    )
    assert description == db.scalar(
        select(models.MetaDatum.string_value).where(
            and_(
                models.MetaDatum.model_id == model_id,
                models.MetaDatum.key == "description",
            )
        )
    )


def test_create_image_dataset_with_detections(
    client: Client,
    gt_dets1: list[GroundTruth],
    gt_dets2: list[GroundTruth],
    db: Session,  # this is unused but putting it here since the teardown of the fixture does cleanup
):
    dataset = _test_create_image_dataset_with_gts(
        client=client,
        gts=gt_dets1 + gt_dets2,
        expected_image_uids={"uid2", "uid8", "uid1", "uid6", "uid5"},
        expected_labels_tuples={
            ("k1", "v1"),
            ("k2", "v2"),
            ("k3", "v3"),
        },
    )

    dets1 = dataset.get_groundtruth("uid1")
    dets2 = dataset.get_groundtruth("uid2")

    # check we get back what we inserted
    gt_dets_uid1 = []
    gt_dets_uid2 = []
    for gt in gt_dets1 + gt_dets2:
        if gt.datum.uid == "uid1":
            gt_dets_uid1.extend(gt.annotations)
        elif gt.datum.uid == "uid2":
            gt_dets_uid2.extend(gt.annotations)
    assert dets1.annotations == gt_dets_uid1
    assert dets2.annotations == gt_dets_uid2


def test_create_image_model_with_predicted_detections(
    client: Client,
    gt_poly_dets1: list[GroundTruth],
    pred_poly_dets: list[Prediction],
    db: Session,
):
    labeled_pred_dets = _test_create_model_with_preds(
        client=client,
        datum_type=DataType.IMAGE,
        gts=gt_poly_dets1,
        preds=pred_poly_dets,
        preds_model_class=models.Prediction,
        preds_expected_number=2,
        expected_labels_tuples={("k1", "v1"), ("k2", "v2")},
        expected_scores={0.3, 0.98},
        db=db,
    )

    # get db polygon
    db_annotation_ids = {pred.annotation_id for pred in labeled_pred_dets}
    db_annotations = [
        db.scalar(
            select(models.Annotation).where(
                and_(
                    models.Annotation.id == id,
                    models.Annotation.model_id.isnot(None),
                )
            )
        )
        for id in db_annotation_ids
    ]
    db_point_lists = [
        _list_of_points_from_wkt_polygon(db, annotation)
        for annotation in db_annotations
    ]

    # get fixture polygons
    fx_point_lists = []
    for pd in pred_poly_dets:
        for ann in pd.annotations:
            fx_point_lists.append(ann.polygon.boundary.points)

    # check boundary
    for fx_points in fx_point_lists:
        assert fx_points in db_point_lists


def test_create_gt_detections_as_bbox_or_poly(db: Session, client: Client):
    """Test that a groundtruth detection can be created as either a bounding box
    or a polygon
    """
    xmin, ymin, xmax, ymax = 10, 25, 30, 50
    image = ImageMetadata(
        dataset=dset_name, uid="uid", height=200, width=150
    ).to_datum()

    dataset = Dataset.create(client, dset_name)
    gt = GroundTruth(
        datum=image,
        annotations=[
            Annotation(
                task_type=TaskType.DETECTION,
                labels=[Label(key="k", value="v")],
                bounding_box=BoundingBox.from_extrema(
                    xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax
                ),
            ),
            Annotation(
                task_type=TaskType.DETECTION,
                labels=[Label(key="k", value="v")],
                polygon=Polygon(
                    boundary=BasicPolygon(
                        points=[
                            Point(x=xmin, y=ymin),
                            Point(x=xmax, y=ymin),
                            Point(x=xmax, y=ymax),
                            Point(x=xmin, y=ymax),
                        ]
                    ),
                    holes=None,
                ),
            ),
        ],
    )
    dataset.add_groundtruth(gt)

    db_dets = db.scalars(
        select(models.Annotation).where(models.Annotation.model_id.is_(None))
    ).all()
    assert len(db_dets) == 2
    assert set([db_det.box is not None for db_det in db_dets]) == {True, False}

    assert (
        str(db.scalar(ST_AsText(db_dets[0].box)))
        == "POLYGON((10 25,30 25,30 50,10 50,10 25))"
        == str(db.scalar(ST_AsText(db_dets[1].polygon)))
    )

    # check that they can be recovered by the client
    detections = dataset.get_groundtruth("uid")
    assert len(detections.annotations) == 2
    assert (
        len(
            [
                det
                for det in detections.annotations
                if det.bounding_box is not None
            ]
        )
        == 1
    )
    for det in detections.annotations:
        if det.bounding_box:
            assert det == gt.annotations[0]
        else:
            assert det == gt.annotations[1]


def test_create_pred_detections_as_bbox_or_poly(
    db: Session,
    client: Client,
    gt_dets1: list[Annotation],
    img1: ImageMetadata,
):
    """Test that a predicted detection can be created as either a bounding box
    or a polygon
    """
    xmin, ymin, xmax, ymax = 10, 25, 30, 50

    dataset = Dataset.create(client, dset_name)
    for gt in gt_dets1:
        dataset.add_groundtruth(gt)
    dataset.finalize()

    model = Model.create(client, model_name)
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


def test_create_image_dataset_with_segmentations(
    client: Client,
    gt_segs: list[GroundTruth],
    db: Session,  # this is unused but putting it here since the teardown of the fixture does cleanup
):
    dataset = _test_create_image_dataset_with_gts(
        client=client,
        gts=gt_segs,
        expected_image_uids={"uid1", "uid2"},
        expected_labels_tuples={("k1", "v1"), ("k2", "v2")},
    )

    gt = dataset.get_groundtruth("uid1")
    image = ImageMetadata.from_datum(gt.datum)
    segs = gt.annotations

    instance_segs = []
    semantic_segs = []
    for seg in segs:
        assert isinstance(seg, Annotation)
        if seg.task_type == TaskType.INSTANCE_SEGMENTATION:
            instance_segs.append(seg)
        elif seg.task_type == TaskType.SEMANTIC_SEGMENTATION:
            semantic_segs.append(seg)

    # should have one instance segmentation that's a rectangle
    # with xmin, ymin, xmax, ymax = 10, 10, 60, 40
    assert len(instance_segs) == 1
    mask = instance_segs[0].raster.to_numpy()
    # check get all True in the box
    assert mask[10:40, 10:60].all()
    # check that outside the box is all False
    assert mask.sum() == (40 - 10) * (60 - 10)
    # check shape agrees with image
    assert mask.shape == (image.height, image.width)

    # should have one semantic segmentation that's a rectangle
    # with xmin, ymin, xmax, ymax = 10, 10, 60, 40 plus a rectangle
    # with xmin, ymin, xmax, ymax = 87, 10, 158, 820
    assert len(semantic_segs) == 1
    mask = semantic_segs[0].raster.to_numpy()
    assert mask[10:40, 10:60].all()
    assert mask[10:820, 87:158].all()
    assert mask.sum() == (40 - 10) * (60 - 10) + (820 - 10) * (158 - 87)
    assert mask.shape == (image.height, image.width)


def test_create_gt_segs_as_polys_or_masks(
    client: Client, img1: ImageMetadata, db: Session
):
    """Test that we can create a dataset with groundtruth segmentations that are defined
    both my polygons and mask arrays
    """
    xmin, xmax, ymin, ymax = 11, 45, 37, 102
    h, w = img1.height, img1.width
    mask = np.zeros((h, w), dtype=bool)
    mask[ymin:ymax, xmin:xmax] = True

    poly = Polygon(
        boundary=BasicPolygon(
            points=[
                Point(x=xmin, y=ymin),
                Point(x=xmin, y=ymax),
                Point(x=xmax, y=ymax),
                Point(x=xmax, y=ymin),
            ]
        )
    )

    dataset = Dataset.create(client, dset_name)

    # check we get an error for adding semantic segmentation with duplicate labels
    with pytest.raises(ClientException) as exc_info:
        gts = GroundTruth(
            datum=img1.to_datum(),
            annotations=[
                Annotation(
                    task_type=TaskType.SEMANTIC_SEGMENTATION,
                    labels=[Label(key="k1", value="v1")],
                    raster=Raster.from_numpy(mask),
                ),
                Annotation(
                    task_type=TaskType.SEMANTIC_SEGMENTATION,
                    labels=[Label(key="k1", value="v1")],
                    multipolygon=MultiPolygon(polygons=[poly]),
                ),
            ],
        )

        dataset.add_groundtruth(gts)

    assert (
        "semantic segmentation tasks can only have one annotation per label"
        in str(exc_info.value)
    )

    # fine with instance segmentation though
    gts = GroundTruth(
        datum=img1.to_datum(),
        annotations=[
            Annotation(
                task_type=TaskType.SEMANTIC_SEGMENTATION,
                labels=[Label(key="k1", value="v1")],
                raster=Raster.from_numpy(mask),
            ),
            Annotation(
                task_type=TaskType.INSTANCE_SEGMENTATION,
                labels=[Label(key="k1", value="v1")],
                multipolygon=MultiPolygon(polygons=[poly]),
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


def test_create_model_with_predicted_segmentations(
    client: Client,
    gt_segs: list[GroundTruth],
    pred_instance_segs: list[Prediction],
    db: Session,
):
    """Tests that we can create a predicted segmentation from a mask array"""
    _test_create_model_with_preds(
        client=client,
        datum_type=DataType.IMAGE,
        gts=gt_segs,
        preds=pred_instance_segs,
        preds_model_class=models.Prediction,
        preds_expected_number=2,
        expected_labels_tuples={("k1", "v1"), ("k2", "v2")},
        expected_scores={0.87, 0.92},
        db=db,
    )

    # grab the segmentation from the db, recover the mask, and check
    # its equal to the mask the client sent over
    db_annotations = (
        db.query(models.Annotation)
        .where(models.Annotation.model_id.isnot(None))
        .all()
    )

    if db_annotations[0].datum_id < db_annotations[1].datum_id:
        raster_uid1 = db_annotations[0].raster
        raster_uid2 = db_annotations[1].raster
    else:
        raster_uid1 = db_annotations[1].raster
        raster_uid2 = db_annotations[0].raster

    # test raster 1
    png_from_db = db.scalar(ST_AsPNG(raster_uid1))
    f = io.BytesIO(png_from_db.tobytes())
    mask_array = np.array(PIL.Image.open(f))
    np.testing.assert_equal(
        mask_array, pred_instance_segs[0].annotations[0].raster.to_numpy()
    )

    # test raster 2
    png_from_db = db.scalar(ST_AsPNG(raster_uid2))
    f = io.BytesIO(png_from_db.tobytes())
    mask_array = np.array(PIL.Image.open(f))
    np.testing.assert_equal(
        mask_array, pred_instance_segs[1].annotations[0].raster.to_numpy()
    )


def test_create_image_dataset_with_classifications(
    client: Client,
    gt_clfs: list[GroundTruth],
    db: Session,  # this is unused but putting it here since the teardown of the fixture does cleanup
):
    _test_create_image_dataset_with_gts(
        client=client,
        gts=gt_clfs,
        expected_image_uids={"uid5", "uid6", "uid8"},
        expected_labels_tuples={
            ("k5", "v5"),
            ("k4", "v4"),
            ("k3", "v3"),
        },
    )


def test_create_image_model_with_predicted_classifications(
    client: Client,
    gt_clfs: list[GroundTruth],
    pred_clfs: list[Prediction],
    db: Session,
):
    _test_create_model_with_preds(
        client=client,
        datum_type=DataType.IMAGE,
        gts=gt_clfs,
        preds=pred_clfs,
        preds_model_class=models.Prediction,
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
    client: Client, db: Session, rect1: Polygon, img1: ImageMetadata
):
    """Test consistency of boundary in backend and client"""
    dataset = Dataset.create(client, dset_name)
    rect1_poly = bbox_to_poly(rect1)
    dataset.add_groundtruth(
        GroundTruth(
            datum=img1.to_datum(),
            annotations=[
                Annotation(
                    task_type=TaskType.DETECTION,
                    labels=[Label(key="k1", value="v1")],
                    polygon=rect1_poly,
                )
            ],
        )
    )

    # get the one detection that exists
    db_det = db.scalar(select(models.Annotation))

    # check boundary
    points = _list_of_points_from_wkt_polygon(db, db_det)

    assert set(points) == set([pt for pt in rect1_poly.boundary.points])


def test_iou(
    client: Client,
    db: Session,
    rect1: Polygon,
    rect2: Polygon,
    img1: ImageMetadata,
):
    rect1_poly = bbox_to_poly(rect1)
    rect2_poly = bbox_to_poly(rect2)

    dataset = Dataset.create(client, dset_name)
    dataset.add_groundtruth(
        GroundTruth(
            datum=img1.to_datum(),
            annotations=[
                Annotation(
                    task_type=TaskType.DETECTION,
                    labels=[Label("k", "v")],
                    polygon=rect1_poly,
                )
            ],
        )
    )
    dataset.finalize()
    db_gt = db.scalar(select(models.Annotation)).polygon

    model = Model.create(client, model_name)
    model.add_prediction(
        Prediction(
            model=model_name,
            datum=img1.to_datum(),
            annotations=[
                Annotation(
                    task_type=TaskType.DETECTION,
                    polygon=rect2_poly,
                    labels=[Label("k", "v", score=0.6)],
                )
            ],
        )
    )
    model.finalize_inferences(dataset)
    db_pred = db.scalar(
        select(models.Annotation).where(models.Annotation.model_id.isnot(None))
    ).polygon

    # scraped from velour_api backend
    gintersection = ST_Intersection(db_gt, db_pred)
    gunion = ST_Union(db_gt, db_pred)
    iou_computation = ST_Area(gintersection) / ST_Area(gunion)

    assert iou(rect1_poly, rect2_poly) == db.scalar(select(iou_computation))


def test_client_delete_dataset(client: Client, db: Session):
    """test that delete dataset returns a job whose status changes from "Processing" to "Done" """
    Dataset.create(client, dset_name)
    assert db.scalar(select(func.count(models.Dataset.name))) == 1
    client.delete_dataset(dset_name, timeout=30)
    time.sleep(1.0)
    assert db.scalar(select(func.count(models.Dataset.name))) == 0


def test_client_delete_model(client: Client, db: Session):
    """test that delete dataset returns a job whose status changes from "Processing" to "Done" """
    Model.create(client, model_name)
    assert db.scalar(select(func.count(models.Model.name))) == 1
    client.delete_model(model_name)
    time.sleep(1.0)
    assert db.scalar(select(func.count(models.Model.name))) == 0


def test_evaluate_ap(
    client: Client,
    gt_dets1: list[GroundTruth],
    pred_dets: list[Prediction],
    db: Session,
):
    dataset = Dataset.create(client, dset_name)
    for gt in gt_dets1:
        dataset.add_groundtruth(gt)
    dataset.finalize()

    model = Model.create(client, model_name)
    for pd in pred_dets:
        model.add_prediction(pd)
    model.finalize_inferences(dataset)

    eval_job = model.evaluate_ap(
        dataset=dataset,
        iou_thresholds=[0.1, 0.6],
        ious_to_keep=[0.1, 0.6],
        label_key="k1",
    )

    assert eval_job.ignored_pred_labels == []
    assert eval_job.missing_pred_labels == []
    assert isinstance(eval_job._id, int)

    eval_job.wait_for_completion()
    assert eval_job.status == JobStatus.DONE

    settings = eval_job.settings
    settings.pop("id")
    assert settings == {
        "model": "test_model",
        "dataset": "test_dataset",
        "task_type": "detection",
        "target_type": "box",
        "label_key": "k1",
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

    assert eval_job.metrics == expected_metrics

    # now test if we set min_area and/or max_area
    areas = db.scalars(
        select(ST_Area(models.Annotation.box)).where(
            models.Annotation.model_id.isnot(None)
        )
    ).all()
    assert sorted(areas) == [1100.0, 1500.0]

    # sanity check this should give us the same thing excpet min_area and max_area
    # are not None
    eval_job_bounded_area_10_2000 = model.evaluate_ap(
        dataset=dataset,
        task_type="detection",
        iou_thresholds=[0.1, 0.6],
        ious_to_keep=[0.1, 0.6],
        label_key="k1",
        min_area=10,
        max_area=2000,
    )
    time.sleep(1)
    settings = eval_job_bounded_area_10_2000.settings
    settings.pop("id")
    assert settings == {
        "model": "test_model",
        "dataset": "test_dataset",
        "task_type": "detection",
        "target_type": "box",
        "label_key": "k1",
        "min_area": 10,
        "max_area": 2000,
    }
    assert eval_job_bounded_area_10_2000.metrics == expected_metrics

    # now check we get different things by setting the thresholds accordingly
    # min area threshold should divide the set of annotations
    eval_job_min_area_1200 = model.evaluate_ap(
        dataset=dataset,
        task_type="detection",
        iou_thresholds=[0.1, 0.6],
        ious_to_keep=[0.1, 0.6],
        label_key="k1",
        min_area=1200,
    )
    time.sleep(1)
    settings = eval_job_min_area_1200.settings
    settings.pop("id")
    assert settings == {
        "model": "test_model",
        "dataset": "test_dataset",
        "task_type": "detection",
        "target_type": "box",
        "label_key": "k1",
        "min_area": 1200,
    }
    assert eval_job_min_area_1200.metrics != expected_metrics

    # check for difference with max area now dividing the set of annotations
    eval_job_max_area_1200 = model.evaluate_ap(
        dataset=dataset,
        task_type="detection",
        iou_thresholds=[0.1, 0.6],
        ious_to_keep=[0.1, 0.6],
        label_key="k1",
        max_area=1200,
    )
    time.sleep(1)
    settings = eval_job_max_area_1200.settings
    settings.pop("id")
    assert settings == {
        "model": "test_model",
        "dataset": "test_dataset",
        "task_type": "detection",
        "target_type": "box",
        "label_key": "k1",
        "max_area": 1200,
    }
    assert eval_job_max_area_1200.metrics != expected_metrics

    # should perform the same as the first min area evaluation
    # except now has an upper bound
    eval_job_bounded_area_1200_1800 = model.evaluate_ap(
        dataset=dataset,
        task_type="detection",
        iou_thresholds=[0.1, 0.6],
        ious_to_keep=[0.1, 0.6],
        label_key="k1",
        min_area=1200,
        max_area=1800,
    )
    time.sleep(1)
    settings = eval_job_bounded_area_1200_1800.settings
    settings.pop("id")
    assert settings == {
        "model": "test_model",
        "dataset": "test_dataset",
        "task_type": "detection",
        "target_type": "box",
        "label_key": "k1",
        "min_area": 1200,
        "max_area": 1800,
    }
    assert eval_job_bounded_area_1200_1800.metrics != expected_metrics
    assert (
        eval_job_bounded_area_1200_1800.metrics
        == eval_job_min_area_1200.metrics
    )


def test_evaluate_image_clf(
    client: Client,
    gt_clfs: list[GroundTruth],
    pred_clfs: list[Prediction],
    db: Session,  # this is unused but putting it here since the teardown of the fixture does cleanup
):
    dataset = Dataset.create(client, dset_name)
    for gt in gt_clfs:
        dataset.add_groundtruth(gt)
    dataset.finalize()

    model = Model.create(client, model_name)
    for pd in pred_clfs:
        model.add_prediction(pd)
    model.finalize_inferences(dataset)

    eval_job = model.evaluate_classification(dataset=dataset)

    assert set(eval_job.ignored_pred_keys) == {"k12", "k13"}
    assert set(eval_job.missing_pred_keys) == {"k3", "k5"}

    # sleep to give the backend time to compute
    time.sleep(1)
    assert eval_job.status == JobStatus.DONE

    metrics = eval_job.metrics

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
        {
            "type": "Precision",
            "value": -1.0,
            "label": {"key": "k4", "value": "v5"},
        },
        {
            "type": "Recall",
            "value": -1.0,
            "label": {"key": "k4", "value": "v5"},
        },
        {"type": "F1", "value": -1.0, "label": {"key": "k4", "value": "v5"}},
    ]
    for m in metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in metrics

    confusion_matrices = eval_job.confusion_matrices
    assert confusion_matrices == [
        {
            "label_key": "k4",
            "entries": [{"prediction": "v4", "groundtruth": "v4", "count": 1}],
        }
    ]


def test_evaluate_semantic_segmentation(
    client: Client,
    db: Session,
    gt_semantic_segs1: list[GroundTruth],
    gt_semantic_segs2: list[GroundTruth],
    pred_semantic_segs: list[Prediction],
):
    dataset = Dataset.create(client, dset_name)
    model = Model.create(client, model_name)

    for gt in gt_semantic_segs1 + gt_semantic_segs2:
        dataset.add_groundtruth(gt)

    for pred in pred_semantic_segs:
        model.add_prediction(pred)

    dataset.finalize()
    model.finalize_inferences(dataset)

    eval_job = model.evaluate_semantic_segmentation(dataset)
    assert eval_job.missing_pred_labels == [
        {"key": "k3", "value": "v3", "score": None}
    ]
    assert eval_job.ignored_pred_labels == [
        {"key": "k1", "value": "v1", "score": None}
    ]

    time.sleep(1)
    metrics = eval_job.metrics

    assert len(metrics) == 3
    assert set(
        [
            (m["label"]["key"], m["label"]["value"])
            for m in metrics
            if "label" in m
        ]
    ) == {("k2", "v2"), ("k3", "v3")}
    assert set([m["type"] for m in metrics]) == {"IOU", "mIOU"}


def test_create_tabular_dataset_and_add_groundtruth(
    client: Client, db: Session, metadata: list[MetaDatum]
):
    dataset = Dataset.create(client, name=dset_name)
    assert isinstance(dataset, Dataset)

    md1, md2, md3 = metadata

    gts = [
        GroundTruth(
            datum=Datum(
                dataset=dset_name,
                uid="uid1",
                metadata=[md1],
            ),
            annotations=[
                Annotation(
                    task_type=TaskType.CLASSIFICATION,
                    labels=[
                        Label(key="k1", value="v1"),
                        Label(key="k2", value="v2"),
                    ],
                )
            ],
        ),
        GroundTruth(
            datum=Datum(
                dataset=dset_name,
                uid="uid2",
                metadata=[md2, md3],
            ),
            annotations=[
                Annotation(
                    task_type=TaskType.CLASSIFICATION,
                    labels=[Label(key="k1", value="v3")],
                )
            ],
        ),
    ]

    for gt in gts:
        dataset.add_groundtruth(gt)

    assert len(db.scalars(select(models.GroundTruth)).all()) == 3
    # check we have two Datums and they have the correct uids
    data = db.scalars(select(models.Datum)).all()
    assert len(data) == 2
    assert set(d.uid for d in data) == {"uid1", "uid2"}

    # check metadata is there
    metadata_links = data[0].metadatums
    assert len(metadata_links) == 1
    metadatum = data[0].metadatums[0]
    assert metadatum.key == "metadatum1"
    assert metadatum.string_value == "temporary"
    # assert json.loads(db.scalar(ST_AsGeoJSON(metadatum.geo))) == {
    #     "type": "Point",
    #     "coordinates": [-48.23456, 20.12345],
    # }

    metadata_links = data[1].metadatums
    assert len(metadata_links) == 2
    metadatum1 = metadata_links[0]
    metadatum2 = metadata_links[1]
    assert metadatum1.key == "metadatum2"
    assert metadatum1.string_value == "a string"
    assert metadatum2.key == "metadatum3"
    assert metadatum2.numeric_value == 0.45

    # check that we can add data with specified uids
    new_gts = [
        GroundTruth(
            datum=Datum(dataset=dset_name, uid="uid3"),
            annotations=[
                Annotation(
                    task_type=TaskType.CLASSIFICATION,
                    labels=[Label(key="k1", value="v1")],
                )
            ],
        ),
        GroundTruth(
            datum=Datum(dataset=dset_name, uid="uid4"),
            annotations=[
                Annotation(
                    task_type=TaskType.CLASSIFICATION,
                    labels=[Label(key="k1", value="v5")],
                )
            ],
        ),
    ]
    for gt in new_gts:
        dataset.add_groundtruth(gt)

    assert len(db.scalars(select(models.GroundTruth)).all()) == 5
    # check we have two Datums and they have the correct uids
    data = db.scalars(select(models.Datum)).all()
    assert len(data) == 4
    assert set(d.uid for d in data) == {"uid1", "uid2", "uid3", "uid4"}


def test_create_tabular_model_with_predicted_classifications(
    client: Client,
    db: Session,
):
    _test_create_model_with_preds(
        client=client,
        datum_type=DataType.TABULAR,
        gts=[
            GroundTruth(
                datum=Datum(dataset=dset_name, uid="uid1"),
                annotations=[
                    Annotation(
                        task_type=TaskType.CLASSIFICATION,
                        labels=[
                            Label(key="k1", value="v1"),
                            Label(key="k2", value="v2"),
                        ],
                    )
                ],
            ),
            GroundTruth(
                datum=Datum(
                    dataset=dset_name,
                    uid="uid2",
                ),
                annotations=[
                    Annotation(
                        task_type=TaskType.CLASSIFICATION,
                        labels=[Label(key="k1", value="v3")],
                    )
                ],
            ),
        ],
        preds=[
            Prediction(
                model=model_name,
                datum=Datum(dataset=dset_name, uid="uid1"),
                annotations=[
                    Annotation(
                        task_type=TaskType.CLASSIFICATION,
                        labels=[
                            Label(key="k1", value="v1", score=0.6),
                            Label(key="k1", value="v2", score=0.4),
                            Label(key="k2", value="v6", score=1.0),
                        ],
                    )
                ],
            ),
            Prediction(
                model=model_name,
                datum=Datum(
                    dataset=dset_name,
                    uid="uid2",
                ),
                annotations=[
                    Annotation(
                        task_type=TaskType.CLASSIFICATION,
                        labels=[
                            Label(key="k1", value="v1", score=0.1),
                            Label(key="k1", value="v2", score=0.9),
                        ],
                    )
                ],
            ),
        ],
        preds_model_class=models.Prediction,
        preds_expected_number=5,
        expected_labels_tuples={
            ("k1", "v1"),
            ("k1", "v2"),
            ("k2", "v6"),
            ("k1", "v2"),
        },
        expected_scores={0.6, 0.4, 1.0, 0.1, 0.9},
        db=db,
    )


def test_evaluate_tabular_clf(
    client: Session,
    db: Session,
    y_true: list[int],
    tabular_preds: list[list[float]],
):
    assert len(y_true) == len(tabular_preds)

    dataset = Dataset.create(client, name=dset_name)
    gts = [
        GroundTruth(
            datum=Datum(dataset=dset_name, uid=f"uid{i}"),
            annotations=[
                Annotation(
                    task_type=TaskType.CLASSIFICATION,
                    labels=[Label(key="class", value=str(t))],
                )
            ],
        )
        for i, t in enumerate(y_true)
    ]
    for gt in gts:
        dataset.add_groundtruth(gt)

    # test
    model = Model.create(client, name=model_name)
    with pytest.raises(ClientException) as exc_info:
        model.evaluate_classification(dataset=dataset)
    assert "has not been finalized" in str(exc_info)

    dataset.finalize()

    pds = [
        Prediction(
            model=model_name,
            datum=Datum(dataset=dset_name, uid=f"uid{i}"),
            annotations=[
                Annotation(
                    task_type=TaskType.CLASSIFICATION,
                    labels=[
                        Label(key="class", value=str(i), score=pred[i])
                        for i in range(len(pred))
                    ],
                )
            ],
        )
        for i, pred in enumerate(tabular_preds)
    ]
    for pd in pds:
        model.add_prediction(pd)

    # test
    with pytest.raises(ClientException) as exc_info:
        model.evaluate_classification(dataset=dataset)
    assert "has not been finalized" in str(exc_info)

    model.finalize_inferences(dataset)

    # evaluate
    eval_job = model.evaluate_classification(dataset=dataset)
    assert eval_job.ignored_pred_keys == []
    assert eval_job.missing_pred_keys == []

    # sleep to give the backend time to compute
    time.sleep(1)
    assert eval_job.status == JobStatus.DONE

    metrics = eval_job.metrics

    expected_metrics = [
        {
            "type": "Accuracy",
            "parameters": {"label_key": "class"},
            "value": 0.5,
        },
        {
            "type": "ROCAUC",
            "parameters": {"label_key": "class"},
            "value": 0.7685185185185185,
        },
        {
            "type": "Precision",
            "value": 0.6666666666666666,
            "label": {"key": "class", "value": "1"},
        },
        {
            "type": "Recall",
            "value": 0.3333333333333333,
            "label": {"key": "class", "value": "1"},
        },
        {
            "type": "F1",
            "value": 0.4444444444444444,
            "label": {"key": "class", "value": "1"},
        },
        {
            "type": "Precision",
            "value": 0.0,
            "label": {"key": "class", "value": "2"},
        },
        {
            "type": "Recall",
            "value": 0.0,
            "label": {"key": "class", "value": "2"},
        },
        {"type": "F1", "value": 0.0, "label": {"key": "class", "value": "2"}},
        {
            "type": "Precision",
            "value": 0.5,
            "label": {"key": "class", "value": "0"},
        },
        {
            "type": "Recall",
            "value": 1.0,
            "label": {"key": "class", "value": "0"},
        },
        {
            "type": "F1",
            "value": 0.6666666666666666,
            "label": {"key": "class", "value": "0"},
        },
    ]
    for m in metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in metrics

    confusion_matrices = eval_job.confusion_matrices

    expected_confusion_matrices = [
        {
            "label_key": "class",
            "entries": [
                {"prediction": "0", "groundtruth": "0", "count": 3},
                {"prediction": "0", "groundtruth": "1", "count": 3},
                {"prediction": "1", "groundtruth": "1", "count": 2},
                {"prediction": "1", "groundtruth": "2", "count": 1},
                {"prediction": "2", "groundtruth": "1", "count": 1},
            ],
        }
    ]

    # check eval maps to expected
    for confusion_matrix in confusion_matrices:
        assert confusion_matrix in expected_confusion_matrices

    # check expected maps to eval
    for expected_matrix in expected_confusion_matrices:
        assert expected_matrix in confusion_matrices

    # check model methods
    labels = model.get_labels()
    df = model.get_metric_dataframes()

    assert model.id == 0
    assert model.name == 0
    assert model.metadata == 0
    assert len(labels) == 3
    assert df[0]["df"] is pd.DataFrame

    # check evaluation
    eval_jobs = model.get_evaluations()
    assert len(eval_jobs) == 1
    eval_settings = eval_jobs[0].settings
    eval_settings.pop("id")
    assert eval_settings == {
        "model": "test_model",
        "dataset": "test_dataset",
        "task_type": "classification",
        "target_type": "none",
    }

    metrics_from_eval_settings_id = eval_jobs[0].metrics
    assert len(metrics_from_eval_settings_id) == len(expected_metrics)
    for m in metrics_from_eval_settings_id:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in metrics_from_eval_settings_id

    assert eval_jobs[0].confusion_matrices == expected_confusion_matrices

    model.delete()

    assert len(client.get_models()) == 0


def test_add_groundtruth(
    client: Client,
    gt_semantic_segs_error: GroundTruth,
):
    dataset = Dataset.create(client, dset_name)

    with pytest.raises(ClientException) as exc_info:
        dataset.add_groundtruth(gt_semantic_segs_error)

    assert "raster and image to have" in str(exc_info)

    client.delete_dataset(dset_name, timeout=30)


def test_get_groundtruth(
    client: Client,
    gt_semantic_segs1_mask: GroundTruth,
    gt_semantic_segs2_mask: GroundTruth,
):
    dataset = Dataset.create(client, dset_name)
    dataset.add_groundtruth(gt_semantic_segs1_mask)
    dataset.add_groundtruth(gt_semantic_segs2_mask)

    try:
        dataset.get_groundtruth("uid1")
        dataset.get_groundtruth("uid2")
    except Exception as e:
        raise AssertionError(e)

    client.delete_dataset(dset_name, timeout=30)


def test_add_raster_and_boundary_box(client: Client, img1: ImageMetadata):
    img_size = [900, 300]
    mask = _generate_mask(height=img_size[0], width=img_size[1])
    raster = Raster.from_numpy(mask)

    gt = GroundTruth(
        datum=img1.to_datum(),
        annotations=[
            Annotation(
                task_type=TaskType.DETECTION,
                labels=[Label(key="k3", value="v3")],
                bounding_box=BoundingBox.from_extrema(
                    xmin=10, ymin=10, xmax=60, ymax=40
                ),
                raster=raster,
            )
        ],
    )

    dataset = Dataset.create(client, dset_name)

    dataset.add_groundtruth(gt)

    fetched_gt = dataset.get_groundtruth("uid1")

    assert (
        fetched_gt.annotations[0].raster is not None
    ), "Raster doesn't exist on fetched gt"
    assert (
        fetched_gt.annotations[0].bounding_box is not None
    ), "Bounding box doesn't exist on fetched gt"

    client.delete_dataset(dset_name, timeout=30)


def test_get_dataset(
    client: Client,
    gt_semantic_segs1_mask: GroundTruth,
):
    dataset = Dataset.create(client, dset_name)
    dataset.add_groundtruth(gt_semantic_segs1_mask)

    # check get
    fetched_dataset = Dataset.get(client, dset_name)
    assert fetched_dataset == dataset

    # check get_info
    info = dataset.get_info()
    assert info.annotation_type == TaskType.SEMANTIC_SEGMENTATION
    assert info.number_of_segmentations == 1
    assert info.number_of_bounding_boxes == 0

    client.delete_dataset(dset_name, timeout=30)


def test_get_dataset_status(client: Client, gt_dets1: list):
    status = client.get_dataset_status(dset_name)
    assert status == "none"

    dataset = Dataset.create(client, dset_name)

    assert client.get_dataset_status(dset_name) == "none"

    gt = gt_dets1[0]

    dataset.add_groundtruth(gt)
    dataset.finalize()
    status = client.get_dataset_status(dset_name)
    assert status == "ready"

    client.delete_dataset(dset_name, timeout=30)

    status = client.get_dataset_status(dset_name)
    assert status == "none"


# @TODO: Implement metadata querying + geojson
# def test_create_images_with_metadata(
#     client: Client, db: Session, metadata: list[MetaDatum], rect1: BoundingBox
# ):
#     dataset = Dataset.create(client, dset_name)

#     md1, md2, md3 = metadata
#     img1 = ImageMetadata(uid="uid1", metadata=[md1], height=100, width=200).to_datum()
#     img2 = ImageMetadata(uid="uid2", metadata=[md2, md3], height=100, width=200).to_datum()

#     print(GroundTruth(
#             dataset=dset_name,
#             datum=img1.to_datum(),
#             annotations=[
#                 Annotation(
#                     task_type=TaskType.DETECTION,
#                     labels=[Label(key="k", value="v")],
#                     bounding_box=rect1,
#                 ),
#             ]
#         ))

#     dataset.add_groundtruth(
#         GroundTruth(
#             dataset=dset_name,
#             datum=img1.to_datum(),
#             annotations=[
#                 Annotation(
#                     task_type=TaskType.DETECTION,
#                     labels=[Label(key="k", value="v")],
#                     bounding_box=rect1,
#                 ),
#             ]
#         )
#     )
#     dataset.add_groundtruth(
#         GroundTruth(
#             dataset=dset_name,
#             datum=img2.to_datum(),
#             annotations=[
#                 Annotation(
#                     task_type=TaskType.CLASSIFICATION,
#                     labels=[Label(key="k", value="v")],
#                 )
#             ]
#         )
#     )

#     data = db.scalars(select(models.Datum)).all()
#     assert len(data) == 2
#     assert set(d.uid for d in data) == {"uid1", "uid2"}

#     metadata_links = data[0].datum_metadatum_links
#     assert len(metadata_links) == 1
#     metadatum = metadata_links[0].metadatum
#     assert metadata_links[0].metadatum.name == "metadatum name1"
#     assert json.loads(db.scalar(ST_AsGeoJSON(metadatum.geo))) == {
#         "type": "Point",
#         "coordinates": [-48.23456, 20.12345],
#     }
#     metadata_links = data[1].datum_metadatum_links
#     assert len(metadata_links) == 2
#     metadatum1 = metadata_links[0].metadatum
#     metadatum2 = metadata_links[1].metadatum
#     assert metadatum1.name == "metadatum name2"
#     assert metadatum1.string_value == "a string"
#     assert metadatum2.name == "metadatum name3"
#     assert metadatum2.numeric_value == 0.45


# @TODO: Need to implement metadatum querying
# def test_stratify_clf_metrics(
#     client: Session,
#     db: Session,
#     y_true: list[int],
#     tabular_preds: list[list[float]],
# ):

#     tabular_datum = Datum(
#         uid="uid"
#     )

#     dataset = Dataset.create(client, name=dset_name)
#     # create data and two-different defining groups of cohorts
#     gt_with_metadata = GroundTruth(
#         dataset=dset_name,
#         datum=tabular_datum,
#         annotations=[
#             Annotation(
#                 task_type=TaskType.CLASSIFICATION,
#                 labels=[Label(key="class", value=str(t))],
#                 metadata=[
#                     MetaDatum(key="md1", value=f"md1-val{i % 3}"),
#                     MetaDatum(key="md2", value=f"md2-val{i % 4}"),
#                 ]
#             )
#             for i, t in enumerate(y_true)
#         ]
#     )
#     dataset.add_groundtruth(gt_with_metadata)
#     dataset.finalize()

#     model = Model.create(client, name=model_name)
#     pd = Prediction(
#         model=model_name,
#         datum=tabular_datum,
#         annotations=[
#             Annotation(
#                 task_type=TaskType.CLASSIFICATION,
#                 labels=[
#                     ScoredLabel(Label(key="class", value=str(i)), score=pred[i])
#                     for i in range(len(pred))
#                 ]
#             )
#             for pred in tabular_preds
#         ]
#     )
#     model.add_prediction(pd)
#     model.finalize_inferences(dataset)

#     eval_job = model.evaluate_classification(dataset=dataset, group_by="md1")
#     time.sleep(2)

#     metrics = eval_job.metrics

#     for m in metrics:
#         assert m["group"] in [
#             {"name": "md1", "value": "md1-val0"},
#             {"name": "md1", "value": "md1-val1"},
#             {"name": "md1", "value": "md1-val2"},
#         ]

#     val2_metrics = [
#         m
#         for m in metrics
#         if m["group"] == {"name": "md1", "value": "md1-val2"}
#     ]

#     # for value 2: the gts are [2, 0, 1] and preds are [[0.03, 0.88, 0.09], [1.0, 0.0, 0.0], [0.78, 0.21, 0.01]]
#     # (hard preds [1, 0, 0])
#     expected_metrics = [
#         {
#             "type": "Accuracy",
#             "parameters": {"label_key": "class"},
#             "value": 0.3333333333333333,
#             "group": {"name": "md1", "value": "md1-val2"},
#         },
#         {
#             "type": "ROCAUC",
#             "parameters": {"label_key": "class"},
#             "value": 0.8333333333333334,
#             "group": {"name": "md1", "value": "md1-val2"},
#         },
#         {
#             "type": "Precision",
#             "value": 0.0,
#             "label": {"key": "class", "value": "1"},
#             "group": {"name": "md1", "value": "md1-val2"},
#         },
#         {
#             "type": "Recall",
#             "value": 0.0,
#             "label": {"key": "class", "value": "1"},
#             "group": {"name": "md1", "value": "md1-val2"},
#         },
#         {
#             "type": "F1",
#             "value": 0.0,
#             "label": {"key": "class", "value": "1"},
#             "group": {"name": "md1", "value": "md1-val2"},
#         },
#         {
#             "type": "Precision",
#             "value": -1,
#             "label": {"key": "class", "value": "2"},
#             "group": {"name": "md1", "value": "md1-val2"},
#         },
#         {
#             "type": "Recall",
#             "value": 0.0,
#             "label": {"key": "class", "value": "2"},
#             "group": {"name": "md1", "value": "md1-val2"},
#         },
#         {
#             "type": "F1",
#             "value": -1,
#             "label": {"key": "class", "value": "2"},
#             "group": {"name": "md1", "value": "md1-val2"},
#         },
#         {
#             "type": "Precision",
#             "value": 0.5,
#             "label": {"key": "class", "value": "0"},
#             "group": {"name": "md1", "value": "md1-val2"},
#         },
#         {
#             "type": "Recall",
#             "value": 1.0,
#             "label": {"key": "class", "value": "0"},
#             "group": {"name": "md1", "value": "md1-val2"},
#         },
#         {
#             "type": "F1",
#             "value": 0.6666666666666666,
#             "label": {"key": "class", "value": "0"},
#             "group": {"name": "md1", "value": "md1-val2"},
#         },
#     ]

#     assert len(val2_metrics) == len(expected_metrics)
#     for m in val2_metrics:
#         assert m in expected_metrics
#     for m in expected_metrics:
#         assert m in val2_metrics


# @TODO: Future PR
# def test_get_info_and_label_distributions(
#     client: Client,
#     gt_clfs1: list[GroundTruth],
#     gt_dets1: list[GroundTruth],
#     gt_poly_dets1: list[GroundTruth],
#     gt_instance_segs: list[GroundTruth],
#     pred_clfs: list[Prediction],
#     pred_dets: list[Prediction],
#     pred_poly_dets: list[Prediction],
#     pred_instance_segs: list[Prediction],
#     db: Session,
# ):
#     """Tests that the client can retrieve info about datasets and models.

#     Parameters
#     ----------
#     client
#     gts
#         list of groundtruth objects (from `velour.data_types`) of each type
#     preds
#         list of prediction objects (from `velour.data_types`) of each type
#     """

#     ds = Dataset.create(client, "info_test_dataset")
#     ds.add_groundtruth(gt_clfs1)
#     ds.add_groundtruth(gt_dets1)
#     ds.add_groundtruth(gt_poly_dets1)
#     ds.add_groundtruth(gt_instance_segs)
#     ds.finalize()

#     md = Model.create(client, "info_test_model")
#     md.add_prediction(ds, pred_clfs)
#     md.add_prediction(ds, pred_dets)
#     md.add_prediction(ds, pred_poly_dets)
#     md.add_prediction(ds, pred_instance_segs)
#     md.finalize_inferences(ds)

#     ds_info = ds.get_info()
#     assert ds_info.annotation_type == [
#         "CLASSIFICATION",
#         "DETECTION",
#         "SEGMENTATION",
#     ]
#     assert ds_info.number_of_classifications == 2
#     assert ds_info.number_of_bounding_boxes == 2
#     assert ds_info.number_of_bounding_polygons == 2
#     assert ds_info.number_of_segmentations == 2
#     assert ds_info.associated_models == ["info_test_model"]

#     md_info = md.get_info()
#     assert md_info.annotation_type == [
#         "CLASSIFICATION",
#         "DETECTION",
#         "SEGMENTATION",
#     ]
#     assert md_info.number_of_classifications == 5
#     assert md_info.number_of_bounding_boxes == 2
#     assert md_info.number_of_bounding_polygons == 2
#     assert md_info.number_of_segmentations == 2
#     assert md_info.associated_datasets == ["info_test_dataset"]

#     ds_dist = ds.get_label_distribution()
#     assert len(ds_dist) == 3
#     assert ds_dist[Label(key="k1", value="v1")] == 6
#     assert ds_dist[Label(key="k4", value="v4")] == 1
#     assert ds_dist[Label(key="k5", value="v5")] == 1

#     md_dist = md.get_label_distribution()
#     assert len(md_dist) == 7
#     assert md_dist[Label(key="k1", value="v1")] == {
#         "count": 3,
#         "scores": [0.3, 0.3, 0.87],
#     }
#     assert md_dist[Label(key="k12", value="v12")] == {
#         "count": 1,
#         "scores": [0.47],
#     }
#     assert md_dist[Label(key="k12", value="v16")] == {
#         "count": 1,
#         "scores": [0.53],
#     }
#     assert md_dist[Label(key="k13", value="v13")] == {
#         "count": 1,
#         "scores": [1.0],
#     }
#     assert md_dist[Label(key="k2", value="v2")] == {
#         "count": 3,
#         "scores": [0.98, 0.98, 0.92],
#     }
#     assert md_dist[Label(key="k4", value="v5")] == {
#         "count": 1,
#         "scores": [0.29],
#     }
#     assert md_dist[Label(key="k4", value="v4")] == {
#         "count": 1,
#         "scores": [0.71],
#     }

#     # Check that info is consistent with distribution
#     N_ds_info = (
#         ds_info.number_of_classifications
#         + ds_info.number_of_bounding_boxes
#         + ds_info.number_of_bounding_polygons
#         + ds_info.number_of_segmentations
#     )
#     N_ds_dist = sum([ds_dist[label] for label in ds_dist])
#     assert N_ds_info == N_ds_dist

#     N_md_info = (
#         md_info.number_of_classifications
#         + md_info.number_of_bounding_boxes
#         + md_info.number_of_bounding_polygons
#         + md_info.number_of_segmentations
#     )
#     N_md_dist = sum([md_dist[label]["count"] for label in md_dist])
#     assert N_md_info == N_md_dist
