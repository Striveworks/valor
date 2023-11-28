""" These integration tests should be run with a backend at http://localhost:8000
that is no auth
"""
import io
import json
from dataclasses import asdict
from typing import Any, Dict, Union

import numpy as np
import pandas
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

from velour import (
    Annotation,
    Dataset,
    Datum,
    GroundTruth,
    Label,
    Model,
    Prediction,
)
from velour.client import Client, ClientException
from velour.data_generation import _generate_mask
from velour.enums import AnnotationType, DataType, JobStatus, TaskType
from velour.metatypes import ImageMetadata
from velour.schemas import (
    BasicPolygon,
    BoundingBox,
    MultiPolygon,
    Point,
    Polygon,
    Raster,
)
from velour_api import exceptions
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


@pytest.fixture
def metadata():
    """Some sample metadata of different types"""
    return {
        "metadatum1": "temporary",
        "metadatum2": "a string",
        "metadatum3": 0.45,
    }


@pytest.fixture
def client():
    return Client(host="http://localhost:8000")


@pytest.fixture
def img1() -> ImageMetadata:
    coordinates = [
        [
            [125.2750725, 38.760525],
            [125.3902365, 38.775069],
            [125.5054005, 38.789613],
            [125.5051935, 38.71402425],
            [125.5049865, 38.6384355],
            [125.3902005, 38.6244225],
            [125.2754145, 38.6104095],
            [125.2752435, 38.68546725],
            [125.2750725, 38.760525],
        ]
    ]

    geo_dict = {"type": "Polygon", "coordinates": coordinates}

    return ImageMetadata(
        dataset=dset_name,
        uid="uid1",
        height=900,
        width=300,
        geospatial=geo_dict,
    )


@pytest.fixture
def img2() -> ImageMetadata:
    coordinates = [44.1, 22.4]

    geo_dict = {"type": "Point", "coordinates": coordinates}

    return ImageMetadata(
        dataset=dset_name,
        uid="uid2",
        height=40,
        width=30,
        geospatial=geo_dict,
    )


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
            client.delete_model(model["name"], timeout=5)
        except exceptions.ModelDoesNotExistError:
            continue

    for dataset in client.get_datasets():
        try:
            client.delete_dataset(dataset["name"], timeout=5)
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
def rect4():
    return BoundingBox.from_extrema(xmin=1, ymin=10, xmax=10, ymax=20)


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
                    task_type=TaskType.DETECTION,
                    labels=[Label(key="k1", value="v1")],
                    multipolygon=MultiPolygon(
                        polygons=[Polygon(boundary=rect1.polygon)]
                    ),
                ),
                Annotation(
                    task_type=TaskType.SEGMENTATION,
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
                    task_type=TaskType.DETECTION,
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
                    task_type=TaskType.SEGMENTATION,
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
                task_type=TaskType.SEGMENTATION,
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
                    task_type=TaskType.SEGMENTATION,
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
                task_type=TaskType.SEGMENTATION,
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
                task_type=TaskType.SEGMENTATION,
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
def pred_dets2(
    rect3: BoundingBox,
    rect4: BoundingBox,
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
                    labels=[Label(key="k1", value="v1", score=0.7)],
                    bounding_box=rect3,
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
                    bounding_box=rect4,
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
                    task_type=TaskType.DETECTION,
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
                    task_type=TaskType.DETECTION,
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
                    task_type=TaskType.SEGMENTATION,
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
                    task_type=TaskType.SEGMENTATION,
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


def test_client():
    bad_url = "localhost:8000"

    with pytest.raises(ValueError):
        Client(host=bad_url)

    bad_url2 = "http://localhost:8111"

    with pytest.raises(Exception):
        Client(host=bad_url2)

    good_url = "http://localhost:8000"

    assert Client(host=good_url)


def test__requests_wrapper(client: Client):
    datasets = client._requests_wrapper("get", "datasets")
    assert len(datasets.json()) == 0

    with pytest.raises(ValueError):
        client._requests_wrapper("get", "/datasets/fake_dataset/status")

    with pytest.raises(AssertionError):
        client._requests_wrapper("bad_method", "datasets/fake_dataset/status")

    with pytest.raises(ClientException):
        client._requests_wrapper("get", "not_an_endpoint")

    with pytest.raises(ClientException):
        client._requests_wrapper("get", "datasets/fake_dataset/status")


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
    Dataset.create(
        client,
        dset_name,
        metadata={
            "href": href,
            "description": description,
        },
    )

    dataset_id = db.scalar(
        select(models.Dataset.id).where(models.Dataset.name == dset_name)
    )
    assert isinstance(dataset_id, int)

    dataset_metadata = db.scalar(
        select(models.Dataset.meta).where(models.Dataset.name == dset_name)
    )
    assert dataset_metadata == {
        "href": "http://a.com/b",
        "description": "a description",
    }


def test_create_model_with_href_and_description(client: Client, db: Session):
    href = "http://a.com/b"
    description = "a description"
    Model.create(
        client,
        model_name,
        metadata={
            "href": href,
            "description": description,
        },
    )

    model_id = db.scalar(
        select(models.Model.id).where(models.Model.name == model_name)
    )
    assert isinstance(model_id, int)

    model_metadata = db.scalar(
        select(models.Model.meta).where(models.Model.name == model_name)
    )
    assert model_metadata == {
        "href": "http://a.com/b",
        "description": "a description",
    }


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
        if seg.task_type == TaskType.DETECTION:
            instance_segs.append(seg)
        elif seg.task_type == TaskType.SEGMENTATION:
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
                    task_type=TaskType.SEGMENTATION,
                    labels=[Label(key="k1", value="v1")],
                    raster=Raster.from_numpy(mask),
                ),
                Annotation(
                    task_type=TaskType.SEGMENTATION,
                    labels=[Label(key="k1", value="v1")],
                    multipolygon=MultiPolygon(polygons=[poly]),
                ),
            ],
        )

        dataset.add_groundtruth(gts)

    assert "one annotation per label" in str(exc_info.value)

    # fine with instance segmentation though
    gts = GroundTruth(
        datum=img1.to_datum(),
        annotations=[
            Annotation(
                task_type=TaskType.SEGMENTATION,
                labels=[Label(key="k1", value="v1")],
                raster=Raster.from_numpy(mask),
            ),
            Annotation(
                task_type=TaskType.DETECTION,
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
    assert db.scalar(select(func.count(models.Dataset.name))) == 0


def test_client_delete_model(client: Client, db: Session):
    """test that delete dataset returns a job whose status changes from "Processing" to "Done" """
    Model.create(client, model_name)
    assert db.scalar(select(func.count(models.Model.name))) == 1
    client.delete_model(model_name, timeout=30)
    assert db.scalar(select(func.count(models.Model.name))) == 0


def test_evaluate_detection(
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

    eval_job = model.evaluate_detection(
        dataset=dataset,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_keep=[0.1, 0.6],
        filters=[
            Label.key == "k1",
            Annotation.type == AnnotationType.BOX,
        ],
        timeout=30,
    )

    assert eval_job.id
    assert eval_job.task_type == "object-detection"
    assert eval_job.status.value == "done"
    assert all(
        [
            key in eval_job.metrics
            for key in [
                "dataset",
                "model",
                "settings",
                "job_id",
                "status",
                "metrics",
                "confusion_matrices",
            ]
        ]
    )
    assert eval_job.ignored_pred_labels == []
    assert eval_job.missing_pred_labels == []
    assert isinstance(eval_job._id, int)

    eval_job.wait_for_completion()
    assert eval_job.status == JobStatus.DONE

    settings = asdict(eval_job.settings)

    settings.pop("id")
    assert settings == {
        "model": model_name,
        "dataset": "test_dataset",
        "task_type": "object-detection",
        "settings": {
            "parameters": {
                "iou_thresholds_to_compute": [0.1, 0.6],
                "iou_thresholds_to_keep": [0.1, 0.6],
            },
            "filters": {
                "annotation_types": ["box"],
                "label_keys": ["k1"],
            },
        },
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

    assert eval_job.metrics["metrics"] == expected_metrics

    # now test if we set min_area and/or max_area
    areas = db.scalars(
        select(ST_Area(models.Annotation.box)).where(
            models.Annotation.model_id.isnot(None)
        )
    ).all()
    assert sorted(areas) == [1100.0, 1500.0]

    # sanity check this should give us the same thing except min_area and max_area are not none
    eval_job_bounded_area_10_2000 = model.evaluate_detection(
        dataset=dataset,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_keep=[0.1, 0.6],
        filters=[
            Label.key == "k1",
            Annotation.type == AnnotationType.BOX,
            Annotation.geometric_area >= 10,
            Annotation.geometric_area <= 2000,
        ],
        timeout=30,
    )
    settings = asdict(eval_job_bounded_area_10_2000.settings)
    settings.pop("id")
    assert settings == {
        "model": model_name,
        "dataset": "test_dataset",
        "task_type": "object-detection",
        "settings": {
            "filters": {
                "annotation_types": ["box"],
                "annotation_geometric_area": [
                    {
                        "operator": ">=",
                        "value": 10.0,
                    },
                    {
                        "operator": "<=",
                        "value": 2000.0,
                    },
                ],
                "label_keys": ["k1"],
            },
            "parameters": {
                "iou_thresholds_to_compute": [0.1, 0.6],
                "iou_thresholds_to_keep": [0.1, 0.6],
            },
        },
    }
    assert eval_job_bounded_area_10_2000.metrics["metrics"] == expected_metrics

    # now check we get different things by setting the thresholds accordingly
    # min area threshold should divide the set of annotations
    eval_job_min_area_1200 = model.evaluate_detection(
        dataset=dataset,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_keep=[0.1, 0.6],
        filters=[
            Label.key == "k1",
            Annotation.type == AnnotationType.BOX,
            Annotation.geometric_area >= 1200,
        ],
        timeout=30,
    )
    settings = asdict(eval_job_min_area_1200.settings)
    settings.pop("id")
    assert settings == {
        "model": model_name,
        "dataset": "test_dataset",
        "task_type": "object-detection",
        "settings": {
            "filters": {
                "annotation_types": ["box"],
                "annotation_geometric_area": [
                    {
                        "operator": ">=",
                        "value": 1200.0,
                    },
                ],
                "label_keys": ["k1"],
            },
            "parameters": {
                "iou_thresholds_to_compute": [0.1, 0.6],
                "iou_thresholds_to_keep": [0.1, 0.6],
            },
        },
    }
    assert eval_job_min_area_1200.metrics["metrics"] != expected_metrics

    # check for difference with max area now dividing the set of annotations
    eval_job_max_area_1200 = model.evaluate_detection(
        dataset=dataset,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_keep=[0.1, 0.6],
        filters=[
            Label.key == "k1",
            Annotation.type == AnnotationType.BOX,
            Annotation.geometric_area <= 1200,
        ],
        timeout=30,
    )
    settings = asdict(eval_job_max_area_1200.settings)
    settings.pop("id")
    assert settings == {
        "model": model_name,
        "dataset": "test_dataset",
        "task_type": "object-detection",
        "settings": {
            "filters": {
                "annotation_types": ["box"],
                "annotation_geometric_area": [
                    {
                        "operator": "<=",
                        "value": 1200.0,
                    },
                ],
                "label_keys": ["k1"],
            },
            "parameters": {
                "iou_thresholds_to_compute": [0.1, 0.6],
                "iou_thresholds_to_keep": [0.1, 0.6],
            },
        },
    }
    assert eval_job_max_area_1200.metrics["metrics"] != expected_metrics

    # should perform the same as the first min area evaluation
    # except now has an upper bound
    eval_job_bounded_area_1200_1800 = model.evaluate_detection(
        dataset=dataset,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_keep=[0.1, 0.6],
        filters=[
            Label.key == "k1",
            Annotation.type == AnnotationType.BOX,
            Annotation.geometric_area >= 1200,
            Annotation.geometric_area <= 1800,
        ],
        timeout=30,
    )
    settings = asdict(eval_job_bounded_area_1200_1800.settings)
    settings.pop("id")
    assert settings == {
        "model": model_name,
        "dataset": "test_dataset",
        "task_type": "object-detection",
        "settings": {
            "filters": {
                "annotation_types": ["box"],
                "annotation_geometric_area": [
                    {
                        "operator": ">=",
                        "value": 1200.0,
                    },
                    {
                        "operator": "<=",
                        "value": 1800.0,
                    },
                ],
                "label_keys": ["k1"],
            },
            "parameters": {
                "iou_thresholds_to_compute": [0.1, 0.6],
                "iou_thresholds_to_keep": [0.1, 0.6],
            },
        },
    }
    assert (
        eval_job_bounded_area_1200_1800.metrics["metrics"] != expected_metrics
    )
    assert (
        eval_job_bounded_area_1200_1800.metrics["metrics"]
        == eval_job_min_area_1200.metrics["metrics"]
    )


def test_evaluate_detection_with_json_filters(
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

    eval_job_min_area_1200 = model.evaluate_detection(
        dataset=dataset,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_keep=[0.1, 0.6],
        filters=[
            Label.key == "k1",
            Annotation.type == AnnotationType.BOX,
            Annotation.geometric_area >= 1200,
        ],
        timeout=30,
    )

    eval_job_bounded_area_1200_1800 = model.evaluate_detection(
        dataset=dataset,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_keep=[0.1, 0.6],
        filters={
            "annotation_types": ["box"],
            "annotation_geometric_area": [
                {
                    "operator": ">=",
                    "value": 1200.0,
                },
                {
                    "operator": "<=",
                    "value": 1800.0,
                },
            ],
            "label_keys": ["k1"],
        },
        timeout=30,
    )

    settings = asdict(eval_job_bounded_area_1200_1800.settings)
    settings.pop("id")
    assert settings == {
        "model": model_name,
        "dataset": "test_dataset",
        "task_type": "object-detection",
        "settings": {
            "filters": {
                "annotation_types": ["box"],
                "annotation_geometric_area": [
                    {
                        "operator": ">=",
                        "value": 1200.0,
                    },
                    {
                        "operator": "<=",
                        "value": 1800.0,
                    },
                ],
                "label_keys": ["k1"],
            },
            "parameters": {
                "iou_thresholds_to_compute": [0.1, 0.6],
                "iou_thresholds_to_keep": [0.1, 0.6],
            },
        },
    }
    assert (
        eval_job_bounded_area_1200_1800.metrics["metrics"] != expected_metrics
    )
    assert (
        eval_job_bounded_area_1200_1800.metrics["metrics"]
        == eval_job_min_area_1200.metrics["metrics"]
    )


def test_get_bulk_evaluations(
    client: Client,
    gt_dets1: list[GroundTruth],
    pred_dets: list[Prediction],
    pred_dets2: list[Prediction],
    db: Session,
):
    dataset_ = dset_name
    model_ = model_name

    dataset = Dataset.create(client, dataset_)
    for gt in gt_dets1:
        dataset.add_groundtruth(gt)
    dataset.finalize()

    model = Model.create(client, model_)
    for pd in pred_dets:
        model.add_prediction(pd)
    model.finalize_inferences(dataset)

    eval_job = model.evaluate_detection(
        dataset=dataset,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_keep=[0.1, 0.6],
        filters=[
            Label.key == "k1",
            Annotation.type == AnnotationType.BOX,
        ],
        timeout=30,
    )
    eval_job.wait_for_completion()

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

    second_model_expected_metrics = [
        {
            "type": "AP",
            "parameters": {"iou": 0.1},
            "value": 0.0,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.6},
            "value": 0.0,
            "label": {"key": "k1", "value": "v1"},
        },
        {"type": "mAP", "parameters": {"iou": 0.1}, "value": 0.0},
        {"type": "mAP", "parameters": {"iou": 0.6}, "value": 0.0},
        {
            "type": "APAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.0,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "mAPAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.0,
        },
    ]

    # test error when we don't pass either a model or dataset
    with pytest.raises(ValueError):
        client.get_bulk_evaluations()

    evaluations = client.get_bulk_evaluations(
        datasets=dset_name, models=model_name
    )

    assert len(evaluations) == 1
    assert all(
        [
            name
            in [
                "dataset",
                "model",
                "settings",
                "job_id",
                "status",
                "metrics",
                "confusion_matrices",
            ]
            for evaluation in evaluations
            for name in evaluation.keys()
        ]
    )

    assert len(evaluations[0]["metrics"])
    assert evaluations[0]["metrics"] == expected_metrics

    # test incorrect names
    assert len(client.get_bulk_evaluations(datasets="wrong_dataset_name")) == 0
    assert len(client.get_bulk_evaluations(models="wrong_model_name")) == 0

    # test with multiple models
    second_model = Model.create(client, "second_model")
    for pd in pred_dets2:
        second_model.add_prediction(pd)
    second_model.finalize_inferences(dataset)

    eval_job = second_model.evaluate_detection(
        dataset=dataset,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_keep=[0.1, 0.6],
        filters=[
            Label.key == "k1",
            Annotation.type == AnnotationType.BOX,
        ],
        timeout=30,
    )
    eval_job.wait_for_completion()

    second_model_evaluations = client.get_bulk_evaluations(
        models="second_model"
    )

    assert len(second_model_evaluations) == 1
    assert all(
        [
            name
            in [
                "dataset",
                "model",
                "settings",
                "job_id",
                "status",
                "metrics",
                "confusion_matrices",
            ]
            for evaluation in second_model_evaluations
            for name in evaluation.keys()
        ]
    )
    assert (
        second_model_evaluations[0]["metrics"] == second_model_expected_metrics
    )

    both_evaluations = client.get_bulk_evaluations(datasets=["test_dataset"])

    # should contain two different entries, one for each model
    assert len(both_evaluations) == 2
    assert all(
        [
            evaluation["model"] in ["second_model", model_name]
            for evaluation in both_evaluations
        ]
    )
    assert both_evaluations[0]["metrics"] == expected_metrics
    assert both_evaluations[1]["metrics"] == second_model_expected_metrics

    # should be equivalent since there are only two models attributed to this dataset
    both_evaluations_from_model_names = client.get_bulk_evaluations(
        models=["second_model", "test_model"]
    )
    assert both_evaluations == both_evaluations_from_model_names


def test_evaluate_image_clf(
    client: Client,
    db: Session,  # this is unused but putting it here since the teardown of the fixture does cleanup
    gt_clfs: list[GroundTruth],
    pred_clfs: list[Prediction],
):
    dataset = Dataset.create(client, dset_name)
    for gt in gt_clfs:
        dataset.add_groundtruth(gt)
    dataset.finalize()

    model = Model.create(client, model_name)
    for pd in pred_clfs:
        model.add_prediction(pd)
    model.finalize_inferences(dataset)

    eval_job = model.evaluate_classification(dataset=dataset, timeout=30)

    assert set(eval_job.ignored_pred_keys) == {"k12", "k13"}
    assert set(eval_job.missing_pred_keys) == {"k3", "k5"}

    assert eval_job.status == JobStatus.DONE

    metrics = eval_job.metrics["metrics"]

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

    confusion_matrices = eval_job.metrics["confusion_matrices"]
    assert confusion_matrices == [
        {
            "label_key": "k4",
            "entries": [{"prediction": "v4", "groundtruth": "v4", "count": 1}],
        }
    ]


def test_evaluate_segmentation(
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

    eval_job = model.evaluate_segmentation(dataset, timeout=30)
    assert eval_job.missing_pred_labels == [
        {"key": "k3", "value": "v3", "score": None}
    ]
    assert eval_job.ignored_pred_labels == [
        {"key": "k1", "value": "v1", "score": None}
    ]

    metrics = eval_job.metrics["metrics"]

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
    client: Client,
    db: Session,
    metadata: Dict[str, Union[float, int, str]],
):
    dataset = Dataset.create(client, name=dset_name)
    assert isinstance(dataset, Dataset)

    md1 = {"metadatum1": metadata["metadatum1"]}
    md23 = {
        "metadatum2": metadata["metadatum2"],
        "metadatum3": metadata["metadatum3"],
    }

    gts = [
        GroundTruth(
            datum=Datum(
                dataset=dset_name,
                uid="uid1",
                metadata=md1,
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
                metadata=md23,
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
    metadata_links = data[0].meta
    assert len(metadata_links) == 1
    assert "metadatum1" in metadata_links
    assert metadata_links["metadatum1"] == "temporary"

    metadata_links = data[1].meta
    assert len(metadata_links) == 2
    assert "metadatum2" in metadata_links
    assert metadata_links["metadatum2"] == "a string"
    assert "metadatum3" in metadata_links
    assert metadata_links["metadatum3"] == 0.45

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
        model.evaluate_classification(dataset=dataset, timeout=30)
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
        model.evaluate_classification(dataset=dataset, timeout=30)
    assert "has not been finalized" in str(exc_info)

    model.finalize_inferences(dataset)

    # evaluate
    eval_job = model.evaluate_classification(dataset=dataset, timeout=30)
    assert eval_job.ignored_pred_keys == []
    assert eval_job.missing_pred_keys == []

    assert eval_job.status == JobStatus.DONE

    metrics = eval_job.metrics["metrics"]

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

    confusion_matrices = eval_job.metrics["confusion_matrices"]

    expected_confusion_matrix = {
        "label_key": "class",
        "entries": [
            {"prediction": "0", "groundtruth": "0", "count": 3},
            {"prediction": "0", "groundtruth": "1", "count": 3},
            {"prediction": "1", "groundtruth": "1", "count": 2},
            {"prediction": "1", "groundtruth": "2", "count": 1},
            {"prediction": "2", "groundtruth": "1", "count": 1},
        ],
    }

    # validate that we can fetch the confusion matrices through get_bulk_evaluations()
    bulk_evals = client.get_bulk_evaluations(datasets=dset_name)

    assert len(bulk_evals) == 1
    assert all(
        [
            name
            in [
                "dataset",
                "model",
                "settings",
                "job_id",
                "status",
                "metrics",
                "confusion_matrices",
            ]
            for evaluation in bulk_evals
            for name in evaluation.keys()
        ]
    )

    for metric in bulk_evals[0]["metrics"]:
        assert metric in expected_metrics

    assert len(bulk_evals[0]["confusion_matrices"][0]) == len(
        expected_confusion_matrix
    )

    # validate return schema
    assert len(confusion_matrices) == 1
    confusion_matrix = confusion_matrices[0]
    assert "label_key" in confusion_matrix
    assert "entries" in confusion_matrix

    # validate values
    assert (
        confusion_matrix["label_key"] == expected_confusion_matrix["label_key"]
    )
    for entry in confusion_matrix["entries"]:
        assert entry in expected_confusion_matrix["entries"]
    for entry in expected_confusion_matrix["entries"]:
        assert entry in confusion_matrix["entries"]

    # check model methods
    labels = model.get_labels()
    df = model.get_metric_dataframes()

    assert isinstance(model.id, int)
    assert model.name == model_name
    assert len(model.metadata) == 0

    assert len(labels) == 3
    assert isinstance(df[0]["df"], pandas.DataFrame)

    # check evaluation
    eval_jobs = model.get_evaluations()
    assert len(eval_jobs) == 1
    eval_settings = asdict(eval_jobs[0].settings)
    eval_settings.pop("id")
    assert eval_settings == {
        "model": model_name,
        "dataset": "test_dataset",
        "task_type": "classification",
        "settings": {},
    }

    metrics_from_eval_settings_id = eval_jobs[0].metrics["metrics"]
    assert len(metrics_from_eval_settings_id) == len(expected_metrics)
    for m in metrics_from_eval_settings_id:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in metrics_from_eval_settings_id

    # check confusion matrix
    confusion_matrices = eval_jobs[0].metrics["confusion_matrices"]

    # validate return schema
    assert len(confusion_matrices) == 1
    confusion_matrix = confusion_matrices[0]
    assert "label_key" in confusion_matrix
    assert "entries" in confusion_matrix

    # validate values
    assert (
        confusion_matrix["label_key"] == expected_confusion_matrix["label_key"]
    )
    for entry in confusion_matrix["entries"]:
        assert entry in expected_confusion_matrix["entries"]
    for entry in expected_confusion_matrix["entries"]:
        assert entry in confusion_matrix["entries"]

    model.delete()

    assert len(client.get_models()) == 0


def test_add_groundtruth(
    client: Client,
    db: Session,
    gt_semantic_segs_error: GroundTruth,
):
    dataset = Dataset.create(client, dset_name)

    with pytest.raises(TypeError):
        dataset.add_groundtruth("not_a_gt")

    with pytest.raises(ClientException) as exc_info:
        dataset.add_groundtruth(gt_semantic_segs_error)

    assert "raster and image to have" in str(exc_info)

    client.delete_dataset(dset_name, timeout=30)


def test_get_groundtruth(
    client: Client,
    db: Session,
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


def test_add_raster_and_boundary_box(
    client: Client, db: Session, img1: ImageMetadata
):
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
    db: Session,
    gt_semantic_segs1_mask: GroundTruth,
):
    dataset = Dataset.create(client, dset_name)
    dataset.add_groundtruth(gt_semantic_segs1_mask)

    # check get
    fetched_dataset = Dataset.get(client, dset_name)
    assert fetched_dataset.id == dataset.id
    assert fetched_dataset.name == dataset.name
    assert fetched_dataset.metadata == dataset.metadata

    client.delete_dataset(dset_name, timeout=30)


def test_set_and_get_geospatial(
    client: Client,
    gt_dets1: list[GroundTruth],
    db: Session,
):
    coordinates = [
        [
            [125.2750725, 38.760525],
            [125.3902365, 38.775069],
            [125.5054005, 38.789613],
            [125.5051935, 38.71402425],
            [125.5049865, 38.6384355],
            [125.3902005, 38.6244225],
            [125.2754145, 38.6104095],
            [125.2752435, 38.68546725],
            [125.2750725, 38.760525],
        ]
    ]
    geo_dict = {"type": "Polygon", "coordinates": coordinates}

    dataset = Dataset.create(
        client=client, name=dset_name, geospatial=geo_dict
    )

    # check Dataset's geospatial coordinates
    fetched_datasets = client.get_datasets()
    assert fetched_datasets[0]["geospatial"] == geo_dict

    # check Model's geospatial coordinates
    Model.create(client=client, name=model_name, geospatial=geo_dict)
    fetched_models = client.get_models()
    assert fetched_models[0]["geospatial"] == geo_dict

    # check Datums's geospatial coordinates
    for gt in gt_dets1:
        dataset.add_groundtruth(gt)
    dataset.finalize()

    expected_coords = [gt.datum.geospatial for gt in gt_dets1]

    returned_datum1 = dataset.get_datums()[0].geospatial
    returned_datum2 = dataset.get_datums()[1].geospatial

    assert expected_coords[0] == returned_datum1
    assert expected_coords[1] == returned_datum2

    dets1 = dataset.get_groundtruth("uid1")

    assert dets1.datum.geospatial == expected_coords[0]


def test_geospatial_filter(
    client: Client,
    gt_dets1: list[GroundTruth],
    pred_dets: list[Prediction],
    db: Session,
):
    coordinates = [
        [
            [125.2750725, 38.760525],
            [125.3902365, 38.775069],
            [125.5054005, 38.789613],
            [125.5051935, 38.71402425],
            [125.5049865, 38.6384355],
            [125.3902005, 38.6244225],
            [125.2754145, 38.6104095],
            [125.2752435, 38.68546725],
            [125.2750725, 38.760525],
        ]
    ]
    geo_dict = {"type": "Polygon", "coordinates": coordinates}

    dataset = Dataset.create(
        client=client, name=dset_name, geospatial=geo_dict
    )
    for gt in gt_dets1:
        gt.datum.geospatial = geo_dict
        dataset.add_groundtruth(gt)
    dataset.finalize()

    model = Model.create(client=client, name=model_name, geospatial=geo_dict)
    for pd in pred_dets:
        pd.datum.geospatial = geo_dict
        model.add_prediction(pd)
    model.finalize_inferences(dataset)

    # filtering by dataset should be disabled as dataset is called explicitly
    with pytest.raises(ClientException) as e:
        model.evaluate_detection(
            dataset=dataset,
            iou_thresholds_to_compute=[0.1, 0.6],
            iou_thresholds_to_keep=[0.1, 0.6],
            filters={
                "dataset_geospatial": [
                    {
                        "operator": "outside",
                        "value": {
                            "geometry": {
                                "type": "Point",
                                "coordinates": [0, 0],
                            }
                        },
                    }
                ],
            },
            timeout=30,
        )
    assert (
        "should not include any dataset, model, prediction score or task type filters"
        in str(e)
    )

    # test datums
    eval_job = model.evaluate_detection(
        dataset=dataset,
        iou_thresholds_to_compute=[0.1, 0.6],
        iou_thresholds_to_keep=[0.1, 0.6],
        filters={
            "datum_geospatial": [
                {
                    "operator": "inside",
                    "value": {
                        "geometry": {"type": "Point", "coordinates": [0, 0]}
                    },
                }
            ],
        },
        timeout=30,
    )

    settings = asdict(eval_job.settings)
    assert settings["settings"]["filters"]["datum_geospatial"] == [
        {
            "value": {
                "geometry": {
                    "type": "Point",
                    "coordinates": [0.0, 0.0],
                }
            },
            "operator": "inside",
        }
    ]

    assert len(eval_job.metrics["metrics"]) == 0

    # filtering by model should be disabled as model is called explicitly
    with pytest.raises(ClientException) as e:
        model.evaluate_detection(
            dataset=dataset,
            iou_thresholds_to_compute=[0.1, 0.6],
            iou_thresholds_to_keep=[0.1, 0.6],
            filters={
                "models_geospatial": [
                    {
                        "operator": "inside",
                        "value": {
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [
                                    [
                                        [124.0, 37.0],
                                        [128.0, 37.0],
                                        [128.0, 40.0],
                                        [124.0, 40.0],
                                    ]
                                ],
                            }
                        },
                    }
                ],
            },
            timeout=30,
        )
    assert (
        "should not include any dataset, model, prediction score or task type filters"
        in str(e)
    )


def test_get_dataset_status(client: Client, db: Session, gt_dets1: list):
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
#     client: Client, db: Session, metadata: list[Metadatum], rect1: BoundingBox
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
#                     Metadatum(key="md1", value=f"md1-val{i % 3}"),
#                     Metadatum(key="md2", value=f"md2-val{i % 4}"),
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

#     metrics = eval_job['metrics']

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
