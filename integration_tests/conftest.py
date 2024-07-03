""" These integration tests should be run with a back end at http://localhost:8000
that is no auth
"""

import os
from typing import Iterator

import numpy as np
import pytest
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session

from valor import (
    Annotation,
    Client,
    GroundTruth,
    Label,
    Prediction,
    exceptions,
)
from valor.client import ClientConnection, connect, reset_connection
from valor.metatypes import Datum
from valor.schemas import Box, MultiPolygon, Point, Polygon, Raster
from valor_api.backend import models


def _generate_mask(
    height: int,
    width: int,
    minimum_mask_percent: float = 0.05,
    maximum_mask_percent: float = 0.4,
) -> np.ndarray:
    """Generate a random mask for an image with a given height and width"""
    mask_cutoff = np.random.uniform(minimum_mask_percent, maximum_mask_percent)
    mask = (np.random.random((height, width))) < mask_cutoff

    return mask


@pytest.fixture
def connection() -> ClientConnection:  # type: ignore - this function technically doesn't return anything, but downstream tests will throw errors if we change the return type to None
    reset_connection()
    connect(host="http://localhost:8000")


@pytest.fixture
def db(connection: ClientConnection) -> Iterator[Session]:
    """This fixture makes sure there's not datasets, models, or labels in the back end
    (raising a RuntimeError if there are). It returns a db session and as cleanup
    clears out all datasets, models, and labels from the back end.
    """
    client = Client(connection)

    if len(client.get_datasets()) > 0:
        raise RuntimeError(
            "Tests should be run on an empty valor back end but found existing datasets.",
            [ds.name for ds in client.get_datasets()],
        )

    if len(client.get_models()) > 0:
        raise RuntimeError(
            "Tests should be run on an empty valor back end but found existing models."
        )

    if len(client.get_labels()) > 0:
        raise RuntimeError(
            "Tests should be run on an empty valor back end but found existing labels."
        )

    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_USERNAME = os.getenv("POSTGRES_USERNAME", "postgres")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")
    POSTGRES_DB = os.getenv("POSTGRES_DB", "valor")
    POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
    SQLALCHEMY_DATABASE_URL = f"postgresql://{POSTGRES_USERNAME}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

    engine = create_engine(SQLALCHEMY_DATABASE_URL)
    sess = Session(engine)
    sess.execute(text("SET postgis.gdal_enabled_drivers = 'ENABLE_ALL';"))
    sess.execute(text("SET postgis.enable_outdb_rasters = True;"))

    yield sess

    for model in client.get_models():
        try:
            client.delete_model(model.name, timeout=360)
        except exceptions.ModelDoesNotExistError:
            continue

    for dataset in client.get_datasets():
        try:
            client.delete_dataset(dataset.name, timeout=360)
        except exceptions.DatasetDoesNotExistError:
            continue

    labels = sess.scalars(select(models.Label))
    for label in labels:
        sess.delete(label)
    sess.commit()


@pytest.fixture
def client(db: Session, connection: ClientConnection) -> Client:
    return Client(connection)


@pytest.fixture
def dataset_name():
    return "test_dataset"


@pytest.fixture
def model_name():
    return "test_model"


"""Metadata"""


@pytest.fixture
def metadata():
    """Some sample metadata of different types"""
    return {
        "metadatum1": "temporary",
        "metadatum2": "a string",
        "metadatum3": 0.45,
    }


"""Images"""


@pytest.fixture
def image_height():
    return 900


@pytest.fixture
def image_width():
    return 300


@pytest.fixture
def img1(
    image_height: int,
    image_width: int,
) -> Datum:
    coordinates = [
        [
            (125.2750725, 38.760525),
            (125.3902365, 38.775069),
            (125.5054005, 38.789613),
            (125.5051935, 38.71402425),
            (125.5049865, 38.6384355),
            (125.3902005, 38.6244225),
            (125.2754145, 38.6104095),
            (125.2752435, 38.68546725),
            (125.2750725, 38.760525),
        ]
    ]
    return Datum(
        uid="uid1",
        metadata={
            "geospatial": Polygon(coordinates),
            "height": image_height,
            "width": image_width,
        },
    )


@pytest.fixture
def img2(
    image_height: int,
    image_width: int,
) -> Datum:
    coordinates = (44.1, 22.4)
    return Datum(
        uid="uid2",
        metadata={
            "geospatial": Point(coordinates),
            "height": image_height,
            "width": image_width,
        },
    )


@pytest.fixture
def img5(
    image_height: int,
    image_width: int,
) -> Datum:
    return Datum(
        uid="uid5",
        metadata={
            "height": image_height,
            "width": image_width,
        },
    )


@pytest.fixture
def img6(
    image_height: int,
    image_width: int,
) -> Datum:
    return Datum(
        uid="uid6",
        metadata={
            "height": image_height,
            "width": image_width,
        },
    )


@pytest.fixture
def img8(
    image_height: int,
    image_width: int,
) -> Datum:
    return Datum(
        uid="uid8",
        metadata={
            "height": image_height,
            "width": image_width,
        },
    )


@pytest.fixture
def img9(
    image_height: int,
    image_width: int,
) -> Datum:
    return Datum(
        uid="uid9",
        metadata={
            "height": image_height,
            "width": image_width,
        },
    )


"""Geometrys"""


@pytest.fixture
def rect1() -> list[tuple[float, float]]:
    """Box with area = 1500."""
    return [
        (10, 10),
        (60, 10),
        (60, 40),
        (10, 40),
        (10, 10),
    ]


@pytest.fixture
def rect2() -> list[tuple[float, float]]:
    """Box with area = 1100."""
    return [
        (15, 0),
        (70, 0),
        (70, 20),
        (15, 20),
        (15, 0),
    ]


@pytest.fixture
def rect3() -> list[tuple[float, float]]:
    """Box with area = 57,510."""
    return [
        (87, 10),
        (158, 10),
        (158, 820),
        (87, 820),
        (87, 10),
    ]


@pytest.fixture
def rect4() -> list[tuple[float, float]]:
    """Box with area = 90."""
    return [
        (1, 10),
        (10, 10),
        (10, 20),
        (1, 20),
        (1, 10),
    ]


@pytest.fixture
def rect5() -> list[tuple[float, float]]:
    """Box with partial overlap to rect3."""
    return [
        (87, 10),
        (158, 10),
        (158, 400),
        (87, 400),
        (87, 10),
    ]


"""GroundTruths"""


@pytest.fixture
def gt_dets1(
    rect1: list[tuple[float, float]],
    rect2: list[tuple[float, float]],
    rect3: list[tuple[float, float]],
    img1: Datum,
    img2: Datum,
) -> list[GroundTruth]:
    return [
        GroundTruth(
            datum=img1,
            annotations=[
                Annotation(
                    is_instance=True,
                    labels=[Label(key="k1", value="v1")],
                    bounding_box=Box([rect1]),
                ),
                Annotation(
                    is_instance=True,
                    labels=[Label(key="k2", value="v2")],
                    bounding_box=Box([rect3]),
                ),
            ],
        ),
        GroundTruth(
            datum=img2,
            annotations=[
                Annotation(
                    is_instance=True,
                    labels=[Label(key="k1", value="v1")],
                    bounding_box=Box([rect2]),
                )
            ],
        ),
    ]


@pytest.fixture
def gt_dets2(
    rect1: list[tuple[float, float]],
    rect2: list[tuple[float, float]],
    rect3: list[tuple[float, float]],
    img5: Datum,
    img6: Datum,
    img8: Datum,
) -> list[GroundTruth]:
    return [
        GroundTruth(
            datum=img5,
            annotations=[
                Annotation(
                    is_instance=True,
                    labels=[Label(key="k1", value="v1")],
                    polygon=Polygon([rect1]),
                ),
                Annotation(
                    is_instance=True,
                    labels=[Label(key="k2", value="v2")],
                    bounding_box=Box([rect3]),
                ),
            ],
        ),
        GroundTruth(
            datum=img6,
            annotations=[
                Annotation(
                    is_instance=True,
                    labels=[Label(key="k1", value="v1")],
                    polygon=Polygon([rect2]),
                )
            ],
        ),
        GroundTruth(
            datum=img8,
            annotations=[
                Annotation(
                    is_instance=True,
                    labels=[Label(key="k3", value="v3")],
                    bounding_box=Box([rect3]),
                )
            ],
        ),
    ]


@pytest.fixture
def gts_det_with_label_maps(
    rect1: list[tuple[float, float]],
    rect2: list[tuple[float, float]],
    rect3: list[tuple[float, float]],
    img1: Datum,
    img2: Datum,
) -> list[GroundTruth]:
    return [
        GroundTruth(
            datum=img1,
            annotations=[
                Annotation(
                    is_instance=True,
                    labels=[Label(key="class_name", value="maine coon cat")],
                    bounding_box=Box([rect1]),
                ),
                Annotation(
                    is_instance=True,
                    labels=[Label(key="class", value="british shorthair")],
                    bounding_box=Box([rect3]),
                ),
                Annotation(
                    is_instance=True,
                    labels=[Label(key="k1", value="v1")],
                    bounding_box=Box([rect1]),
                ),
                Annotation(
                    is_instance=True,
                    labels=[Label(key="k2", value="v2")],
                    bounding_box=Box([rect3]),
                ),
            ],
        ),
        GroundTruth(
            datum=img2,
            annotations=[
                Annotation(
                    is_instance=True,
                    labels=[Label(key="class", value="siamese cat")],
                    bounding_box=Box([rect2]),
                ),
                Annotation(
                    is_instance=True,
                    labels=[Label(key="k1", value="v1")],
                    bounding_box=Box([rect2]),
                ),
            ],
        ),
    ]


@pytest.fixture
def gt_poly_dets1(
    img1: Datum,
    img2: Datum,
    rect1: list[tuple[float, float]],
    rect2: list[tuple[float, float]],
):
    """Same thing as gt_dets1 but represented as a polygon instead of bounding box"""
    return [
        GroundTruth(
            datum=img1,
            annotations=[
                Annotation(
                    is_instance=True,
                    labels=[Label(key="k1", value="v1")],
                    polygon=Polygon([rect1]),
                ),
            ],
        ),
        GroundTruth(
            datum=img2,
            annotations=[
                Annotation(
                    is_instance=True,
                    labels=[Label(key="k1", value="v1")],
                    polygon=Polygon([rect2]),
                )
            ],
        ),
    ]


@pytest.fixture
def gt_segs(
    rect1: list[tuple[float, float]],
    rect2: list[tuple[float, float]],
    rect3: list[tuple[float, float]],
    img1: Datum,
    img2: Datum,
    image_height: int,
    image_width: int,
) -> list[GroundTruth]:

    multipolygon1 = MultiPolygon([[rect1]])
    multipolygon31 = MultiPolygon([[rect3], [rect1]])
    multipolygon2_1 = MultiPolygon([[rect2, rect1]])  # boundary  # hole

    return [
        GroundTruth(
            datum=img1,
            annotations=[
                Annotation(
                    is_instance=True,
                    labels=[Label(key="k1", value="v1")],
                    raster=Raster.from_geometry(
                        geometry=multipolygon1,
                        height=image_height,
                        width=image_width,
                    ),
                ),
                Annotation(
                    is_instance=False,
                    labels=[Label(key="k2", value="v2")],
                    raster=Raster.from_geometry(
                        geometry=multipolygon31,
                        height=image_height,
                        width=image_width,
                    ),
                ),
            ],
        ),
        GroundTruth(
            datum=img2,
            annotations=[
                Annotation(
                    is_instance=True,
                    labels=[Label(key="k1", value="v1")],
                    raster=Raster.from_geometry(
                        geometry=multipolygon2_1,
                        height=image_height,
                        width=image_width,
                    ),
                )
            ],
        ),
    ]


@pytest.fixture
def gt_semantic_segs1(
    rect1: list[tuple[float, float]],
    rect3: list[tuple[float, float]],
    img1: Datum,
    image_height: int,
    image_width: int,
) -> list[GroundTruth]:
    return [
        GroundTruth(
            datum=img1,
            annotations=[
                Annotation(
                    is_instance=False,
                    labels=[Label(key="k2", value="v2")],
                    raster=Raster.from_geometry(
                        MultiPolygon(
                            [
                                [rect3],
                                [rect1],
                            ]
                        ),
                        height=image_height,
                        width=image_width,
                    ),
                )
            ],
        ),
    ]


@pytest.fixture
def gt_semantic_segs1_mask(
    img1: Datum,
    image_height: int,
    image_width: int,
) -> GroundTruth:
    mask = _generate_mask(height=image_height, width=image_width)
    raster = Raster.from_numpy(mask)

    return GroundTruth(
        datum=img1,
        annotations=[
            Annotation(
                is_instance=False,
                labels=[Label(key="k2", value="v2")],
                raster=raster,
            )
        ],
    )


@pytest.fixture
def gt_semantic_segs2(
    rect3: list[tuple[float, float]],
    img2: Datum,
    image_height: int,
    image_width: int,
) -> list[GroundTruth]:
    return [
        GroundTruth(
            datum=img2,
            annotations=[
                Annotation(
                    is_instance=False,
                    labels=[Label(key="k3", value="v3")],
                    raster=Raster.from_geometry(
                        MultiPolygon(
                            [[rect3]],
                        ),
                        height=image_height,
                        width=image_width,
                    ),
                )
            ],
        ),
    ]


@pytest.fixture
def gt_semantic_segs2_mask(
    img2: Datum,
    image_height: int,
    image_width: int,
) -> GroundTruth:
    mask = _generate_mask(height=image_height, width=image_width)
    raster = Raster.from_numpy(mask)

    return GroundTruth(
        datum=img2,
        annotations=[
            Annotation(
                is_instance=False,
                labels=[Label(key="k2", value="v2")],
                raster=raster,
            )
        ],
    )


@pytest.fixture
def gt_semantic_segs_mismatch(img1: Datum) -> GroundTruth:
    mask = _generate_mask(height=100, width=100)
    raster = Raster.from_numpy(mask)
    return GroundTruth(
        datum=img1,
        annotations=[
            Annotation(
                is_instance=False,
                labels=[Label(key="k3", value="v3")],
                raster=raster,
            )
        ],
    )


@pytest.fixture
def gt_clfs(
    img5: Datum,
    img6: Datum,
    img8: Datum,
) -> list[GroundTruth]:
    return [
        GroundTruth(
            datum=img5,
            annotations=[
                Annotation(
                    labels=[
                        Label(key="k4", value="v4"),
                        Label(key="k5", value="v5"),
                    ],
                ),
            ],
        ),
        GroundTruth(
            datum=img6,
            annotations=[
                Annotation(
                    labels=[Label(key="k4", value="v4")],
                )
            ],
        ),
        GroundTruth(
            datum=img8,
            annotations=[
                Annotation(
                    labels=[Label(key="k3", value="v3")],
                )
            ],
        ),
    ]


@pytest.fixture
def gt_clfs_tabular() -> list[int]:
    """ground truth for a tabular classification task"""
    return [1, 1, 2, 0, 0, 0, 1, 1, 1, 1]


"""Predictions"""


@pytest.fixture
def pred_dets(
    model_name: str,
    rect1: list[tuple[float, float]],
    rect2: list[tuple[float, float]],
    img1: Datum,
    img2: Datum,
) -> list[Prediction]:
    return [
        Prediction(
            datum=img1,
            annotations=[
                Annotation(
                    is_instance=True,
                    labels=[Label(key="k1", value="v1", score=0.3)],
                    bounding_box=Box([rect1]),
                )
            ],
        ),
        Prediction(
            datum=img2,
            annotations=[
                Annotation(
                    is_instance=True,
                    labels=[Label(key="k2", value="v2", score=0.98)],
                    bounding_box=Box([rect2]),
                )
            ],
        ),
    ]


@pytest.fixture
def pred_dets2(
    rect3: list[tuple[float, float]],
    rect4: list[tuple[float, float]],
    img1: Datum,
    img2: Datum,
) -> list[Prediction]:
    return [
        Prediction(
            datum=img1,
            annotations=[
                Annotation(
                    is_instance=True,
                    labels=[Label(key="k1", value="v1", score=0.7)],
                    bounding_box=Box([rect3]),
                )
            ],
        ),
        Prediction(
            datum=img2,
            annotations=[
                Annotation(
                    is_instance=True,
                    labels=[Label(key="k2", value="v2", score=0.98)],
                    bounding_box=Box([rect4]),
                )
            ],
        ),
    ]


@pytest.fixture
def preds_det_with_label_maps(
    rect1: list[tuple[float, float]],
    rect2: list[tuple[float, float]],
    img1: Datum,
    img2: Datum,
) -> list[Prediction]:
    return [
        Prediction(
            datum=img1,
            annotations=[
                Annotation(
                    is_instance=True,
                    labels=[Label(key="class", value="cat", score=0.3)],
                    bounding_box=Box([rect1]),
                ),
                Annotation(
                    is_instance=True,
                    labels=[Label(key="k1", value="v1", score=0.3)],
                    bounding_box=Box([rect1]),
                ),
            ],
        ),
        Prediction(
            datum=img2,
            annotations=[
                Annotation(
                    is_instance=True,
                    labels=[Label(key="class_name", value="cat", score=0.98)],
                    bounding_box=Box([rect2]),
                ),
                Annotation(
                    is_instance=True,
                    labels=[Label(key="k2", value="v2", score=0.98)],
                    bounding_box=Box([rect2]),
                ),
            ],
        ),
    ]


@pytest.fixture
def pred_poly_dets(pred_dets: list[Prediction]) -> list[Prediction]:
    return [
        Prediction(
            datum=det.datum,
            annotations=[
                Annotation(
                    is_instance=True,
                    labels=annotation.labels.get_value(),
                    polygon=(
                        Polygon([annotation.bounding_box.boundary])
                        if annotation.bounding_box.get_value()
                        else None
                    ),
                )
                for annotation in det.annotations
                if annotation.bounding_box is not None
            ],
        )
        for det in pred_dets
    ]


def _random_mask(
    img: Datum, image_height: int, image_width: int
) -> np.ndarray:
    return np.random.randint(
        0, 2, size=(image_height, image_width), dtype=bool
    )


@pytest.fixture
def pred_instance_segs(
    model_name: str,
    img1: Datum,
    img2: Datum,
    image_height: int,
    image_width: int,
) -> list[Prediction]:
    mask_1 = _random_mask(img1, image_height, image_width)
    mask_2 = _random_mask(img2, image_height, image_width)
    return [
        Prediction(
            datum=img1,
            annotations=[
                Annotation(
                    is_instance=True,
                    labels=[Label(key="k1", value="v1", score=0.87)],
                    raster=Raster.from_numpy(mask_1),
                )
            ],
        ),
        Prediction(
            datum=img2,
            annotations=[
                Annotation(
                    is_instance=True,
                    labels=[Label(key="k2", value="v2", score=0.92)],
                    raster=Raster.from_numpy(mask_2),
                )
            ],
        ),
    ]


@pytest.fixture
def pred_semantic_segs(
    model_name: str,
    img1: Datum,
    img2: Datum,
    image_height: int,
    image_width: int,
) -> list[Prediction]:
    mask_1 = _random_mask(img1, image_height, image_width)
    mask_2 = _random_mask(img2, image_height, image_width)
    return [
        Prediction(
            datum=img1,
            annotations=[
                Annotation(
                    is_instance=False,
                    labels=[Label(key="k2", value="v2")],
                    raster=Raster.from_numpy(mask_1),
                )
            ],
        ),
        Prediction(
            datum=img2,
            annotations=[
                Annotation(
                    is_instance=False,
                    labels=[Label(key="k1", value="v1")],
                    raster=Raster.from_numpy(mask_2),
                )
            ],
        ),
    ]


@pytest.fixture
def pred_clfs(
    model_name: str, img5: Datum, img6: Datum, img8: Datum
) -> list[Prediction]:
    return [
        Prediction(
            datum=img5,
            annotations=[
                Annotation(
                    labels=[
                        Label(key="k4", value="v1", score=0.47),
                        Label(key="k4", value="v8", score=0.53),
                        Label(key="k5", value="v1", score=1.0),
                    ],
                )
            ],
        ),
        Prediction(
            datum=img6,
            annotations=[
                Annotation(
                    labels=[
                        Label(key="k4", value="v4", score=0.71),
                        Label(key="k4", value="v5", score=0.29),
                    ],
                )
            ],
        ),
        Prediction(
            datum=img8,
            annotations=[
                Annotation(
                    labels=[
                        Label(key="k3", value="v1", score=1.0),
                    ],
                )
            ],
        ),
    ]


@pytest.fixture
def pred_clfs_tabular() -> list[list[float]]:
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
