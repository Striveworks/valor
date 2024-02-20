""" These integration tests should be run with a backend at http://localhost:8000
that is no auth
"""

import os
from typing import Iterator

import numpy as np
import pytest
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session

from valor import Annotation, Client, GroundTruth, Label, Prediction
from valor.client import ClientConnection, connect, reset_connection
from valor.data_generation import _generate_mask
from valor.enums import TaskType
from valor.metatypes import ImageMetadata
from valor.schemas import BoundingBox, MultiPolygon, Polygon, Raster
from valor_api import exceptions
from valor_api.backend import models


@pytest.fixture
def connection() -> ClientConnection:
    reset_connection()
    connect(host="http://localhost:8000")


@pytest.fixture
def db(connection: ClientConnection) -> Iterator[Session]:
    """This fixture makes sure there's not datasets, models, or labels in the backend
    (raising a RuntimeError if there are). It returns a db session and as cleanup
    clears out all datasets, models, and labels from the backend.
    """
    client = Client(connection)

    if len(client.get_datasets()) > 0:
        raise RuntimeError(
            "Tests should be run on an empty valor backend but found existing datasets.",
            [ds.name for ds in client.get_datasets()],
        )

    if len(client.get_models()) > 0:
        raise RuntimeError(
            "Tests should be run on an empty valor backend but found existing models."
        )

    if len(client.get_labels()) > 0:
        raise RuntimeError(
            "Tests should be run on an empty valor backend but found existing labels."
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
        uid="uid1", height=900, width=300, metadata={"geospatial": geo_dict}
    )


@pytest.fixture
def img2() -> ImageMetadata:
    coordinates = [44.1, 22.4]

    geo_dict = {"type": "Point", "coordinates": coordinates}

    return ImageMetadata(
        uid="uid2", height=40, width=30, metadata={"geospatial": geo_dict}
    )


@pytest.fixture
def img5(dataset_name) -> ImageMetadata:
    return ImageMetadata(uid="uid5", height=40, width=30)


@pytest.fixture
def img6() -> ImageMetadata:
    return ImageMetadata(uid="uid6", height=40, width=30)


@pytest.fixture
def img8() -> ImageMetadata:
    return ImageMetadata(uid="uid8", height=40, width=30)


@pytest.fixture
def img9() -> ImageMetadata:
    return ImageMetadata(uid="uid9", height=40, width=30)


"""Geometrys"""


@pytest.fixture
def rect1():
    """Box with area = 1500."""
    return BoundingBox.from_extrema(xmin=10, ymin=10, xmax=60, ymax=40)


@pytest.fixture
def rect2():
    """Box with area = 1100."""
    return BoundingBox.from_extrema(xmin=15, ymin=0, xmax=70, ymax=20)


@pytest.fixture
def rect3():
    """Box with area = 57,510."""
    return BoundingBox.from_extrema(xmin=87, ymin=10, xmax=158, ymax=820)


@pytest.fixture
def rect4():
    """Box with area = 90."""
    return BoundingBox.from_extrema(xmin=1, ymin=10, xmax=10, ymax=20)


"""GroundTruths"""


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
                    task_type=TaskType.OBJECT_DETECTION,
                    labels=[Label(key="k1", value="v1")],
                    bounding_box=rect1,
                ),
                Annotation(
                    task_type=TaskType.OBJECT_DETECTION,
                    labels=[Label(key="k2", value="v2")],
                    bounding_box=rect3,
                ),
            ],
        ),
        GroundTruth(
            datum=img2.to_datum(),
            annotations=[
                Annotation(
                    task_type=TaskType.OBJECT_DETECTION,
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
                    task_type=TaskType.OBJECT_DETECTION,
                    labels=[Label(key="k1", value="v1")],
                    polygon=Polygon(boundary=rect1.polygon, holes=[]),
                ),
                Annotation(
                    task_type=TaskType.OBJECT_DETECTION,
                    labels=[Label(key="k2", value="v2")],
                    bounding_box=rect3,
                ),
            ],
        ),
        GroundTruth(
            datum=img6.to_datum(),
            annotations=[
                Annotation(
                    task_type=TaskType.OBJECT_DETECTION,
                    labels=[Label(key="k1", value="v1")],
                    polygon=Polygon(boundary=rect2.polygon, holes=[]),
                )
            ],
        ),
        GroundTruth(
            datum=img8.to_datum(),
            annotations=[
                Annotation(
                    task_type=TaskType.OBJECT_DETECTION,
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
                    task_type=TaskType.OBJECT_DETECTION,
                    labels=[Label(key="k1", value="v1")],
                    polygon=Polygon(boundary=rect1.polygon, holes=[]),
                ),
            ],
        ),
        GroundTruth(
            datum=img2.to_datum(),
            annotations=[
                Annotation(
                    task_type=TaskType.OBJECT_DETECTION,
                    labels=[Label(key="k1", value="v1")],
                    polygon=Polygon(boundary=rect2.polygon, holes=[]),
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
) -> list[GroundTruth]:
    return [
        GroundTruth(
            datum=img1.to_datum(),
            annotations=[
                Annotation(
                    task_type=TaskType.OBJECT_DETECTION,
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
                    task_type=TaskType.OBJECT_DETECTION,
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
def gt_semantic_segs2(
    rect3: BoundingBox, img2: ImageMetadata
) -> list[GroundTruth]:
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
def gt_clfs_tabular() -> list[int]:
    """ground truth for a tabular classification task"""
    return [1, 1, 2, 0, 0, 0, 1, 1, 1, 1]


"""Predictions"""


@pytest.fixture
def pred_dets(
    model_name: str,
    rect1: BoundingBox,
    rect2: BoundingBox,
    img1: ImageMetadata,
    img2: ImageMetadata,
) -> list[Prediction]:
    return [
        Prediction(
            datum=img1.to_datum(),
            annotations=[
                Annotation(
                    task_type=TaskType.OBJECT_DETECTION,
                    labels=[Label(key="k1", value="v1", score=0.3)],
                    bounding_box=rect1,
                )
            ],
        ),
        Prediction(
            datum=img2.to_datum(),
            annotations=[
                Annotation(
                    task_type=TaskType.OBJECT_DETECTION,
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
            datum=img1.to_datum(),
            annotations=[
                Annotation(
                    task_type=TaskType.OBJECT_DETECTION,
                    labels=[Label(key="k1", value="v1", score=0.7)],
                    bounding_box=rect3,
                )
            ],
        ),
        Prediction(
            datum=img2.to_datum(),
            annotations=[
                Annotation(
                    task_type=TaskType.OBJECT_DETECTION,
                    labels=[Label(key="k2", value="v2", score=0.98)],
                    bounding_box=rect4,
                )
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
                    task_type=TaskType.OBJECT_DETECTION,
                    labels=annotation.labels,
                    polygon=Polygon(
                        boundary=annotation.bounding_box.polygon,
                        holes=[],
                    ),
                )
                for annotation in det.annotations
                if annotation.bounding_box is not None
            ],
        )
        for det in pred_dets
    ]


def _random_mask(img: ImageMetadata) -> np.ndarray:
    return np.random.randint(0, 2, size=(img.height, img.width), dtype=bool)


@pytest.fixture
def pred_instance_segs(
    model_name: str,
    img1: ImageMetadata,
    img2: ImageMetadata,
) -> list[Prediction]:
    mask_1 = _random_mask(img1)
    mask_2 = _random_mask(img2)
    return [
        Prediction(
            datum=img1.to_datum(),
            annotations=[
                Annotation(
                    task_type=TaskType.OBJECT_DETECTION,
                    labels=[Label(key="k1", value="v1", score=0.87)],
                    raster=Raster.from_numpy(mask_1),
                )
            ],
        ),
        Prediction(
            datum=img2.to_datum(),
            annotations=[
                Annotation(
                    task_type=TaskType.OBJECT_DETECTION,
                    labels=[Label(key="k2", value="v2", score=0.92)],
                    raster=Raster.from_numpy(mask_2),
                )
            ],
        ),
    ]


@pytest.fixture
def pred_semantic_segs(
    model_name: str,
    img1: ImageMetadata,
    img2: ImageMetadata,
) -> list[Prediction]:
    mask_1 = _random_mask(img1)
    mask_2 = _random_mask(img2)
    return [
        Prediction(
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
def pred_clfs(
    model_name: str, img5: ImageMetadata, img6: ImageMetadata
) -> list[Prediction]:
    return [
        Prediction(
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
