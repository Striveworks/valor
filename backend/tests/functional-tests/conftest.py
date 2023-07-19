import io

import numpy as np
import pytest
from PIL import Image
from sqlalchemy import inspect, text
from sqlalchemy.orm import Session

from velour_api import crud, models, schemas
from velour_api.database import Base, create_db, make_session

# get all velour table names
classes = [
    v
    for v in models.__dict__.values()
    if isinstance(v, type) and issubclass(v, Base)
]
tablenames = [v.__tablename__ for v in classes if hasattr(v, "__tablename__")]


np.random.seed(29)


def drop_all(db):
    db.execute(text(f"DROP TABLE {', '.join(tablenames)};"))
    db.commit()


def random_mask_bytes(size: tuple[int, int]) -> bytes:
    mask = np.random.randint(0, 2, size=size, dtype=bool)
    mask = Image.fromarray(mask)
    f = io.BytesIO()
    mask.save(f, format="PNG")
    f.seek(0)
    return f.read()


@pytest.fixture
def mask_bytes1():
    return random_mask_bytes(size=(32, 64))


@pytest.fixture
def mask_bytes2():
    return random_mask_bytes(size=(16, 12))


@pytest.fixture
def mask_bytes3():
    return random_mask_bytes(size=(20, 27))


@pytest.fixture
def img1() -> schemas.Datum:
    return schemas.Datum(
        uid="uid1",
        metadata=[
            schemas.MetaDatum(
                name="type",
                value="image",
            ),
            schemas.MetaDatum(
                name="height",
                value=1000,
            ),
            schemas.MetaDatum(
                name="width",
                value=2000,
            ),
        ],
    )


@pytest.fixture
def img2() -> schemas.Datum:
    return schemas.Datum(
        uid="uid2",
        metadata=[
            schemas.MetaDatum(
                name="type",
                value="image",
            ),
            schemas.MetaDatum(
                name="height",
                value=1600,
            ),
            schemas.MetaDatum(
                name="width",
                value=1200,
            ),
        ],
    )


@pytest.fixture
def db():
    """This fixture provides a db session. a `RuntimeError` is raised if
    a velour tablename already exists. At teardown, all velour tables are wiped.
    """
    db = make_session()
    inspector = inspect(db.connection())
    for tablename in tablenames:
        if inspector.has_table(tablename):
            raise RuntimeError(
                f"Table {tablename} already exists; "
                "functional tests should be run with an empty db."
            )

    create_db()
    yield db
    # teardown
    drop_all(db)


@pytest.fixture
def dset(db: Session) -> models.Dataset:
    dset = models.Dataset(name="dset")
    db.add(dset)
    db.commit()

    return dset


@pytest.fixture
def img(db: Session, dset: models.Dataset) -> models.Datum:
    img = models.Datum(uid="uid", dataset_id=dset.id, height=1000, width=2000)
    db.add(img)
    db.commit()

    return img


def bounding_box(xmin, ymin, xmax, ymax) -> list[tuple[int, int]]:
    return [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)]


@pytest.fixture
def images() -> list[schemas.Image]:
    return [
        schemas.Image(uid=f"{i}", height=1000, width=2000) for i in range(4)
    ]


# groundtruths to use for testing AP
@pytest.fixture
def groundtruths(
    db: Session, images: list[schemas.Image]
) -> list[list[models.LabeledGroundTruthDetection]]:
    """Creates a dataset called "test dataset" with some groundtruth
    detections. These detections are taken from a torchmetrics unit test (see test_metrics.py)
    """
    dataset_name = "test dataset"
    crud.create_dataset(
        db,
        dataset=schemas.DatasetCreate(
            name=dataset_name, type=schemas.DatumTypes.IMAGE
        ),
    )
    gts_per_img = [
        {"boxes": [[214.1500, 41.2900, 562.4100, 285.0700]], "labels": ["4"]},
        {
            "boxes": [
                [13.00, 22.75, 548.98, 632.42],
                [1.66, 3.32, 270.26, 275.23],
            ],
            "labels": ["2", "2"],
        },
        {
            "boxes": [
                [61.87, 276.25, 358.29, 379.43],
                [2.75, 3.66, 162.15, 316.06],
                [295.55, 93.96, 313.97, 152.79],
                [326.94, 97.05, 340.49, 122.98],
                [356.62, 95.47, 372.33, 147.55],
                [462.08, 105.09, 493.74, 146.99],
                [277.11, 103.84, 292.44, 150.72],
            ],
            "labels": ["4", "1", "0", "0", "0", "0", "0"],
        },
        {
            "boxes": [
                [72.92, 45.96, 91.23, 80.57],
                [50.17, 45.34, 71.28, 79.83],
                [81.28, 47.04, 98.66, 78.50],
                [63.96, 46.17, 84.35, 80.48],
                [75.29, 23.01, 91.85, 50.85],
                [56.39, 21.65, 75.66, 45.54],
                [73.14, 1.10, 98.96, 28.33],
                [62.34, 55.23, 78.14, 79.57],
                [44.17, 45.78, 63.99, 78.48],
                [58.18, 44.80, 66.42, 56.25],
            ],
            "labels": [
                "49",
                "49",
                "49",
                "49",
                "49",
                "49",
                "49",
                "49",
                "49",
                "49",
            ],
        },
    ]
    db_gts_per_img = [
        [
            schemas.GroundTruthDetection(
                bbox=box,
                labels=[schemas.Label(key="class", value=class_label)],
                image=image,
            )
            for box, class_label in zip(gts["boxes"], gts["labels"])
        ]
        for gts, image in zip(gts_per_img, images)
    ]

    created_ids = [
        crud.create_groundtruth_detections(
            db,
            schemas.GroundTruthDetectionsCreate(
                dataset_name=dataset_name, detections=gts
            ),
        )
        for gts in db_gts_per_img
    ]
    return [
        [db.get(models.LabeledGroundTruthDetection, det_id) for det_id in ids]
        for ids in created_ids
    ]


# predictions to use for testing AP
@pytest.fixture
def predictions(
    db: Session, images: list[schemas.Image]
) -> list[list[models.LabeledPredictedDetection]]:
    """Creates a model called "test model" with some predicted
    detections on the dataset "test dataset". These predictions are taken
    from a torchmetrics unit test (see test_metrics.py)
    """
    model_name = "test model"
    dset_name = "test dataset"
    crud.create_model(
        db, schemas.Model(name=model_name, type=schemas.DatumTypes.IMAGE)
    )

    # predictions for four images taken from
    # https://github.com/Lightning-AI/metrics/blob/107dbfd5fb158b7ae6d76281df44bd94c836bfce/tests/unittests/detection/test_map.py#L59
    preds_per_img = [
        {
            "boxes": [[258.15, 41.29, 606.41, 285.07]],
            "scores": [0.236],
            "labels": ["4"],
        },
        {
            "boxes": [
                [61.00, 22.75, 565.00, 632.42],
                [12.66, 3.32, 281.26, 275.23],
            ],
            "scores": [0.318, 0.726],
            "labels": ["3", "2"],
        },
        {
            "boxes": [
                [87.87, 276.25, 384.29, 379.43],
                [0.00, 3.66, 142.15, 316.06],
                [296.55, 93.96, 314.97, 152.79],
                [328.94, 97.05, 342.49, 122.98],
                [356.62, 95.47, 372.33, 147.55],
                [464.08, 105.09, 495.74, 146.99],
                [276.11, 103.84, 291.44, 150.72],
            ],
            "scores": [0.546, 0.3, 0.407, 0.611, 0.335, 0.805, 0.953],
            "labels": ["4", "1", "0", "0", "0", "0", "0"],
        },
        {
            "boxes": [
                [72.92, 45.96, 91.23, 80.57],
                [45.17, 45.34, 66.28, 79.83],
                [82.28, 47.04, 99.66, 78.50],
                [59.96, 46.17, 80.35, 80.48],
                [75.29, 23.01, 91.85, 50.85],
                [71.14, 1.10, 96.96, 28.33],
                [61.34, 55.23, 77.14, 79.57],
                [41.17, 45.78, 60.99, 78.48],
                [56.18, 44.80, 64.42, 56.25],
            ],
            "scores": [
                0.532,
                0.204,
                0.782,
                0.202,
                0.883,
                0.271,
                0.561,
                0.204,
                0.349,
            ],
            "labels": ["49", "49", "49", "49", "49", "49", "49", "49", "49"],
        },
    ]

    db_preds_per_img = [
        [
            schemas.PredictedDetection(
                bbox=box,
                scored_labels=[
                    schemas.ScoredLabel(
                        label=schemas.Label(key="class", value=class_label),
                        score=score,
                    )
                ],
                image=image,
            )
            for box, class_label, score in zip(
                preds["boxes"], preds["labels"], preds["scores"]
            )
        ]
        for preds, image in zip(preds_per_img, images)
    ]

    created_ids = [
        crud.create_predicted_detections(
            db,
            schemas.PredictedDetectionsCreate(
                model_name=model_name, dataset_name=dset_name, detections=preds
            ),
        )
        for preds in db_preds_per_img
    ]
    return [
        [db.get(models.LabeledPredictedDetection, det_id) for det_id in ids]
        for ids in created_ids
    ]
