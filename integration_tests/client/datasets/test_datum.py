""" These integration tests should be run with a backend at http://localhost:8000
that is no auth
"""
from sqlalchemy import select
from sqlalchemy.orm import Session

from velour import Annotation, Client, Dataset, GroundTruth, Label
from velour.enums import TaskType
from velour.metatypes import ImageMetadata
from velour.schemas import BoundingBox
from velour_api.backend import models


def test_create_images_with_metadata(
    client: Client,
    db: Session,
    dataset_name: str,
    metadata: dict,
    rect1: BoundingBox,
):
    # split metadata
    md1 = {
        "metadatum1": metadata["metadatum1"],
    }
    md23 = {
        "metadatum2": metadata["metadatum2"],
        "metadatum3": metadata["metadatum3"],
    }

    # create via image metatypes
    img1 = ImageMetadata(
        uid="uid1", metadata=md1, height=100, width=200
    ).to_datum()
    img2 = ImageMetadata(
        uid="uid2", metadata=md23, height=200, width=100
    ).to_datum()

    # create dataset
    dataset = Dataset.create(dataset_name)
    dataset.add_groundtruth(
        GroundTruth(
            datum=img1,
            annotations=[
                Annotation(
                    task_type=TaskType.OBJECT_DETECTION,
                    labels=[Label(key="k", value="v")],
                    bounding_box=rect1,
                ),
            ],
        )
    )
    dataset.add_groundtruth(
        GroundTruth(
            datum=img2,
            annotations=[
                Annotation(
                    task_type=TaskType.CLASSIFICATION,
                    labels=[Label(key="k", value="v")],
                )
            ],
        )
    )

    data = db.scalars(select(models.Datum)).all()
    assert len(data) == 2
    assert set(d.uid for d in data) == {"uid1", "uid2"}

    assert len(data[0].meta) == 3
    assert data[0].meta["metadatum1"] == metadata["metadatum1"]
    assert data[0].meta["height"] == 100
    assert data[0].meta["width"] == 200

    assert len(data[1].meta) == 4
    assert data[1].meta["metadatum2"] == metadata["metadatum2"]
    assert data[1].meta["metadatum3"] == metadata["metadatum3"]
    assert data[1].meta["height"] == 200
    assert data[1].meta["width"] == 100
