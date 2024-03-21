""" These integration tests should be run with a back end at http://localhost:8000
that is no auth
"""

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.valor_api.backend import models
from client.valor import Annotation, Client, Dataset, Datum, GroundTruth, Label
from client.valor.enums import TaskType
from client.valor.metatypes import ImageMetadata
from client.valor.schemas import BoundingBox


@pytest.fixture
def dataset_with_metadata(
    client: Client,
    dataset_name: str,
    metadata: dict,
    rect1: BoundingBox,
) -> Dataset:
    # split metadata
    md1 = {"metadatum1": metadata["metadatum1"]}
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

    return dataset


def test_create_images_with_metadata(
    db: Session, dataset_with_metadata: Dataset, metadata: dict
):
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


def test_get_datums(
    db: Session, dataset_with_metadata: Dataset, metadata: dict
):
    assert len(dataset_with_metadata.get_datums()) == 2
    assert (
        len(
            dataset_with_metadata.get_datums(
                filter_by=[
                    Datum.metadata["metadatum1"] == metadata["metadatum1"]
                ]
            )
        )
        == 1
    )
    assert (
        len(
            dataset_with_metadata.get_datums(
                filter_by=[Datum.metadata["metadatum1"] == "nonexistent value"]  # type: ignore - purposefully using bad filter criteria
            )
        )
        == 0
    )

    with pytest.raises(ValueError) as exc_info:
        dataset_with_metadata.get_datums(
            filter_by=[Dataset.name == "dataset name"]  # type: ignore - purposefully throwing error
        )
    assert (
        "Cannot filter by dataset_names when calling `Dataset.get_datums`"
        in str(exc_info)
    )
