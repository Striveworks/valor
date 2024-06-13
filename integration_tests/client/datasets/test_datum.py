""" These integration tests should be run with a back end at http://localhost:8000
that is no auth
"""

from typing import List, Tuple

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from valor import (
    Annotation,
    Client,
    Dataset,
    Datum,
    Filter,
    GroundTruth,
    Label,
)
from valor.metatypes import ImageMetadata
from valor.schemas import Box
from valor_api.backend import models


@pytest.fixture
def dataset_with_metadata(
    client: Client,
    dataset_name: str,
    metadata: dict,
    rect1: List[Tuple[float, float]],
) -> Dataset:
    # split metadata
    md1 = {"metadatum1": metadata["metadatum1"]}
    md23 = {
        "metadatum2": metadata["metadatum2"],
        "metadatum3": metadata["metadatum3"],
    }

    # create via image metatypes
    img1 = ImageMetadata.create(
        uid="uid1", metadata=md1, height=100, width=200
    ).datum
    img2 = ImageMetadata.create(
        uid="uid2", metadata=md23, height=200, width=100
    ).datum

    # create dataset
    dataset = Dataset.create(dataset_name)
    dataset.add_groundtruth(
        GroundTruth(
            datum=img1,
            annotations=[
                Annotation(
                    labels=[Label(key="k", value="v")],
                    bounding_box=Box([rect1]),
                    is_instance=True,
                ),
            ],
        )
    )
    dataset.add_groundtruth(
        GroundTruth(
            datum=img2,
            annotations=[
                Annotation(
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
    assert data[0].meta["metadatum1"] == "temporary"
    assert data[0].meta["height"] == 100
    assert data[0].meta["width"] == 200

    assert len(data[1].meta) == 4
    assert data[1].meta["metadatum2"] == "a string"
    assert data[1].meta["metadatum3"] == 0.45
    assert data[1].meta["height"] == 200
    assert data[1].meta["width"] == 100


def test_get_datums(
    db: Session, dataset_with_metadata: Dataset, metadata: dict
):
    assert len(dataset_with_metadata.get_datums()) == 2

    assert (
        len(
            dataset_with_metadata.get_datums(
                filters=Filter(
                    datums=(
                        Datum.metadata["metadatum1"] == metadata["metadatum1"]
                    )
                )
            )
        )
        == 1
    )
    assert (
        len(
            dataset_with_metadata.get_datums(
                filters=Filter(
                    datums=(Datum.metadata["metadatum1"] == "nonexistent value")  # type: ignore - issue #605
                )
            )
        )
        == 0
    )
