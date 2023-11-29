""" These integration tests should be run with a backend at http://localhost:8000
that is no auth
"""
from typing import Any, Dict, Union

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from velour import Annotation, Dataset, Datum, GroundTruth, Label
from velour.client import Client, ClientException
from velour.enums import TaskType
from velour.metatypes import ImageMetadata
from velour_api.backend import models


def _test_create_image_dataset_with_gts(
    client: Client,
    dataset_name: str,
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

    dataset = Dataset.create(client, dataset_name)

    with pytest.raises(ClientException) as exc_info:
        Dataset.create(client, dataset_name)
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


def test_create_image_dataset_with_href_and_description(
    db: Session,
    client: Client,
    dataset_name: str,
):
    href = "http://a.com/b"
    description = "a description"
    Dataset.create(
        client,
        dataset_name,
        metadata={
            "href": href,
            "description": description,
        },
    )

    dataset_id = db.scalar(
        select(models.Dataset.id).where(models.Dataset.name == dataset_name)
    )
    assert isinstance(dataset_id, int)

    dataset_metadata = db.scalar(
        select(models.Dataset.meta).where(models.Dataset.name == dataset_name)
    )
    assert dataset_metadata == {
        "href": "http://a.com/b",
        "description": "a description",
    }


def test_create_image_dataset_with_detections(
    client: Client,
    dataset_name: str,
    gt_dets1: list[GroundTruth],
    gt_dets2: list[GroundTruth],
):
    dataset = _test_create_image_dataset_with_gts(
        client=client,
        dataset_name=dataset_name,
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


def test_create_image_dataset_with_segmentations(
    client: Client,
    dataset_name: str,
    gt_segs: list[GroundTruth],
    db: Session,  # this is unused but putting it here since the teardown of the fixture does cleanup
):
    dataset = _test_create_image_dataset_with_gts(
        client=client,
        dataset_name=dataset_name,
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


def test_create_image_dataset_with_classifications(
    client: Client,
    dataset_name: str,
    gt_clfs: list[GroundTruth],
):
    _test_create_image_dataset_with_gts(
        client=client,
        dataset_name=dataset_name,
        gts=gt_clfs,
        expected_image_uids={"uid5", "uid6", "uid8"},
        expected_labels_tuples={
            ("k5", "v5"),
            ("k4", "v4"),
            ("k3", "v3"),
        },
    )


def test_client_delete_dataset(
    db: Session,
    client: Client,
    dataset_name: str,
):
    """test that delete dataset returns a job whose status changes from "Processing" to "Done" """
    Dataset.create(client, dataset_name)
    assert db.scalar(select(func.count(models.Dataset.name))) == 1
    client.delete_dataset(dataset_name, timeout=30)
    assert db.scalar(select(func.count(models.Dataset.name))) == 0


def test_create_tabular_dataset_and_add_groundtruth(
    client: Client,
    db: Session,
    metadata: Dict[str, Union[float, int, str]],
    dataset_name: str,
):
    dataset = Dataset.create(client, name=dataset_name)
    assert isinstance(dataset, Dataset)

    md1 = {"metadatum1": metadata["metadatum1"]}
    md23 = {
        "metadatum2": metadata["metadatum2"],
        "metadatum3": metadata["metadatum3"],
    }

    gts = [
        GroundTruth(
            datum=Datum(
                dataset=dataset_name,
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
                dataset=dataset_name,
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
            datum=Datum(dataset=dataset_name, uid="uid3"),
            annotations=[
                Annotation(
                    task_type=TaskType.CLASSIFICATION,
                    labels=[Label(key="k1", value="v1")],
                )
            ],
        ),
        GroundTruth(
            datum=Datum(dataset=dataset_name, uid="uid4"),
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


def test_get_dataset(
    client: Client,
    dataset_name: str,
    gt_semantic_segs1_mask: GroundTruth,
):
    dataset = Dataset.create(client, dataset_name)
    dataset.add_groundtruth(gt_semantic_segs1_mask)

    # check get
    fetched_dataset = Dataset.get(client, dataset_name)
    assert fetched_dataset.id == dataset.id
    assert fetched_dataset.name == dataset.name
    assert fetched_dataset.metadata == dataset.metadata


def test_get_dataset_status(
    client: Client,
    dataset_name: str,
    gt_dets1: list,
):
    status = client.get_dataset_status(dataset_name)
    assert status == "none"

    dataset = Dataset.create(client, dataset_name)

    assert client.get_dataset_status(dataset_name) == "none"

    gt = gt_dets1[0]

    dataset.add_groundtruth(gt)
    dataset.finalize()
    status = client.get_dataset_status(dataset_name)
    assert status == "ready"

    dataset.delete()

    status = client.get_dataset_status(dataset_name)
    assert status == "none"


def test_validate_dataset(client: Client, dataset_name: str):
    with pytest.raises(TypeError):
        Dataset.create(client, name=123)

    with pytest.raises(TypeError):
        Dataset.create(client, name=dataset_name, id="not an int")
