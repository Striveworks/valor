import io
from base64 import b64decode
from dataclasses import asdict

import PIL
import pytest
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session

from velour import ImageMetadata
from velour.client import Client
from velour.data_generation import (
    generate_prediction_data,
    generate_segmentation_data,
)
from velour.enums import AnnotationType, JobStatus, TaskType
from velour_api.backend import jobs, models

dset_name = "test_dataset"


def _mask_bytes_to_pil(mask_bytes):
    with io.BytesIO(mask_bytes) as f:
        return PIL.Image.open(f)


@pytest.fixture
def client():
    """This fixture makes sure there's not datasets, models, or labels in the backend
    (raising a RuntimeError if there are). It returns a db session and as cleanup
    clears out all datasets, models, and labels from the backend.
    """

    client = Client(host="http://localhost:8000")

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

    yield client

    for model in client.get_models():
        client.delete_model(model["name"], timeout=30)

    for dataset in client.get_datasets():
        client.delete_dataset(dataset["name"], timeout=5)

    labels = sess.scalars(select(models.Label))
    for label in labels:
        sess.delete(label)

    sess.commit()

    # clean redis
    jobs.connect_to_redis()
    jobs.r.flushdb()


def test_generate_segmentation_data(
    client: Client,
    n_images: int = 10,
    n_annotations: int = 2,
    n_labels: int = 2,
):
    """Check that our generated dataset correctly matches our input parameters"""

    dataset = generate_segmentation_data(
        client=client,
        dataset_name=dset_name,
        n_images=n_images,
        n_annotations=n_annotations,
        n_labels=n_labels,
    )

    sample_images = dataset.get_images()
    assert (
        len(sample_images) == n_images
    ), "Number of images doesn't match the test input"

    for image in dataset.get_images():
        uid = image.uid
        sample_gt = dataset.get_groundtruth(uid)

        sample_annotations = sample_gt.annotations
        sample_mask_size = _mask_bytes_to_pil(
            b64decode(sample_annotations[0].raster.mask)
        ).size

        sample_image = ImageMetadata.from_datum(sample_gt.datum)
        sample_image_size = (sample_image.width, sample_image.height)

        assert (
            len(sample_annotations) == n_annotations
        ), "Number of annotations doesn't match the test input"
        assert (
            len(sample_annotations[0].labels) == n_labels
        ), "Number of labels on the sample annotation doesn't match the test input"
        assert (
            sample_image_size == sample_mask_size
        ), f"Image is size {sample_image_size}, but mask is size {sample_mask_size}"


def test_generate_prediction_data(client: Client):
    """Check that our generated predictions correctly matches our input parameters"""

    n_images = 2
    n_annotations = 2
    n_labels = 2
    dset_name = "dset"
    model_name = "model"

    dataset = generate_segmentation_data(
        client=client,
        dataset_name=dset_name,
        n_images=n_images,
        n_annotations=n_annotations,
        n_labels=n_labels,
    )

    assert len(dataset.get_images()) == n_images
    assert len(dataset.get_datums()) == n_images

    model = generate_prediction_data(
        client=client,
        dataset=dataset,
        model_name=model_name,
        n_annotations=2,
        n_labels=2,
    )

    eval_job = model.evaluate_detection(
        dataset=dataset,
        annotation_type=AnnotationType.BOX,
        iou_thresholds_to_compute=[0, 1],
        iou_thresholds_to_keep=[0, 1],
        label_key="k1",
        timeout=30,
    )

    assert eval_job.status == JobStatus.DONE

    settings = asdict(eval_job.settings)
    settings.pop("id")
    assert settings == {
        "model": model_name,
        "dataset": dset_name,
        "settings": {
            "task_type": TaskType.DETECTION.value,
            "parameters": {
                "iou_thresholds_to_compute": [0.0, 1.0],
                "iou_thresholds_to_keep": [0.0, 1.0],
            },
            "filters": {
                "annotations": {
                    "allow_conversion": False,
                    "annotation_types": ["box"],
                    "geo": [],
                    "geometry": [],
                    "json_": [],
                    "metadata": [],
                    "task_types": [],
                },
                "labels": {"ids": [], "keys": ["k1"], "labels": []},
            },
        },
    }
    assert len(eval_job.metrics) > 0
