import io
import time
from base64 import b64decode

import PIL
import pytest

from velour.client import Client
from velour.data_generation import (
    generate_prediction_data,
    generate_segmentation_data,
)
from velour.enums import JobStatus

dset_name = "test_dataset"


def _mask_bytes_to_pil(mask_bytes):
    with io.BytesIO(mask_bytes) as f:
        return PIL.Image.open(f)


@pytest.fixture
def client():
    return Client(host="http://localhost:8000")


def test_generate_segmentation_data(client: Client):
    """Check that our generated dataset correctly matches our input parameters"""

    n_images = 10
    n_annotations = 10
    n_labels = 2

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
        sample_image_size = (
            sample_gt.datum.metadata[1].value,
            sample_gt.datum.metadata[0].value,
        )

        assert (
            len(sample_annotations) == n_annotations
        ), "Number of annotations doesn't match the test input"
        assert (
            len(sample_annotations[0].labels) == n_labels
        ), "Number of labels on the sample annotation doesn't match the test input"
        assert (
            sample_image_size == sample_mask_size
        ), f"Image is size {sample_image_size}, but mask is size {sample_mask_size}"

    client.delete_dataset(dset_name, timeout=30)


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

    eval_job = model.evaluate_ap(
        dataset=dataset,
        iou_thresholds=[0, 1],
        ious_to_keep=[0, 1],
        label_key="k1",
    )

    # sleep to give the backend time to compute
    time.sleep(1)
    assert eval_job.status == JobStatus.DONE

    settings = eval_job.settings
    settings.pop("id")
    assert settings == {
        "model": model_name,
        "dataset": dset_name,
        "task_type": "detection",
        "target_type": "box",
        "label_key": "k1",
    }
    assert len(eval_job.metrics) > 0

    client.delete_dataset(dset_name, timeout=30)
