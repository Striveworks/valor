import io
from base64 import b64decode

import PIL
import pytest
from velour.client import Client
from velour.data_generation import generate_segmentation_data

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

    client.delete_dataset(dset_name)
