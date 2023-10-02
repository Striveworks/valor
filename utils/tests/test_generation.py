import io
from base64 import b64decode

import PIL
import pytest
from utils.src.generation import generate_segmentation_data

from velour.client import Client

LOCAL_HOST = "http://localhost:8000"
client = Client(LOCAL_HOST)


def _mask_bytes_to_pil(mask_bytes):
    with io.BytesIO(mask_bytes) as f:
        return PIL.Image.open(f)


def test_generate_segmentation_data():
    """Check that our generated dataset correctly matches our input parameters"""
    with pytest.raises(ValueError):
        dataset = generate_segmentation_data(
            dataset_name="test_generate_segmentation_data",
            metadata_json_path="utils/sample_images/sample_coco_images.json",
            client=client,
            n_images=10,
            n_annotations=10,
            n_labels=2,
        )

    dataset = generate_segmentation_data(
        dataset_name="test_generate_segmentation_data",
        metadata_json_path="utils/sample_images/sample_coco_images.json",
        client=client,
        n_images=5,
        n_annotations=10,
        n_labels=2,
    )

    sample_images = dataset.get_images()
    assert (
        len(sample_images) == 5
    ), "Number of images doesn't match the test input"

    for image in dataset.get_images()[:5]:
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
            len(sample_annotations) == 10
        ), "Number of annotations doesn't match the test input"
        assert (
            len(sample_annotations[0].labels) == 2
        ), "Number of labels on the sample annotation doesn't match the test input"
        assert (
            sample_image_size == sample_mask_size
        ), f"Image is size {sample_image_size}, but mask is size {sample_mask_size}"

    client.delete_dataset("test_generate_segmentation_data")
