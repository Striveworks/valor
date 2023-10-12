# NOTE: These tests aren't run automatically on each commit. They are intended to be manually kicked-off using the [GitHub UI](https://leonardomontini.dev/github-action-manual-trigger/)

import pytest
from test_data_generation import test_generate_segmentation_data

from velour.client import Client


@pytest.fixture
def client():
    return Client(host="http://localhost:8000")


def test_large_dataset_upload(client: Client):
    """Tests the upload of a large dataset to velour (runtime: ~20 minutes)"""
    test_generate_segmentation_data(
        n_images=1000, n_annotations=10, n_labels=2
    )
