import time

import pytest

from velour.client import Client
from velour.integrations.chariot import chariot_ds_to_velour_ds

chariot_client = pytest.importorskip("chariot.client")
chariot_datasets = pytest.importorskip("chariot.datasets.dataset")

# Reference: https://github.com/Striveworks/chariot/blob/main/py/libs/sdk/integration_tests/test_datasets_cv.py
