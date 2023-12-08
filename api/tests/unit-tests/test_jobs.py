from unittest.mock import MagicMock

import pytest

from velour_api.crud import jobs
from velour_api.crud.jobs import Job


@pytest.fixture(autouse=True)
def teardown():
    # reset the global redis instance (which may have been set by other tests)
    jobs.r = None


@pytest.fixture
def dataset_name() -> str:
    return "dataset1"


@pytest.fixture
def model_name() -> str:
    return "model1"


@pytest.fixture
def evaluation_id() -> int:
    return 1234


def test_generate_uuid(dataset_name, model_name, evaluation_id):
    assert Job.generate_uuid(dataset_name=dataset_name) == f"{dataset_name}+None+None"
    assert Job.generate_uuid(model_name=model_name) == f"None+{model_name}+None"
    assert Job.generate_uuid(
        dataset_name=dataset_name,
        model_name=model_name
    ) == f"{dataset_name}+{model_name}+None"
    assert Job.generate_uuid(
        dataset_name=dataset_name, 
        model_name=model_name,
        evaluation_id=evaluation_id,
    ) == f"{dataset_name}+{model_name}+{evaluation_id}"

    assert Job.generate_uuid(evaluation_id=evaluation_id) == f"None+None+{evaluation_id}"
    assert Job.generate_uuid(dataset_name=dataset_name, evaluation_id=evaluation_id) == f"{dataset_name}+None+{evaluation_id}"
    assert Job.generate_uuid(model_name=model_name, evaluation_id=evaluation_id) == f"None+{model_name}+{evaluation_id}"
