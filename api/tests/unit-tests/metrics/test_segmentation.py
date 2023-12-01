from velour_api import enums
import velour_api.backend.metrics.segmentation as segmentation
import pytest


class MockNestedObject:
    def __init__(self, **response):
        for k, v in response.items():
            if isinstance(v, dict):
                self.__dict__[k] = MockNestedObject(**v)
            elif isinstance(v, list):
                self.__dict__[k] = [MockNestedObject(**item) for item in v]
            else:
                self.__dict__[k] = v


def test_create_semantic_segmentation_evaluation():
    # check wrong task type
    mock_job_request = MockNestedObject(
        **{
            "model": "model1",
            "dataset": "dataset1",
            "task_type": enums.TaskType.CLASSIFICATION,
            "settings": None,
        }
    )

    with pytest.raises(TypeError):
        segmentation.create_semantic_segmentation_evaluation(
            db=None, job_request=mock_job_request
        )

    # check what happens if we pass parameters
    mock_job_request = MockNestedObject(
        **{
            "model": "model1",
            "dataset": "dataset1",
            "task_type": enums.TaskType.SEGMENTATION,
            "settings": {"parameters": "fake_parms"},
        }
    )

    with pytest.raises(ValueError):
        segmentation.create_semantic_segmentation_evaluation(
            db=None, job_request=mock_job_request
        )

    # check invalid filter type
    mock_job_request = MockNestedObject(
        **{
            "model": "model1",
            "dataset": "dataset1",
            "task_type": enums.TaskType.SEGMENTATION,
            "settings": {
                "parameters": None,
                "filters": {"dataset_names": "fake_dataset_name"},
            },
        }
    )

    with pytest.raises(ValueError):
        segmentation.create_semantic_segmentation_evaluation(
            db=None, job_request=mock_job_request
        )
