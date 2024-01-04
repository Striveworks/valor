import pytest

from velour_api.backend.core import create_or_get_evaluation
import velour_api.backend.metrics.segmentation as segmentation
from velour_api import enums, schemas


def test_create_semantic_segmentation_evaluation():
    # check wrong task type
    mock_job_request = schemas.EvaluationJob(
        **{
            "model": "model1",
            "dataset": "dataset1",
            "task_type": enums.TaskType.CLASSIFICATION,
            "settings": {"parameters": None, "filters": None},
        }
    )

    with pytest.raises(TypeError):
        create_or_get_evaluation(
            db=None, job_request=mock_job_request
        )

    # check passing parametric inputs into a semantic segmentation
    mock_job_request = schemas.EvaluationJob(
        **{
            "model": "model1",
            "dataset": "dataset1",
            "task_type": enums.TaskType.SEGMENTATION,
            "settings": schemas.EvaluationSettings(
                parameters=schemas.DetectionParameters(
                    iou_thresholds_to_compute=[0.2, 0.6],
                    iou_thresholds_to_keep=[0.2],
                ),
            ),
        }
    )

    with pytest.raises(TypeError):
        create_or_get_evaluation(
            db=None, job_request=mock_job_request
        )

    # check invalid filter type for semantic segmentation tasks
    mock_job_request = schemas.EvaluationJob(
        **{
            "model": "model1",
            "dataset": "dataset1",
            "task_type": enums.TaskType.SEGMENTATION,
            "settings": schemas.EvaluationSettings(
                filters=schemas.Filter(
                    annotation_types=[enums.AnnotationType.BOX],
                    label_keys=["class"],
                ),
            ),
        }
    )

    with pytest.raises(TypeError):
        create_or_get_evaluation(
            db=None, job_request=mock_job_request
        )
