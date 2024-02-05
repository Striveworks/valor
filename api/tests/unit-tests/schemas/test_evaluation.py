import pytest
from pydantic import ValidationError

from velour_api import enums, schemas


def test_EvaluationParameters():
    schemas.EvaluationParameters(task_type=enums.TaskType.CLASSIFICATION)

    schemas.EvaluationParameters(
        task_type=enums.TaskType.OBJECT_DETECTION,
        iou_thresholds_to_compute=[0.2, 0.6],
        iou_thresholds_to_return=[],
    )

    schemas.EvaluationParameters(
        task_type=enums.TaskType.OBJECT_DETECTION,
        iou_thresholds_to_compute=[],
        iou_thresholds_to_return=[],
    )

    schemas.EvaluationParameters(
        task_type=enums.TaskType.OBJECT_DETECTION,
        iou_thresholds_to_compute=[],
        iou_thresholds_to_return=[],
        label_map=[
            [["class_name", "maine coon cat"], ["class", "cat"]],
            [["class", "siamese cat"], ["class", "cat"]],
            [["class", "british shorthair"], ["class", "cat"]],
        ],
    )

    with pytest.raises(ValidationError):
        schemas.EvaluationParameters(
            task_type=enums.TaskType.CLASSIFICATION,
            iou_thresholds_to_compute=[0.2, 0.6],
            iou_thresholds_to_return=[],
        )

    with pytest.raises(ValidationError):
        schemas.EvaluationParameters(
            task_type=enums.TaskType.SEMANTIC_SEGMENTATION,
            iou_thresholds_to_compute=[0.2, 0.6],
            iou_thresholds_to_return=[],
        )

    with pytest.raises(ValidationError):
        schemas.EvaluationParameters(
            task_type=enums.TaskType.OBJECT_DETECTION,
            iou_thresholds_to_compute=None,
            iou_thresholds_to_return=[0.2],
        )

    with pytest.raises(ValidationError):
        schemas.EvaluationParameters(
            task_type=enums.TaskType.OBJECT_DETECTION,
            iou_thresholds_to_compute=None,
            iou_thresholds_to_return=0.2,
        )

    with pytest.raises(ValidationError):
        schemas.EvaluationParameters(
            task_type=enums.TaskType.OBJECT_DETECTION,
            iou_thresholds_to_compute=[0.2, "test"],
            iou_thresholds_to_return=[],
        )

    with pytest.raises(ValidationError):
        schemas.EvaluationParameters(
            task_type=enums.TaskType.OBJECT_DETECTION,
            iou_thresholds_to_compute=[0.2, "test"],
            iou_thresholds_to_return=[],
            label_map={"not a": "valid grouper"},
        )


def test_EvaluationRequest():
    schemas.EvaluationRequest(
        model_names="name",
        datum_filter=schemas.Filter(),
        parameters=schemas.EvaluationParameters(
            task_type=enums.TaskType.CLASSIFICATION
        ),
    )
    schemas.EvaluationRequest(
        model_names=["name"],
        datum_filter=schemas.Filter(),
        parameters=schemas.EvaluationParameters(
            task_type=enums.TaskType.CLASSIFICATION
        ),
    )
    schemas.EvaluationRequest(
        model_names=["name", "other"],
        datum_filter=schemas.Filter(),
        parameters=schemas.EvaluationParameters(
            task_type=enums.TaskType.CLASSIFICATION
        ),
    )

    # test missing args
    with pytest.raises(ValidationError):
        schemas.EvaluationRequest(
            model_filter=None,
            datum_filter=schemas.Filter(),
            parameters=schemas.EvaluationParameters(
                task_type=enums.TaskType.CLASSIFICATION
            ),
        )
    with pytest.raises(ValidationError):
        schemas.EvaluationRequest(
            model_names=["name"],
            datum_filter=None,
            parameters=schemas.EvaluationParameters(
                task_type=enums.TaskType.CLASSIFICATION
            ),
        )
    with pytest.raises(ValidationError):
        schemas.EvaluationRequest(
            model_names=["name"],
            datum_filter=schemas.Filter(),
            parameters=None,
        )

    # test `model_names` validator
    with pytest.raises(ValidationError):
        schemas.EvaluationRequest(
            model_names=[],
            datum_filter=schemas.Filter(),
            parameters=schemas.EvaluationParameters(
                task_type=enums.TaskType.CLASSIFICATION
            ),
        )

    # test `datum_filter` validator
    with pytest.raises(ValidationError):
        schemas.EvaluationRequest(
            model_filter=schemas.Filter(),
            datum_filter=schemas.Filter(
                task_types=[enums.TaskType.CLASSIFICATION]
            ),
            parameters=schemas.EvaluationParameters(
                task_type=enums.TaskType.CLASSIFICATION
            ),
        )


def test_EvaluationResponse():
    schemas.EvaluationResponse(
        id=1,
        model_name="test",
        datum_filter=schemas.Filter(),
        parameters=schemas.EvaluationParameters(
            task_type=enums.TaskType.CLASSIFICATION
        ),
        status=enums.EvaluationStatus.DONE,
        metrics=[],
        confusion_matrices=[],
    )

    # test missing evaluation_id
    with pytest.raises(ValidationError):
        schemas.EvaluationResponse(
            id=None,
            model_name="test",
            datum_filter=schemas.Filter(),
            parameters=schemas.EvaluationParameters(
                task_type=enums.TaskType.CLASSIFICATION
            ),
            status=enums.EvaluationStatus.DONE,
            metrics=[],
            confusion_matrices=[],
        )

    # test missing model name
    with pytest.raises(ValidationError):
        schemas.EvaluationResponse(
            id=1,
            model_name=None,
            datum_filter=schemas.Filter(),
            parameters=schemas.EvaluationParameters(
                task_type=enums.TaskType.CLASSIFICATION
            ),
            status=enums.EvaluationStatus.DONE,
            metrics=[],
            confusion_matrices=[],
        )

    # test missing EvaluationParameters
    with pytest.raises(ValidationError):
        schemas.EvaluationResponse(
            id=1,
            model_name="name",
            datum_filter=schemas.Filter(),
            parameters=None,
            status=enums.EvaluationStatus.DONE,
            metrics=[],
            confusion_matrices=[],
        )

    # test missing EvaluationStatus
    with pytest.raises(ValidationError):
        schemas.EvaluationResponse(
            id=1,
            model_name="name",
            datum_filter=schemas.Filter(),
            parameters=schemas.EvaluationParameters(
                task_type=enums.TaskType.CLASSIFICATION
            ),
            status=None,
            metrics=[],
            confusion_matrices=[],
        )
