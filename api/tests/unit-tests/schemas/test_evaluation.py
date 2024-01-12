import pytest
from pydantic import ValidationError

from velour_api import schemas, enums


def test_DetectionParameters():
    schemas.DetectionParameters()

    schemas.DetectionParameters(
        iou_thresholds_to_compute=[0.2, 0.6],
        iou_thresholds_to_return=[],
    )

    schemas.DetectionParameters(
        iou_thresholds_to_compute=[],
        iou_thresholds_to_return=[],
    )

    with pytest.raises(ValidationError):
        schemas.DetectionParameters(
            iou_thresholds_to_compute=None,
            iou_thresholds_to_return=[0.2],
        )

    with pytest.raises(ValidationError):
        schemas.DetectionParameters(
            iou_thresholds_to_compute=None,
            iou_thresholds_to_return=0.2,
        )

    with pytest.raises(ValidationError):
        schemas.DetectionParameters(
            iou_thresholds_to_compute=[0.2, "test"],
            iou_thresholds_to_return=[],
        )


def test_EvaluationParameters():
    schemas.EvaluationParameters()
    schemas.EvaluationParameters(
        parameters=schemas.DetectionParameters(
            iou_thresholds_to_compute=[0.2, 0.6],
            iou_thresholds_to_return=[],
        ),
    )
    schemas.EvaluationParameters(
        parameters=schemas.DetectionParameters(
            iou_thresholds_to_compute=[],
            iou_thresholds_to_return=[],
        ),
    )
    schemas.EvaluationParameters(
        parameters=schemas.DetectionParameters(
            iou_thresholds_to_compute=[],
            iou_thresholds_to_return=[],
        ),
        filters=schemas.Filter(
            annotation_types=[enums.AnnotationType.BOX],
            label_keys=["class"],
        ),
    )

    with pytest.raises(ValidationError):
        schemas.EvaluationParameters(detection="random_string")


def test_EvaluationRequest():
    schemas.EvaluationRequest(
        model_filter=schemas.Filter(),
        evaluation_filter=schemas.Filter(task_types=[enums.TaskType.CLASSIFICATION]),
        parameters=schemas.EvaluationParameters(),
    )

    # test missing args
    with pytest.raises(ValidationError):
        schemas.EvaluationRequest(
            model_filter=None,
            evaluation_filter=schemas.Filter(task_types=[enums.TaskType.CLASSIFICATION]),
            parameters=schemas.EvaluationParameters(),
        )
    with pytest.raises(ValidationError):
        schemas.EvaluationRequest(
            model_filter=schemas.Filter(),
            evaluation_filter=None,
            parameters=schemas.EvaluationParameters(),
        )
    with pytest.raises(ValidationError):
        schemas.EvaluationRequest(
            model_filter=schemas.Filter(),
            evaluation_filter=schemas.Filter(task_types=[enums.TaskType.CLASSIFICATION]),
            parameters=None,
        )

    # test `model_filter` validator
    with pytest.raises(ValidationError):
        schemas.EvaluationRequest(
            model_filter=schemas.Filter(task_types=[enums.TaskType.CLASSIFICATION]),
            evaluation_filter=schemas.Filter(task_types=[enums.TaskType.CLASSIFICATION]),
            parameters=schemas.EvaluationParameters(),
        )
    with pytest.raises(ValidationError):
        schemas.EvaluationRequest(
            model_filter=schemas.Filter(annotation_types=[enums.AnnotationType.RASTER]),
            evaluation_filter=schemas.Filter(task_types=[enums.TaskType.CLASSIFICATION]),
            parameters=schemas.EvaluationParameters(),
        )

    # test `evaluation_filter` validator
    with pytest.raises(ValidationError):
        schemas.EvaluationRequest(
            model_filter=schemas.Filter(),
            evaluation_filter=schemas.Filter(),
            parameters=schemas.EvaluationParameters(),
        )
        req = schemas.EvaluationRequest(
            model_filter=schemas.Filter(),
            evaluation_filter=schemas.Filter(task_types=[enums.TaskType.DETECTION]),
            parameters=schemas.EvaluationParameters(),
        )
        assert req.parameters.detection is not None


def test_EvaluationResponse():
    schemas.EvaluationResponse(
        evaluation_id=1,
        model_name="model",
        model_filter=schemas.Filter(),
        evaluation_filter=schemas.Filter(),
        parameters=schemas.DetectionParameters(),
        status=enums.EvaluationStatus.DONE,
        metrics=[],
        confusion_matrices=[],
    )
    schemas.EvaluationResponse(
        evaluation_id=1,
        model_name="model",
        model_filter=schemas.Filter(),
        evaluation_filter=schemas.Filter(),
        parameters=None,
        status=enums.EvaluationStatus.DONE,
        metrics=[],
        confusion_matrices=[],
    )

    # test missing evaluation_id
    with pytest.raises(ValidationError):
        schemas.EvaluationResponse(
            evaluation_id=None,
            model_name="model",
            model_filter=schemas.Filter,
            evaluation_filter=schemas.Filter(),
            parameters=None,
            status=enums.EvaluationStatus.DONE,
            metrics=[],
            confusion_matrices=[],
        )

    # test missing model name
    with pytest.raises(ValidationError):
        schemas.EvaluationResponse(
            evaluation_id=1,
            model_name=None,
            model_filter=schemas.Filter,
            evaluation_filter=schemas.Filter(),
            parameters=None,
            status=enums.EvaluationStatus.DONE,
            metrics=[],
            confusion_matrices=[],
        )

    # test missing model_filters
    with pytest.raises(ValidationError):
        schemas.EvaluationResponse(
            evaluation_id=1,
            model_name="model",
            model_filter=None,
            evaluation_filter=schemas.Filter(),
            parameters=None,
            status=enums.EvaluationStatus.DONE,
            metrics=[],
            confusion_matrices=[],
        )
    with pytest.raises(ValidationError):
        schemas.EvaluationResponse(
            evaluation_id=1,
            model_name="model",
            model_filter=schemas.Filter(),
            evaluation_filter=None,
            parameters=None,
            status=enums.EvaluationStatus.DONE,
            metrics=[],
            confusion_matrices=[],
        )
