from datetime import datetime

import pytest
from pydantic import ValidationError

from valor_api import enums, schemas


def test_EvaluationParameters():
    schemas.EvaluationParameters(
        task_type=enums.TaskType.CLASSIFICATION,
        metrics=[
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROCAUC",
            "PrecisionRecallCurve",
        ],
    )

    schemas.EvaluationParameters(
        task_type=enums.TaskType.OBJECT_DETECTION,
        iou_thresholds_to_compute=[0.2, 0.6],
        iou_thresholds_to_return=[],
        metrics=[
            "AP",
            "AR",
            "mAP",
            "APAveragedOverIOUs",
            "mAR",
            "mAPAveragedOverIOUs",
        ],
    )

    schemas.EvaluationParameters(
        task_type=enums.TaskType.OBJECT_DETECTION,
        iou_thresholds_to_compute=[],
        iou_thresholds_to_return=[],
        metrics=[
            "AP",
            "AR",
            "mAP",
            "APAveragedOverIOUs",
            "mAR",
            "mAPAveragedOverIOUs",
        ],
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
        metrics=[
            "AP",
            "AR",
            "mAP",
            "APAveragedOverIOUs",
            "mAR",
            "mAPAveragedOverIOUs",
        ],
    )

    with pytest.raises(ValidationError):
        schemas.EvaluationParameters(
            task_type=enums.TaskType.CLASSIFICATION,
            iou_thresholds_to_compute=[0.2, 0.6],
            iou_thresholds_to_return=[],
            metrics=[
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
                "ROCAUC",
                "PrecisionRecallCurve",
            ],
        )

    with pytest.raises(ValidationError):
        schemas.EvaluationParameters(
            task_type=enums.TaskType.SEMANTIC_SEGMENTATION,
            iou_thresholds_to_compute=[0.2, 0.6],
            iou_thresholds_to_return=[],
            metrics=["IOU"],
        )

    with pytest.raises(ValidationError):
        schemas.EvaluationParameters(
            task_type=enums.TaskType.OBJECT_DETECTION,
            iou_thresholds_to_compute=None,
            iou_thresholds_to_return=[0.2],
            metrics=[
                "AP",
                "AR",
                "mAP",
                "APAveragedOverIOUs",
                "mAR",
                "mAPAveragedOverIOUs",
            ],
        )

    with pytest.raises(ValidationError):
        schemas.EvaluationParameters(
            task_type=enums.TaskType.OBJECT_DETECTION,
            iou_thresholds_to_compute=None,
            iou_thresholds_to_return=0.2,  # type: ignore - purposefully throwing error,
            metrics=[
                "AP",
                "AR",
                "mAP",
                "APAveragedOverIOUs",
                "mAR",
                "mAPAveragedOverIOUs",
            ],
        )

    with pytest.raises(ValidationError):
        schemas.EvaluationParameters(
            task_type=enums.TaskType.OBJECT_DETECTION,
            iou_thresholds_to_compute=[0.2, "test"],  # type: ignore - purposefully throwing error
            iou_thresholds_to_return=[],
            metrics=[
                "AP",
                "AR",
                "mAP",
                "APAveragedOverIOUs",
                "mAR",
                "mAPAveragedOverIOUs",
            ],
        )

    with pytest.raises(ValidationError):
        schemas.EvaluationParameters(
            task_type=enums.TaskType.OBJECT_DETECTION,
            iou_thresholds_to_compute=[0.2, "test"],  # type: ignore - purposefully throwing error
            iou_thresholds_to_return=[],
            label_map={"not a": "valid grouper"},  # type: ignore - purposefully throwing error
            metrics=[
                "AP",
                "AR",
                "mAP",
                "APAveragedOverIOUs",
                "mAR",
                "mAPAveragedOverIOUs",
            ],
        )


def test_EvaluationRequest():
    schemas.EvaluationRequest(
        model_names=["name"],
        datum_filter=schemas.Filter(),
        parameters=schemas.EvaluationParameters(
            task_type=enums.TaskType.CLASSIFICATION,
            metrics=[
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
                "ROCAUC",
                "PrecisionRecallCurve",
            ],
        ),
        meta={},
    )
    schemas.EvaluationRequest(
        model_names=["name"],
        datum_filter=schemas.Filter(),
        parameters=schemas.EvaluationParameters(
            task_type=enums.TaskType.CLASSIFICATION,
            metrics=[
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
                "ROCAUC",
                "PrecisionRecallCurve",
            ],
        ),
        meta={},
    )
    schemas.EvaluationRequest(
        model_names=["name", "other"],
        datum_filter=schemas.Filter(),
        parameters=schemas.EvaluationParameters(
            task_type=enums.TaskType.CLASSIFICATION,
            metrics=[
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
                "ROCAUC",
                "PrecisionRecallCurve",
            ],
        ),
        meta={},
    )

    # test missing args
    with pytest.raises(ValidationError):
        schemas.EvaluationRequest(
            model_filter=None,  # type: ignore - purposefully throwing error
            datum_filter=schemas.Filter(),
            parameters=schemas.EvaluationParameters(
                task_type=enums.TaskType.CLASSIFICATION,
                metrics=[
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1",
                    "ROCAUC",
                    "PrecisionRecallCurve",
                ],
            ),
        )
    with pytest.raises(ValidationError):
        schemas.EvaluationRequest(
            model_names=["name"],
            datum_filter=None,  # type: ignore - purposefully throwing error
            parameters=schemas.EvaluationParameters(
                task_type=enums.TaskType.CLASSIFICATION,
                metrics=[
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1",
                    "ROCAUC",
                    "PrecisionRecallCurve",
                ],
            ),
        )
    with pytest.raises(ValidationError):
        schemas.EvaluationRequest(
            model_names=["name"],
            datum_filter=schemas.Filter(),
            parameters=None,  # type: ignore - purposefully throwing error
        )

    # test `model_names` validator
    with pytest.raises(ValidationError):
        schemas.EvaluationRequest(
            model_names=[],
            datum_filter=schemas.Filter(),
            parameters=schemas.EvaluationParameters(
                task_type=enums.TaskType.CLASSIFICATION,
                metrics=[
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1",
                    "ROCAUC",
                    "PrecisionRecallCurve",
                ],
            ),
            meta={},
        )

    # test `datum_filter` validator
    with pytest.raises(ValidationError):
        schemas.EvaluationRequest(
            model_filter=schemas.Filter(),  # type: ignore - purposefully throwing error
            datum_filter=schemas.Filter(
                task_types=[enums.TaskType.CLASSIFICATION]
            ),
            parameters=schemas.EvaluationParameters(
                task_type=enums.TaskType.CLASSIFICATION,
                metrics=[
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1",
                    "ROCAUC",
                    "PrecisionRecallCurve",
                ],
            ),
        )


def test_EvaluationResponse():
    schemas.EvaluationResponse(
        id=1,
        model_name="test",
        datum_filter=schemas.Filter(),
        parameters=schemas.EvaluationParameters(
            task_type=enums.TaskType.CLASSIFICATION,
            metrics=[
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
                "ROCAUC",
                "PrecisionRecallCurve",
            ],
        ),
        status=enums.EvaluationStatus.DONE,
        metrics=[],
        confusion_matrices=[],
        created_at=datetime.now(),
        meta={},
    )

    # test missing evaluation_id
    with pytest.raises(ValidationError):
        schemas.EvaluationResponse(
            id=None,  # type: ignore - purposefully throwing error
            model_name="test",
            datum_filter=schemas.Filter(),
            parameters=schemas.EvaluationParameters(
                task_type=enums.TaskType.CLASSIFICATION,
                metrics=[
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1",
                    "ROCAUC",
                    "PrecisionRecallCurve",
                ],
            ),
            status=enums.EvaluationStatus.DONE,
            metrics=[],
            confusion_matrices=[],
            created_at=datetime.now(),
            meta={},
        )

    # test missing model name
    with pytest.raises(ValidationError):
        schemas.EvaluationResponse(
            id=1,
            model_name=None,  # type: ignore - purposefully throwing error
            datum_filter=schemas.Filter(),
            parameters=schemas.EvaluationParameters(
                task_type=enums.TaskType.CLASSIFICATION,
                metrics=[
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1",
                    "ROCAUC",
                    "PrecisionRecallCurve",
                ],
            ),
            status=enums.EvaluationStatus.DONE,
            metrics=[],
            confusion_matrices=[],
            created_at=datetime.now(),
            meta={},
        )

    # test missing EvaluationParameters
    with pytest.raises(ValidationError):
        schemas.EvaluationResponse(
            id=1,
            model_name="name",
            datum_filter=schemas.Filter(),
            parameters=None,  # type: ignore - purposefully throwing error
            status=enums.EvaluationStatus.DONE,
            metrics=[],
            confusion_matrices=[],
            created_at=datetime.now(),
            meta={},
        )

    # test missing EvaluationStatus
    with pytest.raises(ValidationError):
        schemas.EvaluationResponse(
            id=1,
            model_name="name",
            datum_filter=schemas.Filter(),
            parameters=schemas.EvaluationParameters(
                task_type=enums.TaskType.CLASSIFICATION,
                metrics=[
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1",
                    "ROCAUC",
                    "PrecisionRecallCurve",
                ],
            ),
            status=None,  # type: ignore - purposefully throwing error
            metrics=[],
            confusion_matrices=[],
            created_at=datetime.now(),
            meta={},
        )
