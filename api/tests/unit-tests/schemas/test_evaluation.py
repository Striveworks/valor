from datetime import datetime

import pytest
from pydantic import ValidationError

from valor_api import enums, schemas
from valor_api.enums import MetricType, ROUGEType


@pytest.fixture
def llm_api_params():
    return {
        "client": "openai",
        "data": {
            "seed": 2024,
            "model": "gpt-4o-2024-05-13",
        },
    }


def test_EvaluationParameters(llm_api_params):
    schemas.EvaluationParameters(
        task_type=enums.TaskType.CLASSIFICATION,
    )

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

    # If no llm-guided metrics are requested, then llm_api_params is not required.
    schemas.EvaluationParameters(
        task_type=enums.TaskType.TEXT_GENERATION,
        metrics_to_return=[
            MetricType.BLEU,
            MetricType.ROUGE,
        ],
    )

    # If llm-guided metrics are requested, then llm_api_params is required.
    schemas.EvaluationParameters(
        task_type=enums.TaskType.TEXT_GENERATION,
        metrics_to_return=[
            MetricType.AnswerRelevance,
            MetricType.Bias,
            MetricType.BLEU,
            MetricType.Coherence,
            MetricType.ContextRelevance,
            MetricType.Hallucination,
            MetricType.ROUGE,
            MetricType.Toxicity,
        ],
        llm_api_params=llm_api_params,
    )

    # Test with metric parameters
    schemas.EvaluationParameters(
        task_type=enums.TaskType.TEXT_GENERATION,
        metrics_to_return=[
            MetricType.AnswerRelevance,
            MetricType.Bias,
            MetricType.BLEU,
            MetricType.Coherence,
            MetricType.ContextRelevance,
            MetricType.Hallucination,
            MetricType.ROUGE,
            MetricType.Toxicity,
        ],
        llm_api_params=llm_api_params,
        bleu_weights=[0.5, 0.25, 0.25, 0],
        rouge_types=[ROUGEType.ROUGE1, ROUGEType.ROUGELSUM],
        rouge_use_stemmer=True,
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
            iou_thresholds_to_return=0.2,  # type: ignore - purposefully throwing error,
        )

    with pytest.raises(ValidationError):
        schemas.EvaluationParameters(
            task_type=enums.TaskType.OBJECT_DETECTION,
            iou_thresholds_to_compute=[0.2, "test"],  # type: ignore - purposefully throwing error
            iou_thresholds_to_return=[],
        )

    with pytest.raises(ValidationError):
        schemas.EvaluationParameters(
            task_type=enums.TaskType.OBJECT_DETECTION,
            iou_thresholds_to_compute=[0.2, "test"],  # type: ignore - purposefully throwing error
            iou_thresholds_to_return=[],
            label_map={"not a": "valid grouper"},  # type: ignore - purposefully throwing error
        )

    # For TaskType.TEXT_GENERATION, metrics_to_return must be provided.
    with pytest.raises(ValidationError):
        schemas.EvaluationParameters(
            task_type=enums.TaskType.TEXT_GENERATION,
        )

    # If any llm-guided metrics are requested, then llm_api_params must be provided.
    with pytest.raises(ValidationError):
        schemas.EvaluationParameters(
            task_type=enums.TaskType.TEXT_GENERATION,
            metrics_to_return=[
                MetricType.AnswerRelevance,
                MetricType.Bias,
                MetricType.BLEU,
                MetricType.Coherence,
                MetricType.ContextRelevance,
                MetricType.Hallucination,
                MetricType.ROUGE,
                MetricType.Toxicity,
            ],
        )

    # BLEU weights must be 0 <= weight <= 1.
    with pytest.raises(ValidationError):
        schemas.EvaluationParameters(
            task_type=enums.TaskType.TEXT_GENERATION,
            metrics_to_return=[
                MetricType.Bias,
                MetricType.BLEU,
            ],
            llm_api_params=llm_api_params,
            bleu_weights=[1.1, 0.3, -0.5, 0.1],
        )

    # BLEU weights must sum to 1.
    with pytest.raises(ValidationError):
        schemas.EvaluationParameters(
            task_type=enums.TaskType.TEXT_GENERATION,
            metrics_to_return=[
                MetricType.AnswerRelevance,
                MetricType.Bias,
                MetricType.BLEU,
                MetricType.Coherence,
                MetricType.ContextRelevance,
                MetricType.Hallucination,
                MetricType.ROUGE,
                MetricType.Toxicity,
            ],
            llm_api_params=llm_api_params,
            bleu_weights=[0.5, 0.25, 0.25, 0.25],
        )


def test_EvaluationRequest():
    schemas.EvaluationRequest(
        dataset_names=["ds"],
        model_names=["name"],
        filters=schemas.Filter(),
        parameters=schemas.EvaluationParameters(
            task_type=enums.TaskType.CLASSIFICATION,
        ),
    )
    schemas.EvaluationRequest(
        dataset_names=["ds"],
        model_names=["name"],
        filters=schemas.Filter(),
        parameters=schemas.EvaluationParameters(
            task_type=enums.TaskType.CLASSIFICATION,
        ),
    )
    schemas.EvaluationRequest(
        dataset_names=["ds"],
        model_names=["name", "other"],
        filters=schemas.Filter(),
        parameters=schemas.EvaluationParameters(
            task_type=enums.TaskType.CLASSIFICATION,
        ),
    )

    # test missing args
    with pytest.raises(ValidationError):
        schemas.EvaluationRequest(
            dataset_names=["ds"],
            model_names=None,  # type: ignore - purposefully throwing error
            filters=schemas.Filter(),
            parameters=schemas.EvaluationParameters(
                task_type=enums.TaskType.CLASSIFICATION,
            ),
        )
    with pytest.raises(ValidationError):
        schemas.EvaluationRequest(
            dataset_names=["ds"],
            model_names=["name"],
            filters=None,  # type: ignore - purposefully throwing error
            parameters=schemas.EvaluationParameters(
                task_type=enums.TaskType.CLASSIFICATION,
            ),
        )
    with pytest.raises(ValidationError):
        schemas.EvaluationRequest(
            dataset_names=["ds"],
            model_names=["name"],
            filters=schemas.Filter(),
            parameters=None,  # type: ignore - purposefully throwing error
        )

    # test `dataset_names` validator
    with pytest.raises(ValidationError):
        schemas.EvaluationRequest(
            dataset_names=[],
            model_names=["md"],
            filters=schemas.Filter(),
            parameters=schemas.EvaluationParameters(
                task_type=enums.TaskType.CLASSIFICATION
            ),
        )

    # test `model_names` validator
    with pytest.raises(ValidationError):
        schemas.EvaluationRequest(
            dataset_names=["ds"],
            model_names=[],
            filters=schemas.Filter(),
            parameters=schemas.EvaluationParameters(
                task_type=enums.TaskType.CLASSIFICATION,
            ),
        )

    # test `filters` validator
    with pytest.raises(ValidationError):
        schemas.EvaluationRequest(
            model_filter=schemas.Filter(),  # type: ignore - purposefully throwing error
            filters=schemas.Filter(),
            parameters=schemas.EvaluationParameters(
                task_type=enums.TaskType.CLASSIFICATION,
            ),
        )


def test_EvaluationResponse():
    schemas.EvaluationResponse(
        id=1,
        dataset_names=["ds"],
        model_name="test",
        filters=schemas.Filter(),
        parameters=schemas.EvaluationParameters(
            task_type=enums.TaskType.CLASSIFICATION,
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
            dataset_names=["ds"],
            model_name="test",
            filters=schemas.Filter(),
            parameters=schemas.EvaluationParameters(
                task_type=enums.TaskType.CLASSIFICATION,
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
            dataset_names=["ds"],
            model_name=None,  # type: ignore - purposefully throwing error
            filters=schemas.Filter(),
            parameters=schemas.EvaluationParameters(
                task_type=enums.TaskType.CLASSIFICATION,
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
            dataset_names=["ds"],
            model_name="name",
            filters=schemas.Filter(),
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
            dataset_names=["ds"],
            model_name="name",
            filters=schemas.Filter(),
            parameters=schemas.EvaluationParameters(
                task_type=enums.TaskType.CLASSIFICATION,
            ),
            status=None,  # type: ignore - purposefully throwing error
            metrics=[],
            confusion_matrices=[],
            created_at=datetime.now(),
            meta={},
        )
