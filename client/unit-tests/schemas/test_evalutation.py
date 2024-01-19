from velour import enums, schemas


def test_evaluation_evaluation_job():
    params = {
        "model_filter": {
            "model_names": ["md"],
        },
        "dataset_filter": {
            "dataset_names": ["ds"],
        },
        "parameters": {
            "task_type": enums.TaskType.DETECTION.value,
            "force_annotation_type": enums.AnnotationType.BOX.value,
        },
    }
    schemas.EvaluationRequest(**params)
