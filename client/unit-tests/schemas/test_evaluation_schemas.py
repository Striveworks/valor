from valor import enums, schemas


def test_evaluation_request():
    params = {
        "dataset_names": ["ds"],
        "model_names": ["md"],
        "filter": {},
        "parameters": {
            "task_type": enums.TaskType.OBJECT_DETECTION.value,
            "convert_annotations_to_type": enums.AnnotationType.BOX.value,
        },
    }
    schemas.EvaluationRequest(**params)
