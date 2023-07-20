from velour_api import schemas, enums
from velour_api.backend.metrics.mot_metrics import (
    MOT_METRICS_NAMES,
    OBJECT_ID_LABEL_KEY,
    compute_mot_metrics,
)


def generate_mot_data(num_frames: int):
    """
    Create synthetic predictions and detections which are identical (perfect detector).
    Three objects in each frame, moving in different directions at constant speed.
    """

    create_img = lambda frame: schemas.Datum(  # noqa: E731
        uid="test",
        metadata=[
            schemas.MetaDatum(name="type", value="image"),
            schemas.MetaDatum(name="height", value=500),
            schemas.MetaDatum(name="width", value=500),
            schemas.MetaDatum(name="frame", value=frame),
        ]
    )
    create_label = lambda obj_id: schemas.Label(  # noqa: E731
        key=OBJECT_ID_LABEL_KEY, value=obj_id
    )
    create_scored_label = (
        lambda obj_id, score: schemas.ScoredLabel(  # noqa: E731
            label=schemas.Label(key=OBJECT_ID_LABEL_KEY, value=obj_id),
            score=score,
        )
    )

    # noqa: E731
    square = lambda x, y: schemas.BoundingBox(
        polygon=schemas.geometry.BasicPolygon(
            points=[
                schemas.geometry.Point(x=x,     y=y),
                schemas.geometry.Point(x=x+10,  y=y),
                schemas.geometry.Point(x=x,     y=y+10),
                schemas.geometry.Point(x=x+10,  y=y+10),
            ]
        )
    )
    # Square Boundary moving diagonally
    create_boundary_1 = lambda frame: square(  # noqa: E731
        5 * frame, 5 * frame
    )
    # Square Boundary moving horizontally
    create_boundary_2 = lambda frame: square(5 * frame, 0)  # noqa: E731
    # Square Boundary moving vertically
    create_boundary_3 = lambda frame: square(0, 5 * frame)  # noqa: E731

    predictions = []
    groundtruths = []
    for frame in range(1, num_frames + 1):
        boundary1 = create_boundary_1(frame)
        boundary2 = create_boundary_2(frame)
        boundary3 = create_boundary_3(frame)

        image = create_img(frame)

        scored_labels1 = [
            create_scored_label("object 1", 1.0), 
        ]
        scored_labels2 = [
            create_scored_label("object 2", 1.0), 
        ]
        scored_labels3 = [
            create_scored_label("object 3", 1.0), 
        ]

        labels1 = [create_label("object 1")]
        labels2 = [create_label("object 2")]
        labels3 = [create_label("object 3")]

        gts = schemas.GroundTruth(
            dataset_name="test",
            datum=image,
            annotations=[
                schemas.GroundTruthAnnotation(
                    labels=labels1,
                    annotation=schemas.Annotation(
                        task_type=enums.TaskType.DETECTION,
                        bounding_box=boundary1,
                    )
                ),
                schemas.GroundTruthAnnotation(
                    labels=labels2,
                    annotation=schemas.Annotation(
                        task_type=enums.TaskType.DETECTION,
                        bounding_box=boundary2,
                    )
                ),
                schemas.GroundTruthAnnotation(
                    labels=labels3,
                    annotation=schemas.Annotation(
                        task_type=enums.TaskType.DETECTION,
                        bounding_box=boundary3,
                    )
                ),
            ]
        )
        groundtruths.append(gts)

        pds = schemas.Prediction(
            model_name="test",
            datum=image,
            annotations=[
                schemas.PredictedAnnotation(
                    scored_labels=scored_labels1,
                    annotation=schemas.Annotation(
                        task_type=enums.TaskType.DETECTION,
                        bounding_box=boundary1,
                    )
                ),
                schemas.PredictedAnnotation(
                    scored_labels=scored_labels2,
                    annotation=schemas.Annotation(
                        task_type=enums.TaskType.DETECTION,
                        bounding_box=boundary2,
                    )
                ),
                schemas.PredictedAnnotation(
                    scored_labels=scored_labels3,
                    annotation=schemas.Annotation(
                        task_type=enums.TaskType.DETECTION,
                        bounding_box=boundary3,
                    )
                ),
            ]
        )
        predictions.append(pds)

    return predictions, groundtruths


def test_compute_mot_metrics():
    num_frames = 10
    predictions, groundtruths = generate_mot_data(num_frames)

    out = compute_mot_metrics(predictions, groundtruths)

    perfect_score = [
        num_frames,
        1,
        1,
        1,
        1,
        1,
        3 * num_frames,
        3,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
    ]
    perfect_score = {
        name: score for name, score in zip(MOT_METRICS_NAMES, perfect_score)
    }

    assert perfect_score == out
