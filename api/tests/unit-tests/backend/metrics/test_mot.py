from velour_api import enums, schemas
from velour_api.backend.metrics.multi_object_tracking import (
    MOT_METRICS_NAMES,
    OBJECT_ID_LABEL_KEY,
    compute_mot_metrics,
)


# noqa: E731
def square(x: int, y: int) -> schemas.BasicPolygon:
    return schemas.BoundingBox(
        polygon=schemas.geometry.BasicPolygon(
            points=[
                schemas.geometry.Point(x=x, y=y),
                schemas.geometry.Point(x=x + 10, y=y),
                schemas.geometry.Point(x=x, y=y + 10),
                schemas.geometry.Point(x=x + 10, y=y + 10),
            ]
        )
    )


def generate_mot_data(num_frames: int):
    """
    Create synthetic predictions and detections which are identical (perfect detector).
    Three objects in each frame, moving in different directions at constant speed.
    """

    create_img = lambda frame: schemas.Datum(  # noqa: E731
        uid="test",
        dataset_name="name",
        metadata={
            "type": "image",
            "height": 500,
            "width": 500,
            "frame": frame,
        },
    )
    create_label_lambda = lambda obj_id: schemas.Label(  # noqa: E731
        key=OBJECT_ID_LABEL_KEY, value=obj_id
    )
    create_scored_label = lambda obj_id, score: schemas.Label(  # noqa: E731
        key=OBJECT_ID_LABEL_KEY, value=obj_id, score=score
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

        labels1 = [create_label_lambda("object 1")]
        labels2 = [create_label_lambda("object 2")]
        labels3 = [create_label_lambda("object 3")]

        gts = schemas.GroundTruth(
            datum=image,
            annotations=[
                schemas.Annotation(
                    labels=labels1,
                    task_type=enums.TaskType.OBJECT_DETECTION,
                    bounding_box=boundary1,
                ),
                schemas.Annotation(
                    labels=labels2,
                    task_type=enums.TaskType.OBJECT_DETECTION,
                    bounding_box=boundary2,
                ),
                schemas.Annotation(
                    labels=labels3,
                    task_type=enums.TaskType.OBJECT_DETECTION,
                    bounding_box=boundary3,
                ),
            ],
        )
        groundtruths.append(gts)

        pds = schemas.Prediction(
            model_name="test",
            datum=image,
            annotations=[
                schemas.Annotation(
                    labels=scored_labels1,
                    task_type=enums.TaskType.OBJECT_DETECTION,
                    bounding_box=boundary1,
                ),
                schemas.Annotation(
                    labels=scored_labels2,
                    task_type=enums.TaskType.OBJECT_DETECTION,
                    bounding_box=boundary2,
                ),
                schemas.Annotation(
                    labels=scored_labels3,
                    task_type=enums.TaskType.OBJECT_DETECTION,
                    bounding_box=boundary3,
                ),
            ],
        )
        predictions.append(pds)

    return predictions, groundtruths


def test_compute_mot_metrics():
    num_frames = 10
    predictions, groundtruths = generate_mot_data(num_frames)

    out = compute_mot_metrics(predictions, groundtruths).to_dict(
        orient="records"
    )[0]

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
