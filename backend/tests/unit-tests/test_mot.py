from velour_api.mot_metrics import (
    compute_mot_metrics,
    OBJECT_ID_LABEL_KEY,
    MOT_METRICS_NAMES,
)
from velour_api import schemas


def generate_mot_data(num_frames: int):
    """
    Create synthetic predictions and detections which are identical (perfect detector).
    Three objects in each frame, moving in different directions at constant speed.
    """

    create_img = lambda frame: schemas.Image(
        uri="", height=500, width=500, frame=frame
    )
    create_label = lambda obj_id: schemas.Label(
        key=OBJECT_ID_LABEL_KEY, value=obj_id
    )
    create_scored_label = lambda obj_id, score: schemas.ScoredLabel(
        key=OBJECT_ID_LABEL_KEY, value=obj_id, score=score
    )

    square = lambda x, y: [(x, y), (x + 10, y), (x, y + 10), (x + 10, y + 10)]
    # Square Boundary moving diagonally
    create_boundary_1 = lambda frame: square(5 * frame, 5 * frame)
    # Square Boundary moving horizontally
    create_boundary_2 = lambda frame: square(5 * frame, 0)
    # Square Boundary moving vertically
    create_boundary_3 = lambda frame: square(0, 5 * frame)

    predictions = []
    groundtruths = []
    for frame in range(1, num_frames + 1):
        boundary1 = create_boundary_1(frame)
        boundary2 = create_boundary_2(frame)
        boundary3 = create_boundary_3(frame)

        image = create_img(frame)

        scored_labels1 = [create_scored_label("object 1", 0.99)]
        scored_labels2 = [create_scored_label("object 2", 0.99)]
        scored_labels3 = [create_scored_label("object 3", 0.99)]

        labels1 = [create_label("object 1")]
        labels2 = [create_label("object 2")]
        labels3 = [create_label("object 3")]

        pred1 = schemas.PredictedDetection(
            boundary=boundary1, image=image, scored_labels=scored_labels1
        )
        pred2 = schemas.PredictedDetection(
            boundary=boundary2, image=image, scored_labels=scored_labels2
        )
        pred3 = schemas.PredictedDetection(
            boundary=boundary3, image=image, scored_labels=scored_labels3
        )

        gt1 = schemas.GroundTruthDetection(
            boundary=boundary1, image=image, labels=labels1
        )
        gt2 = schemas.GroundTruthDetection(
            boundary=boundary2, image=image, labels=labels2
        )
        gt3 = schemas.GroundTruthDetection(
            boundary=boundary3, image=image, labels=labels3
        )

        predictions += [pred1, pred2, pred3]
        groundtruths += [gt1, gt2, gt3]

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
