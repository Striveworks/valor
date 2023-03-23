from typing import List

from velour.data_types import (
    BoundingPolygon,
    Image,
    Label,
    PredictedDetection,
    ScoredLabel,
)


def chariot_detections_to_velour(
    dets: dict, image: Image, label_key: str = "class"
) -> List[PredictedDetection]:
    """Converts the outputs of a Chariot detection model
    to velour's format
    """
    expected_keys = {
        "num_detections",
        "detection_classes",
        "detection_boxes",
        "detection_scores",
    }
    if set(dets.keys()) != expected_keys:
        raise ValueError(
            f"Expected `dets` to have keys {expected_keys} but got {dets.keys()}"
        )

    return [
        PredictedDetection(
            boundary=BoundingPolygon.from_ymin_xmin_ymax_xmax(*box),
            scored_labels=[
                ScoredLabel(
                    label=Label(key=label_key, value=label), score=score
                )
            ],
            image=image,
        )
        for box, score, label in zip(
            dets["detection_boxes"],
            dets["detection_scores"],
            dets["detection_classes"],
        )
    ]
