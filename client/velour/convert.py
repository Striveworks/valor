from typing import Any, Dict, List

import numpy as np

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


def coco_rle_to_mask(coco_rle_seg_dict: Dict[str, Any]) -> np.ndarray:
    """Converts a COCO run-length-encoded segmentation to a binary mask

    Parameters
    ----------
    coco_rle_seg_dict
        a COCO formatted RLE segmentation dictionary. This should have keys
        "counts" and "size".

    Returns
    -------
    the corresponding binary mask
    """
    if not set(coco_rle_seg_dict.keys()) == {"counts", "size"}:
        raise ValueError(
            "`coco_rle_seg_dict` expected to be dict with keys 'counts' and 'size'."
        )

    starts, lengths = (
        coco_rle_seg_dict["counts"][::2],
        coco_rle_seg_dict["counts"][1::2],
    )
    run_length_encoding = list(zip(starts, lengths))

    h, w = coco_rle_seg_dict["size"]

    res = np.zeros((h, w), dtype=bool)
    idx = 0
    for start, length in run_length_encoding:
        idx += start
        for i in range(idx, idx + length):
            y, x = divmod(i, h)
            res[x, y] = True
        idx += length
    return res
