import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import PIL.Image
from tqdm import tqdm

from velour.client import Dataset
from velour.data_types import (
    BoundingPolygon,
    GroundTruthInstanceSegmentation,
    GroundTruthSemanticSegmentation,
    Image,
    Label,
    PredictedDetection,
    ScoredLabel,
    _GroundTruthSegmentation,
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


def upload_coco_panoptic(
    dataset: Dataset,
    annotation_file: str,
    masks_path: str,
) -> None:
    masks_path = Path(masks_path)
    with open(annotation_file) as f:
        panoptic_anns = json.load(f)

    category_id_to_category = {
        cat["id"]: cat for cat in panoptic_anns["categories"]
    }

    image_id_to_height, image_id_to_width, image_id_to_coco_url = {}, {}, {}
    for image in panoptic_anns["images"]:
        image_id_to_height[image["id"]] = image["height"]
        image_id_to_width[image["id"]] = image["width"]
        image_id_to_coco_url[image["id"]] = image["coco_url"]

    def _get_segs_for_single_image(
        ann_dict: dict,
    ) -> List[_GroundTruthSegmentation]:
        mask = np.array(
            PIL.Image.open(masks_path / ann_dict["file_name"])
        ).astype(int)
        # convert the colors in the mask to ids
        mask_ids = np.apply_along_axis(
            lambda a: a[0] + 256 * a[1] + 256**2 * a[2], 2, mask
        )

        image_id = ann_dict["image_id"]
        img = Image(
            uid=image_id,
            height=image_id_to_height[image_id],
            width=image_id_to_width[image_id],
        )

        segs = []
        for segment in ann_dict["segments_info"]:
            binary_mask = mask_ids == segment["id"]

            category = category_id_to_category[segment["category_id"]]
            labels = [
                Label(key=k, value=category[k])
                for k in ["supercategory", "name"]
            ] + [Label(key="iscrowd", value=segment["iscrowd"])]

            if category["isthing"]:
                seg = GroundTruthInstanceSegmentation(
                    shape=binary_mask, image=img, labels=labels
                )
            else:
                seg = GroundTruthSemanticSegmentation(
                    shape=binary_mask, image=img, labels=labels
                )
            segs.append(seg)

        return segs

    for ann in tqdm(panoptic_anns["annotations"]):
        segs = _get_segs_for_single_image(ann)
        dataset.add_groundtruth_segmentations(segs)
