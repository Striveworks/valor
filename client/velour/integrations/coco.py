import json
from pathlib import Path, PosixPath
from typing import Any, Dict, List, Union

import numpy as np
import PIL.Image
from tqdm.auto import tqdm

from velour import enums
from velour.client import Dataset
from velour.schemas import Annotation, GroundTruth, Image, Label, Raster


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
    annotations: Union[str, PosixPath, dict],
    masks_path: str,
) -> None:
    masks_path = Path(masks_path)
    if isinstance(annotations, (str, PosixPath)):
        with open(annotations) as f:
            annotations = json.load(f)

    category_id_to_category = {
        cat["id"]: cat for cat in annotations["categories"]
    }

    image_id_to_height, image_id_to_width, image_id_to_coco_url = {}, {}, {}
    for image in annotations["images"]:
        image_id_to_height[image["id"]] = image["height"]
        image_id_to_width[image["id"]] = image["width"]
        image_id_to_coco_url[image["id"]] = image["coco_url"]

    def _get_segs_groundtruth_for_single_image(
        ann_dict: dict,
    ) -> List[GroundTruth]:
        mask = np.array(
            PIL.Image.open(masks_path / ann_dict["file_name"])
        ).astype(int)
        # convert the colors in the mask to ids
        mask_ids = (
            mask[:, :, 0] + 256 * mask[:, :, 1] + (256**2) * mask[:, :, 2]
        )

        # create datum
        image_id = ann_dict["image_id"]
        img = Image(
            dataset=dataset.name,
            uid=str(image_id),
            height=image_id_to_height[image_id],
            width=image_id_to_width[image_id],
        ).to_datum()

        # create groundtruth
        return GroundTruth(
            datum=img,
            annotations=[
                Annotation(
                    task_type=(
                        enums.TaskType.INSTANCE_SEGMENTATION
                        if category_id_to_category[segment["category_id"]][
                            "isthing"
                        ]
                        else enums.TaskType.SEMANTIC_SEGMENTATION
                    ),
                    labels=[
                        Label(
                            key=k,
                            value=str(
                                category_id_to_category[
                                    segment["category_id"]
                                ][k]
                            ),
                        )
                        for k in ["supercategory", "name"]
                    ]
                    + [Label(key="iscrowd", value=str(segment["iscrowd"]))],
                    raster=Raster.from_numpy(mask_ids == segment["id"]),
                )
                for segment in ann_dict["segments_info"]
            ],
        )

    for ann in tqdm(annotations["annotations"]):
        gt = _get_segs_groundtruth_for_single_image(ann)
        dataset.add_groundtruth(gt)
