import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path, PosixPath
from typing import Any, Dict, List, Union

import numpy as np
import PIL.Image
from tqdm import tqdm

from velour import Annotation, GroundTruth, enums
from velour.client import Dataset
from velour.metatypes import ImageMetadata
from velour.schemas import Label, Raster


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


def _get_task_type(isthing: bool) -> enums.TaskType:
    """Get the correct TaskType for a given label"""
    return enums.TaskType.DETECTION if isthing else enums.TaskType.SEGMENTATION


def _is_semantic_task_type(task_type: enums.TaskType) -> bool:
    """Check if a label is a semantic segmentation"""
    return task_type == enums.TaskType.SEGMENTATION


def _merge_annotations(annotation_list: list, label_map: dict):
    """Aggregate masks of annotations that share a common label"""

    # deepcopy since we use .remove()
    annotation_list = deepcopy(annotation_list)

    for label, indices in label_map.items():
        if len(indices) > 1:
            joined_mask = annotation_list[indices[0]]["mask"]
            task_type = annotation_list[indices[0]]["task_type"]

            # remove the label from the parent node
            annotation_list[indices[0]]["labels"].remove(label)

            for child_index in indices[1:]:
                if indices[0] != child_index:
                    child = annotation_list[child_index]
                    joined_mask = np.logical_or(joined_mask, child["mask"])

                    # remove the label from the child node
                    annotation_list[child_index]["labels"].remove(label)

            annotation_list.append(
                dict(
                    task_type=task_type, labels=set([label]), mask=joined_mask
                )
            )

    # delete any annotations without labels remaining (i.e., their mask is now incorporated into grouped annotations)
    annotation_list = [
        annotation
        for annotation in annotation_list
        if len(annotation["labels"]) > 0
    ]

    return annotation_list


def _get_segs_groundtruth_for_single_image(
    ann_dict: dict,
    masks_path: str,
    image_id_to_height: dict,
    image_id_to_width: dict,
    category_id_to_category: dict,
) -> List[GroundTruth]:
    mask = np.array(PIL.Image.open(masks_path / ann_dict["file_name"])).astype(
        int
    )
    # convert the colors in the mask to ids
    mask_ids = mask[:, :, 0] + 256 * mask[:, :, 1] + (256**2) * mask[:, :, 2]

    # create datum
    image_id = ann_dict["image_id"]

    height = image_id_to_height[image_id]
    width = image_id_to_width[image_id]

    assert isinstance(height, int)
    assert isinstance(width, int)

    img = ImageMetadata(
        uid=str(image_id),
        height=height,
        width=width,
    ).to_datum()

    # create initial list of annotations
    annotation_list = []

    semantic_labels = defaultdict(list)

    for index, segment in enumerate(ann_dict["segments_info"]):
        mask = mask_ids == segment["id"]
        task_type = _get_task_type(
            category_id_to_category[segment["category_id"]]["isthing"]
        )
        is_semantic = _is_semantic_task_type(task_type=task_type)

        labels = set()

        for k in ["supercategory", "name"]:
            category_desc = str(
                category_id_to_category[segment["category_id"]][k]
            )

            label = Label(
                key=k,
                value=category_desc,
            )

            # identify the location of all semantic segmentation labels
            if is_semantic:
                semantic_labels[label].append(index)

            labels.add(label)

        annotation_list.append(
            dict(task_type=task_type, labels=labels, mask=mask)
        )

    # add iscrowed label
    iscrowd_label = Label(key="iscrowd", value=str(segment["iscrowd"]))
    if is_semantic:
        semantic_labels[iscrowd_label].append(
            len(ann_dict["segments_info"]) - 1
        )
    else:
        for annotation in annotation_list:
            annotation["labels"].add(iscrowd_label)

    # combine semantic segmentation masks by label
    final_annotation_list = _merge_annotations(
        annotation_list=annotation_list, label_map=semantic_labels
    )

    # create groundtruth
    return GroundTruth(
        datum=img,
        annotations=[
            Annotation(
                task_type=annotation["task_type"],
                labels=list(annotation["labels"]),
                raster=Raster.from_numpy(annotation["mask"]),
            )
            for annotation in final_annotation_list
        ],
    )


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

    for ann in tqdm(annotations["annotations"]):
        gt = _get_segs_groundtruth_for_single_image(
            ann_dict=ann,
            masks_path=masks_path,
            image_id_to_height=image_id_to_height,
            image_id_to_width=image_id_to_width,
            category_id_to_category=category_id_to_category,
        )
        dataset.add_groundtruth(gt)
