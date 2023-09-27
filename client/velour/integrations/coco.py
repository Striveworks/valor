import json
from pathlib import Path, PosixPath
from typing import Any, Dict, List, Union
from collections import defaultdict

import numpy as np
import PIL.Image
from tqdm import tqdm

from velour import enums
from velour.client import Dataset
from velour.schemas import (
    Annotation,
    GroundTruth,
    ImageMetadata,
    Label,
    Raster,
)


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
    return (
        enums.TaskType.INSTANCE_SEGMENTATION
        if isthing
        else enums.TaskType.SEMANTIC_SEGMENTATION
    )


def _is_segmentation_task_type(task_type: enums.TaskType) -> bool:
    """Check if a label is a semantic segmentation"""
    return True if task_type == enums.TaskType.SEMANTIC_SEGMENTATION else False


class DisjointSet:
    """Implement a Disjoint Union Set to match annotations that have at least one similar label"""

    def __init__(self, parents, weights):
        self.parents = parents
        self.weights = weights

    def find(self, item):
        if self.parents[item] == item:
            return item
        else:
            res = self.find(self.parents[item])
            self.parents[item] = res
            return res

    def union(self, set1, set2):
        root1 = self.find(set1)
        root2 = self.find(set2)

        if self.weights[root1] > self.weights[root2]:
            self.weights[root1] += self.weights[root2]
            self.parents[root2] = root1
        else:
            self.weights[root2] += self.weights[root1]
            self.parents[root1] = root2


def _merge_annotation_list(annotation_list: list, label_map: dict):
    """Use a disjoint union set to merge the masks and labels of annotations that share similar labels"""
    disjoint_set = DisjointSet(
        parents={i: i for i, v in enumerate(annotation_list)},
        weights={i: 1 for i, v in enumerate(annotation_list)},
    )

    for indices in label_map.values():
        for index in indices[1:]:
            disjoint_set.union(indices[0], index)

    # iterate through the parents, merging the labels and masks of any related annotations
    parent_to_child_mappings = defaultdict(set)
    for key, value in disjoint_set.parents.items():
        parent_to_child_mappings[value].add(key)

    final_annotation_list = []
    for parent_index, child_indices in parent_to_child_mappings.items():
        parent = annotation_list[parent_index]

        for child_index in child_indices:
            if child_index != parent_index:
                child = annotation_list[child_index]
                parent["mask"] = np.logical_or(parent["mask"], child["mask"])
                parent["labels"] = parent["labels"].union(child["labels"])
        final_annotation_list.append(parent)

    return final_annotation_list


def _get_segs_groundtruth_for_single_image(
    ann_dict: dict,
    masks_path: str,
    image_id_to_height: dict,
    image_id_to_width: dict,
    category_id_to_category: dict,
) -> List[GroundTruth]:
    mask = np.array(PIL.Image.open(masks_path / ann_dict["file_name"])).astype(int)
    # convert the colors in the mask to ids
    mask_ids = mask[:, :, 0] + 256 * mask[:, :, 1] + (256**2) * mask[:, :, 2]

    # create datum
    image_id = ann_dict["image_id"]
    img = ImageMetadata(
        uid=str(image_id),
        height=image_id_to_height[image_id],
        width=image_id_to_width[image_id],
    ).to_datum()

    # create initial list of annotations
    annotation_list = []
    segmentation_labels = defaultdict(list)

    for index, segment in enumerate(ann_dict["segments_info"]):
        mask = mask_ids == segment["id"]
        task_type = _get_task_type(
            category_id_to_category[segment["category_id"]]["isthing"]
        )

        labels = set()
        for k in ["supercategory", "name"]:
            category_desc = str(category_id_to_category[segment["category_id"]][k])

            label = Label(
                key=k,
                value=category_desc,
            )

            # if multiple annotations have the same label, then we need to combine them
            if _is_segmentation_task_type(task_type=task_type):
                segmentation_labels[label].append(index)

            labels.add(label)

        annotation_list.append(dict(task_type=task_type, labels=labels, mask=mask))

    final_annotation_list = _merge_annotation_list(
        annotation_list=annotation_list, label_map=segmentation_labels
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

    category_id_to_category = {cat["id"]: cat for cat in annotations["categories"]}

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
