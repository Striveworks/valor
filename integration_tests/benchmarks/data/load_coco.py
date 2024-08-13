import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import PIL.Image
import requests
from shapely import geometry, ops
from skimage import measure
from tqdm import tqdm

from valor import Annotation, Datum, GroundTruth, Label
from valor.enums import TaskType
from valor.metatypes import ImageMetadata
from valor.schemas import Box, MultiPolygon, Polygon, Raster


def download_coco_panoptic(
    destination: Path = Path("./coco"),
    coco_url: str = "http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip",
) -> dict:
    """
    Download and return COCO panoptic dataset.
    """

    # append the location of the annotations within the destination folder
    annotations_zipfile = destination / Path(
        "annotations/panoptic_val2017.zip"
    )

    if not os.path.exists(str(destination)):
        # Make a GET request to the URL
        response = requests.get(coco_url, stream=True)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Get the total file size (if available)
            total_size = int(response.headers.get("content-length", 0))

            # Create a temporary file to save the downloaded content
            with tempfile.TemporaryFile() as temp_file:
                # Initialize tqdm with the total file size
                with tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc="Downloading",
                ) as pbar:
                    # Iterate over the response content and update progress
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            temp_file.write(chunk)
                            pbar.update(1024)

                # Once the file is downloaded, extract it
                with zipfile.ZipFile(temp_file, "r") as zip_ref:
                    total_files = len(zip_ref.infolist())
                    with tqdm(
                        total=total_files, unit="file", desc="Extracting"
                    ) as extraction_pbar:
                        for file_info in zip_ref.infolist():
                            zip_ref.extract(file_info, str(destination))
                            extraction_pbar.update(1)

        # unzip the validation set
        folder = str(annotations_zipfile.parent.absolute())
        filepath = str(annotations_zipfile.absolute())
        with zipfile.ZipFile(filepath, "r") as zip_ref:
            zip_ref.extractall(folder)
    else:
        print(f"coco already exists at {destination}!")

    with open(str(annotations_zipfile.with_suffix(".json"))) as f:
        panoptic_val2017 = json.load(f)

    return panoptic_val2017


def _parse_image_to_datum(image: dict) -> Datum:
    """
    Parse COCO image to Valor Datum
    """
    image = image.copy()
    uid = str(image.pop("id"))
    height = image.pop("height")
    width = image.pop("width")
    image_metadata = ImageMetadata.create(
        uid=uid,
        height=height,
        width=width,
        metadata=image,
    )
    return image_metadata.datum


def _parse_categories(
    categories: List[dict],
) -> Dict[int, Union[bool, Dict[str, str]]]:
    """
    Parse COCO categories into `valor.enums.TaskType` and `valor.Label`
    """
    return {
        category["id"]: {
            "labels": {
                "supercategory": category["supercategory"],
                "name": category["name"],
            },
            "is_instance": (True if category["isthing"] else False),
        }
        for category in categories
    }  # type: ignore - dict typing


def _create_masks(filename: str) -> np.ndarray:
    """
    Convert the colors in the mask to ids.
    """
    mask = np.array(PIL.Image.open(filename)).astype(int)
    return mask[:, :, 0] + 256 * mask[:, :, 1] + (256**2) * mask[:, :, 2]


def bitmask_to_bbox(bitmask) -> Box:
    bitmask = np.array(bitmask, dtype=bool)
    true_indices = np.argwhere(bitmask)
    if true_indices.size == 0:
        raise RuntimeError

    xmin, ymin = true_indices.min(axis=0)
    xmax, ymax = true_indices.max(axis=0)

    return Box.from_extrema(
        xmin=float(xmin),
        xmax=float(xmax),
        ymin=float(ymin),
        ymax=float(ymax),
    )


def bitmask_to_multipolygon_raster(bitmask) -> Raster:
    bitmask = np.array(bitmask, dtype=bool)
    labeled_array, num_features = measure.label(
        bitmask, background=0, return_num=True
    )
    polygons = []
    for region_index in range(1, num_features + 1):
        contours = measure.find_contours(labeled_array == region_index, 0.5)
        for contour in contours:
            if len(contour) >= 3:
                polygon = geometry.Polygon(contour)
                if polygon.is_valid:
                    polygons.append(polygon)
    mp = geometry.MultiPolygon(polygons).simplify(tolerance=0.6)
    values = []
    if isinstance(mp, geometry.MultiPolygon):
        for polygon in mp.geoms:
            boundary = list(polygon.exterior.coords)
            holes = [list(interior.coords) for interior in polygon.interiors]
            values.append([boundary, *holes])
    else:
        boundary = list(mp.exterior.coords)
        holes = [list(interior.coords) for interior in mp.interiors]
        values = [[boundary, *holes]]
    height, width = bitmask.shape
    return Raster.from_geometry(
        MultiPolygon(values), height=height, width=width
    )


def bitmask_to_polygon(bitmask) -> Polygon:
    bitmask = np.array(bitmask, dtype=bool)
    labeled_array, num_features = measure.label(
        bitmask, background=0, return_num=True
    )
    polygons = []
    for region_index in range(1, num_features + 1):
        contours = measure.find_contours(labeled_array == region_index, 0.5)
        for contour in contours:
            if len(contour) >= 3:
                polygon = geometry.Polygon(contour)
                if polygon.is_valid:
                    polygons.append(polygon)
    polygon = ops.unary_union(
        geometry.MultiPolygon(polygons).simplify(tolerance=0.6)
    )
    if not isinstance(polygon, geometry.Polygon):
        return None
    boundary = list(polygon.exterior.coords)
    holes = [list(interior.coords) for interior in polygon.interiors]
    return Polygon([boundary, *holes])


def create_bounding_boxes(
    image: dict,
    category_id_to_labels_and_task: Dict[int, Union[TaskType, Dict[str, str]]],
    mask_ids,
) -> List[Annotation]:
    return [
        Annotation(
            labels=[
                Label(
                    key="supercategory",
                    value=str(
                        category_id_to_labels_and_task[
                            segmentation["category_id"]
                        ]["labels"][
                            "supercategory"
                        ]  # type: ignore - dict typing
                    ),
                ),
                Label(
                    key="name",
                    value=str(
                        category_id_to_labels_and_task[
                            segmentation["category_id"]
                        ]["labels"][
                            "name"
                        ]  # type: ignore - dict typing
                    ),
                ),
                Label(key="iscrowd", value=str(segmentation["iscrowd"])),
            ],
            bounding_box=bitmask_to_bbox(mask_ids == segmentation["id"]),
            is_instance=True,
        )
        for segmentation in image["segments_info"]
        if category_id_to_labels_and_task[segmentation["category_id"]][
            "is_instance"
        ]  # type: ignore - dict typing
        is True
    ]


def create_bounding_polygons(
    image: dict,
    category_id_to_labels_and_task: Dict[int, Union[TaskType, Dict[str, str]]],
    mask_ids,
) -> List[Annotation]:
    return [
        Annotation(
            labels=[
                Label(
                    key="supercategory",
                    value=str(
                        category_id_to_labels_and_task[
                            segmentation["category_id"]
                        ]["labels"][
                            "supercategory"
                        ]  # type: ignore - dict typing
                    ),
                ),
                Label(
                    key="name",
                    value=str(
                        category_id_to_labels_and_task[
                            segmentation["category_id"]
                        ]["labels"][
                            "name"
                        ]  # type: ignore - dict typing
                    ),
                ),
                Label(key="iscrowd", value=str(segmentation["iscrowd"])),
            ],
            polygon=bitmask_to_polygon(mask_ids == segmentation["id"]),
            is_instance=True,
        )
        for segmentation in image["segments_info"]
        if category_id_to_labels_and_task[segmentation["category_id"]][
            "is_instance"
        ]  # type: ignore - dict typing
        is True
    ]


def create_raster_from_bitmask(
    image: dict,
    category_id_to_labels_and_task: Dict[int, Union[TaskType, Dict[str, str]]],
    mask_ids,
) -> List[Annotation]:
    return [
        Annotation(
            labels=[
                Label(
                    key="supercategory",
                    value=str(
                        category_id_to_labels_and_task[
                            segmentation["category_id"]
                        ]["labels"][
                            "supercategory"
                        ]  # type: ignore - dict typing
                    ),
                ),
                Label(
                    key="name",
                    value=str(
                        category_id_to_labels_and_task[
                            segmentation["category_id"]
                        ]["labels"][
                            "name"
                        ]  # type: ignore - dict typing
                    ),
                ),
                Label(key="iscrowd", value=str(segmentation["iscrowd"])),
            ],
            raster=Raster.from_numpy(mask_ids == segmentation["id"]),
            is_instance=True,
        )
        for segmentation in image["segments_info"]
        if category_id_to_labels_and_task[segmentation["category_id"]][
            "is_instance"
        ]  # type: ignore - dict typing
        is True
    ]


def create_raster_from_multipolygon(
    image: dict,
    category_id_to_labels_and_task: Dict[int, Union[TaskType, Dict[str, str]]],
    mask_ids,
) -> List[Annotation]:
    return [
        Annotation(
            labels=[
                Label(
                    key="supercategory",
                    value=str(
                        category_id_to_labels_and_task[
                            segmentation["category_id"]
                        ]["labels"][
                            "supercategory"
                        ]  # type: ignore - dict typing
                    ),
                ),
                Label(
                    key="name",
                    value=str(
                        category_id_to_labels_and_task[
                            segmentation["category_id"]
                        ]["labels"][
                            "name"
                        ]  # type: ignore - dict typing
                    ),
                ),
                Label(key="iscrowd", value=str(segmentation["iscrowd"])),
            ],
            raster=bitmask_to_multipolygon_raster(
                mask_ids == segmentation["id"]
            ),
            is_instance=True,
        )
        for segmentation in image["segments_info"]
        if category_id_to_labels_and_task[segmentation["category_id"]][
            "is_instance"
        ]  # type: ignore - dict typing
        is True
    ]


def create_semantic_segmentations(
    image: dict,
    category_id_to_labels_and_task: Dict[int, Union[TaskType, Dict[str, str]]],
    mask_ids,
):
    # combine semantic segmentations
    semantic_masks = {
        "supercategory": {},
        "name": {},
        "iscrowd": {},
    }
    for segmentation in image["segments_info"]:
        category_id = segmentation["category_id"]
        if (
            category_id_to_labels_and_task[category_id]["is_instance"]  # type: ignore - dict typing
            is False
        ):
            for key, value in [
                (
                    "supercategory",
                    category_id_to_labels_and_task[category_id]["labels"][  # type: ignore - dict typing
                        "supercategory"
                    ],
                ),
                (
                    "name",
                    category_id_to_labels_and_task[category_id]["labels"][  # type: ignore - dict typing
                        "name"
                    ],
                ),
                ("iscrowd", segmentation["iscrowd"]),
            ]:
                if value not in semantic_masks[key]:
                    semantic_masks[key][value] = mask_ids == segmentation["id"]
                else:
                    semantic_masks[key][value] = np.logical_or(
                        semantic_masks[key][value],
                        (mask_ids == segmentation["id"]),
                    )

    # create annotations for semantic segmentation
    return [
        Annotation(
            labels=[Label(key=key, value=str(value))],
            raster=Raster.from_numpy(semantic_masks[key][value]),
            is_instance=False,
        )
        for key in semantic_masks
        for value in semantic_masks[key]
    ]


def _create_groundtruths_from_coco_panoptic(
    data: dict,
    masks_path: Path,
    objdet_bbox_file,
    objdet_polygon_file,
    objdet_multipolygon_file,
    objdet_raster_file,
    semseg_raster_file,
):

    # extract labels from categories
    category_id_to_labels_and_task = _parse_categories(data["categories"])
    # create datums
    image_id_to_datum = {
        image["id"]: _parse_image_to_datum(image) for image in data["images"]
    }

    # create groundtruths
    for image in tqdm(data["annotations"], "Saving to JSON."):
        # exract masks from annotations
        mask_ids = _create_masks(masks_path / image["file_name"])

        # create bounding boxes
        bbox_annotations = create_bounding_boxes(
            image,
            category_id_to_labels_and_task,
            mask_ids,
        )
        gt = GroundTruth(
            datum=image_id_to_datum[image["image_id"]],
            annotations=bbox_annotations,
        )
        objdet_bbox_file.write(json.dumps(gt.encode_value()).encode("utf-8"))
        objdet_bbox_file.write("\n".encode("utf-8"))

        # create bounding polygons
        bbox_annotations = create_bounding_polygons(
            image,
            category_id_to_labels_and_task,
            mask_ids,
        )
        gt = GroundTruth(
            datum=image_id_to_datum[image["image_id"]],
            annotations=bbox_annotations,
        )
        objdet_polygon_file.write(
            json.dumps(gt.encode_value()).encode("utf-8")
        )
        objdet_polygon_file.write("\n".encode("utf-8"))

        # create instance segmentations using multipolygon rasters
        instance_annotations = create_raster_from_multipolygon(
            image,
            category_id_to_labels_and_task,
            mask_ids,
        )
        gt = GroundTruth(
            datum=image_id_to_datum[image["image_id"]],
            annotations=instance_annotations,
        )
        objdet_multipolygon_file.write(
            json.dumps(gt.encode_value()).encode("utf-8")
        )
        objdet_multipolygon_file.write("\n".encode("utf-8"))

        # create instance segmentations using bitmask rasters
        instance_annotations = create_raster_from_bitmask(
            image,
            category_id_to_labels_and_task,
            mask_ids,
        )
        gt = GroundTruth(
            datum=image_id_to_datum[image["image_id"]],
            annotations=instance_annotations,
        )
        objdet_raster_file.write(json.dumps(gt.encode_value()).encode("utf-8"))
        objdet_raster_file.write("\n".encode("utf-8"))

        # create semantic segmentations
        semantic_annotations = create_semantic_segmentations(
            image,
            category_id_to_labels_and_task,
            mask_ids,
        )
        gt = GroundTruth(
            datum=image_id_to_datum[image["image_id"]],
            annotations=semantic_annotations,
        )
        semseg_raster_file.write(json.dumps(gt.encode_value()).encode("utf-8"))
        semseg_raster_file.write("\n".encode("utf-8"))


def create_gts_from_coco_panoptic(
    path: str = "./",
    destination: str = "coco",
    coco_url: str = "http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip",
    limit: int = 0,
):
    """
    Creates Dataset and associated GroundTruths.

    Parameters
    ----------
    name : str
        Desired dataset name.
    path : str
        The working directory.
    destination : str
        Desired output path for dataset annotations.
    coco_url : str
        URL to the COCO dataset.
    limit : int, default=0
        Limits the number of datums. Default to 0 for no action.
    """
    coco_path = Path(path) / Path(destination)

    # download and unzip coco dataset
    data = download_coco_panoptic(
        destination=coco_path,
        coco_url=coco_url,
    )

    # path of mask locations
    masks_path = coco_path / Path("annotations/panoptic_val2017/")

    # slice if limited
    if limit > 0:
        data["annotations"] = data["annotations"][:limit]

    objdet_bbox_filepath = Path(path) / Path("gt_objdet_coco_bbox.jsonl")
    objdet_polygon_filepath = Path(path) / Path("gt_objdet_coco_polygon.jsonl")
    objdet_multipolygon_filepath = Path(path) / Path(
        "gt_objdet_coco_raster_multipolygon.jsonl"
    )
    objdet_raster_filepath = Path(path) / Path(
        "gt_objdet_coco_raster_bitmask.jsonl"
    )
    semseg_raster_filepath = Path(path) / Path("gt_semseg_coco.jsonl")

    # create groundtruths
    with open(objdet_bbox_filepath, mode="wb") as f_objdet_box:
        with open(objdet_polygon_filepath, mode="wb") as f_objdet_polygon:
            with open(
                objdet_multipolygon_filepath, mode="wb"
            ) as f_objdet_multipolygon:
                with open(
                    objdet_raster_filepath, mode="wb"
                ) as f_objdet_raster:
                    with open(
                        semseg_raster_filepath, mode="wb"
                    ) as f_semseg_raster:
                        _create_groundtruths_from_coco_panoptic(
                            data=data,
                            masks_path=masks_path,
                            objdet_bbox_file=f_objdet_box,
                            objdet_polygon_file=f_objdet_polygon,
                            objdet_multipolygon_file=f_objdet_multipolygon,
                            objdet_raster_file=f_objdet_raster,
                            semseg_raster_file=f_semseg_raster,
                        )


if __name__ == "__main__":

    filepath = os.path.dirname(os.path.realpath(__file__))
    create_gts_from_coco_panoptic(
        path=filepath,
        # limit=50
    )
