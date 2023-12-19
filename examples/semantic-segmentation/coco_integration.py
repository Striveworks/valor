import os
import zipfile
import tempfile
import json
import PIL.Image
import requests
import numpy as np
from tqdm import tqdm
from io import BytesIO
from copy import deepcopy
from pathlib import Path, PosixPath
from typing import Any, Dict, List, Tuple, Union
from collections import defaultdict

from velour import (
    Client,
    Dataset,
    Datum,
    Annotation,
    GroundTruth,
    Label,
)
from velour.schemas import Raster
from velour.enums import TaskType, JobStatus
from velour.metatypes import ImageMetadata


def _download_and_unzip(url, output_folder):
    # Make a GET request to the URL
    response = requests(url, stream=True)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Get the total file size (if available)
        total_size = int(response.headers('content-length', 0))

        # Create a temporary file to save the downloaded content
        with tempfile.TemporaryFile() as temp_file:
            # Initialize tqdm with the total file size
            with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024, desc="Downloading") as pbar:
                # Iterate over the response content and update progress
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        temp_file.write(chunk)
                        pbar.update(1024)

            # Once the file is downloaded, extract it
            with zipfile.ZipFile(temp_file, 'r') as zip_ref:
                total_files = len(zip_ref.infolist())
                with tqdm(total=total_files, unit='file', desc="Extracting") as extraction_pbar:
                    for file_info in zip_ref.infolist():
                        zip_ref.extract(file_info, output_folder)
                        extraction_pbar.update(1)


def _unzip(filepath: Path):
    folder = str(filepath.parent.absolute())
    filepath = str(filepath.absolute())
    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        zip_ref.extractall(folder)


def _load(
    destination: str,
    coco_url: str = "http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip",
    annotations_zipfile: Path = Path("./coco/annotations/panoptic_val2017.zip"),
):
    """
    Download and unzip COCO panoptic dataset.
    """
    if not os.path.exists(destination):
        _download_and_unzip(coco_url, destination)
        _unzip(annotations_zipfile)
    else:
        print(f"coco already exists at {destination}!")

    with open(str(annotations_zipfile.with_suffix(".json"))) as f:
        panoptic_val2017 = json.load(f)

    return panoptic_val2017


def _parse_image_to_datum(image: dict) -> Datum:
    """
    Parse COCO image to Velour Datum
    """
    image = image.copy()
    uid = str(image.pop("id"))
    height = image.pop("height")
    width = image.pop("width")
    image_metadata = ImageMetadata(
        uid=uid,
        height=height,
        width=width,
        metadata=image,
    )
    return image_metadata.to_datum()


def _parse_categories(categories: list) -> Dict[int, Union[TaskType, Dict[str, str]]]:
    """
    Parse COCO categories into `velour.enums.TaskType` and `velour.Label`
    """
    return {
        category["id"] : {
            "task_type": (
                TaskType.DETECTION
                if category["isthing"]
                else TaskType.SEGMENTATION
            ),
            "labels": {
                "supercategory": category["supercategory"],
                "name": category["name"],
            }
        }
        for category in categories
    }


def _create_masks(filename: str) -> np.ndarray:
    """
    Convert the colors in the mask to ids.
    """
    mask = np.array(
        PIL.Image.open(filename)
    ).astype(int)
    return mask[:, :, 0] + 256 * mask[:, :, 1] + (256**2) * mask[:, :, 2]


def _create_annotations_from_instance_segmentations(
    image: dict,
    category_id_to_labels_and_task: Dict[int, Union[TaskType, Dict[str, str]]],
    mask_ids,
) -> List[Annotation]:
    return [
        Annotation(
            task_type=TaskType.DETECTION,
            labels=[
                Label(key="supercategory", value=str(category_id_to_labels_and_task[segmentation["category_id"]]["labels"]["supercategory"])),
                Label(key="name", value=str(category_id_to_labels_and_task[segmentation["category_id"]]["labels"]["name"])),
                Label(key="iscrowd", value=str(segmentation["iscrowd"])),
            ],
            raster=Raster.from_numpy(
                mask_ids == segmentation["id"]
            ),
        )
        for segmentation in image["segments_info"]
        if category_id_to_labels_and_task[segmentation["category_id"]]["task_type"] == TaskType.DETECTION
    ]


def _create_annotations_from_semantic_segmentations(
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
        if category_id_to_labels_and_task[category_id]["task_type"] == TaskType.SEGMENTATION:
            for key, value in [
                ("supercategory", category_id_to_labels_and_task[category_id]["labels"]["supercategory"]),
                ("name", category_id_to_labels_and_task[category_id]["labels"]["name"]),
                ("iscrowd", segmentation["iscrowd"]),
            ]:
                if value not in semantic_masks[key]:
                    semantic_masks[key][value] = (mask_ids == segmentation["id"])
                else:
                    semantic_masks[key][value] = np.logical_or(semantic_masks[key][value], (mask_ids == segmentation["id"]))                

    # create annotations for semantic segmentation
    return [
        Annotation(
            task_type=TaskType.SEGMENTATION,
            labels=[Label(key=key, value=str(value))],
            raster=Raster.from_numpy(
                mask_ids == segmentation["id"]
            ),
        )
        for key in semantic_masks
        for value in semantic_masks[key]
    ]


def _create_groundtruths_from_coco_panoptic(
    data: dict,
    masks_path: Path,
) -> List[GroundTruth]:

    # extract task_type and labels from categories
    category_id_to_labels_and_task = _parse_categories(data["categories"])

    # create datums
    image_id_to_datum = {
        image["id"] : _parse_image_to_datum(image)
        for image in data["images"]
    }

    # create groundtruths
    groundtruths = []
    for image in tqdm(data["annotations"]):

        # exract masks from annotations
        mask_ids = _create_masks(masks_path / image["file_name"])
        
        # create instance segmentations
        objdet_annotations = _create_annotations_from_instance_segmentations(
            image,
            category_id_to_labels_and_task,
            mask_ids,
        )

        # create semantic segmentations
        semseg_annotations = _create_annotations_from_semantic_segmentations(
            image,
            category_id_to_labels_and_task,
            mask_ids,
        )

        # create groundTruth
        groundtruths.append(
            GroundTruth(
                datum=image_id_to_datum[image["image_id"]],
                annotations=(
                    objdet_annotations + semseg_annotations
                ),
            )
        )

    return groundtruths


def create_dataset_from_coco_panoptic(
    client: Client,
    name: str = "coco2017-panoptic",
    destination: str = "./coco",
    coco_url: str = "http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip",
    annotations_zipfile: Path = Path("./coco/annotations/panoptic_val2017.zip"),
    masks_path: Path = Path("./coco/annotations/panoptic_val2017/"),
    limit: int = 0,
    reset: bool = False,
) -> Dataset:
    """
    Creates Dataset and associated GroundTruths.

    Parameters
    ----------
    client : Client
        Velour client object.
    name : str
        Desired dataset name.
    destination : str
        Desired output path for dataset annotations.
    coco_url : str
        URL to the COCO dataset.
    annotations_zipfile : Path
        Local path to annotations zipfile.
    masks_path : Path
        Local path to unzipped annotations.
    limit : int, default=0
        Limits the number of datums. Default to 0 for no action.
    reset : bool, default=False
        Reset the Velour dataset before attempting creation.

    """

    # download and unzip coco dataset
    data = _load(
        destination=destination,
        coco_url=coco_url,
        annotations_zipfile=annotations_zipfile,
    )

    # slice if limited
    if limit > 0:
        data["annotations"] = data["annotations"][:limit]

    # if reset, delete the dataset if it exists
    if reset and client.get_dataset_status(name) != JobStatus.NONE:
        client.delete_dataset(name, timeout=5)

    if client.get_dataset_status(name) != JobStatus.NONE:
        dataset = Dataset(client, name)
    else:
        # create groundtruths
        gts = _create_groundtruths_from_coco_panoptic(
            data=data,
            masks_path=masks_path,
        )

        # extract metadata
        metadata = data["info"].copy()
        metadata["licenses"] = str(data["licenses"])

        # create dataset
        dataset = Dataset(
            client,
            name,
            metadata=metadata,
        )
        for gt in gts:
            dataset.add_groundtruth(gt)
        dataset.finalize()
        
    return dataset


def download_image(datum: Datum) -> PIL.Image:
    """
    Download image using Datum.
    """
    url = datum.metadata["coco_url"]
    img_data = BytesIO(requests.get(url).content)
    return PIL.Image.open(img_data)