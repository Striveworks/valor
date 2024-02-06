import json
import os
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import PIL.Image
import requests
from tqdm import tqdm

from velour import Annotation, Client, Dataset, Datum, GroundTruth, Label
from velour.enums import TaskType
from velour.metatypes import ImageMetadata
from velour.schemas import Raster


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


def download_image(datum: Datum) -> PIL.Image:
    """
    Download image using Datum.
    """
    url = datum.metadata["coco_url"]
    img_data = BytesIO(requests.get(url).content)
    return PIL.Image.open(img_data)


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


def _parse_categories(
    categories: list,
) -> Dict[int, Union[TaskType, Dict[str, str]]]:
    """
    Parse COCO categories into `velour.enums.TaskType` and `velour.Label`
    """
    return {
        category["id"]: {
            "task_type": (
                TaskType.OBJECT_DETECTION
                if category["isthing"]
                else TaskType.SEMANTIC_SEGMENTATION
            ),
            "labels": {
                "supercategory": category["supercategory"],
                "name": category["name"],
            },
        }
        for category in categories
    }


def _create_masks(filename: str) -> np.ndarray:
    """
    Convert the colors in the mask to ids.
    """
    mask = np.array(PIL.Image.open(filename)).astype(int)
    return mask[:, :, 0] + 256 * mask[:, :, 1] + (256**2) * mask[:, :, 2]


def create_annotations_from_instance_segmentations(
    image: dict,
    category_id_to_labels_and_task: Dict[int, Union[TaskType, Dict[str, str]]],
    mask_ids,
) -> List[Annotation]:
    return [
        Annotation(
            task_type=TaskType.OBJECT_DETECTION,
            labels=[
                Label(
                    key="supercategory",
                    value=str(
                        category_id_to_labels_and_task[
                            segmentation["category_id"]
                        ]["labels"]["supercategory"]
                    ),
                ),
                Label(
                    key="name",
                    value=str(
                        category_id_to_labels_and_task[
                            segmentation["category_id"]
                        ]["labels"]["name"]
                    ),
                ),
                Label(key="iscrowd", value=str(segmentation["iscrowd"])),
            ],
            raster=Raster.from_numpy(mask_ids == segmentation["id"]),
        )
        for segmentation in image["segments_info"]
        if category_id_to_labels_and_task[segmentation["category_id"]][
            "task_type"
        ]
        == TaskType.OBJECT_DETECTION
    ]


def create_annotations_from_semantic_segmentations(
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
            category_id_to_labels_and_task[category_id]["task_type"]
            == TaskType.SEMANTIC_SEGMENTATION
        ):
            for key, value in [
                (
                    "supercategory",
                    category_id_to_labels_and_task[category_id]["labels"][
                        "supercategory"
                    ],
                ),
                (
                    "name",
                    category_id_to_labels_and_task[category_id]["labels"][
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
            task_type=TaskType.SEMANTIC_SEGMENTATION,
            labels=[Label(key=key, value=str(value))],
            raster=Raster.from_numpy(semantic_masks[key][value]),
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
        image["id"]: _parse_image_to_datum(image) for image in data["images"]
    }

    # create groundtruths
    groundtruths = []
    for image in tqdm(data["annotations"], "Formatting"):
        # exract masks from annotations
        mask_ids = _create_masks(masks_path / image["file_name"])

        # create instance segmentations
        instance_annotations = create_annotations_from_instance_segmentations(
            image,
            category_id_to_labels_and_task,
            mask_ids,
        )

        # create semantic segmentations
        semantic_annotations = create_annotations_from_semantic_segmentations(
            image,
            category_id_to_labels_and_task,
            mask_ids,
        )

        # create groundTruth
        groundtruths.append(
            GroundTruth(
                datum=image_id_to_datum[image["image_id"]],
                annotations=instance_annotations + semantic_annotations,
            )
        )

    return groundtruths


def create_dataset_from_coco_panoptic(
    name: str = "coco2017-panoptic-semseg",
    destination: str = "./coco",
    coco_url: str = "http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip",
    limit: int = 0,
    delete_if_exists: bool = False,
) -> Dataset:
    """
    Creates Dataset and associated GroundTruths.

    Parameters
    ----------
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
    delete_if_exists : bool, default=False
        Reset the Velour dataset before attempting creation.

    """
    client = Client()

    # download and unzip coco dataset
    data = download_coco_panoptic(
        destination=destination,
        coco_url=coco_url,
    )

    # path of mask locations
    masks_path = destination / Path("annotations/panoptic_val2017/")

    # slice if limited
    if limit > 0:
        data["annotations"] = data["annotations"][:limit]

    # if reset, delete the dataset if it exists
    if delete_if_exists and client.get_dataset(name) is not None:
        client.delete_dataset(name, timeout=5)

    if client.get_dataset(name) is not None:
        dataset = Dataset.create(name)
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
        dataset = Dataset.create(
            name,
            metadata=metadata,
        )
        for gt in tqdm(gts, desc="Uploading"):
            dataset.add_groundtruth(gt)
        dataset.finalize()

    return dataset
