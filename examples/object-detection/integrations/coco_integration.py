import json
import os
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Dict, Generator, List, Union

import numpy as np
import PIL.Image
import requests
from shapely import geometry, ops
from skimage import measure
from tqdm import tqdm

from valor import Annotation, Datum, GroundTruth, Label
from valor.enums import AnnotationType, TaskType
from valor.metatypes import ImageMetadata
from valor.schemas import Box, MultiPolygon, Polygon, Raster


def download_image(datum: Datum) -> PIL.Image.Image:
    """
    Download image using Datum.
    """
    url = datum.metadata["coco_url"]
    if not isinstance(url, str):
        raise TypeError("datum.metadata['coco_url'] is not type 'str'.")
    img_data = BytesIO(requests.get(url).content)
    return PIL.Image.open(img_data)


def download_data_if_not_exists(
    filename: str,
    filepath: Path,
    url: str,
):
    """Download the data from a public bucket if it doesn't exist locally."""

    if not os.path.exists(filepath):
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            total_size = int(response.headers.get("content-length", 0))
            with open(filepath, "wb") as f:
                with tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=filename,
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            f.write(chunk)
                            pbar.update(1024)
        else:
            raise RuntimeError(response)
    else:
        print(f"{filename} already exists locally.")


def download_coco_panoptic(
    destination: Path = Path("./coco"),
    coco_url: str = "http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip",
) -> dict:
    """
    Download the COCO panoptic dataset.

    Parameters
    ----------
    destination: Path
        The filepath where the dataset will be stored.
    coco_url: str
        The url where the coco dataset is stored.

    Returns
    -------
    dict
        A dictionary containing the coco dataset.
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


def convert_bitmask_to_bbox(bitmask: np.ndarray) -> Box:
    """
    Converts a bitmask to a Valor Box schema.

    Parameters
    ----------
    bitmask: np.ndarray
        The bitmask to convert.

    Returns
    -------
    valor.schemas.Box
    """
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


def convert_bitmask_to_multipolygon_raster(bitmask: np.ndarray) -> Raster:
    """
    Converts a bitmask to a Valor Raster schema.

    Parameters
    ----------
    bitmask: np.ndarray
        The bitmask to convert.

    Returns
    -------
    valor.schemas.Raster
    """
    bitmask = np.array(bitmask, dtype=bool)
    labeled_array, num_features = measure.label(
        bitmask, background=0, return_num=True
    )  # type: ignore - skimage
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


def convert_bitmask_to_polygon(bitmask: np.ndarray) -> Polygon:
    """
    Converts a bitmask to a Valor Polygon schema.

    Parameters
    ----------
    bitmask: np.ndarray
        The bitmask to convert.

    Returns
    -------
    valor.schemas.Polygon
    """
    bitmask = np.array(bitmask, dtype=bool)
    labeled_array, num_features = measure.label(
        bitmask,
        background=0,
        return_num=True,
    )  # type: ignore - skimage
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


def _parse_image_to_datum(image: dict) -> Datum:
    """
    Parse COCO image to Valor Datum.
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
    Parse COCO categories into `valor.enums.TaskType` and `valor.Label`.
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


def create_bounding_boxes(
    image: dict,
    category_id_to_labels_and_task: Dict[int, Union[TaskType, Dict[str, str]]],
    mask_ids,
) -> List[Annotation]:
    """
    Create bounding box annotations from COCO annotations.
    """
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
            bounding_box=convert_bitmask_to_bbox(
                mask_ids == segmentation["id"]
            ),
            is_instance=True,
        )
        for segmentation in image["segments_info"]
        if category_id_to_labels_and_task[segmentation["category_id"]][
            "is_instance"
        ]  # type: ignore - dict typing
        is True
        and convert_bitmask_to_bbox(mask_ids == segmentation["id"]) is not None
    ]


def create_bounding_polygons(
    image: dict,
    category_id_to_labels_and_task: Dict[int, Union[TaskType, Dict[str, str]]],
    mask_ids,
) -> List[Annotation]:
    """
    Create bounding polygon annotations from COCO annotations.
    """
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
            polygon=convert_bitmask_to_polygon(mask_ids == segmentation["id"]),
            is_instance=True,
        )
        for segmentation in image["segments_info"]
        if category_id_to_labels_and_task[segmentation["category_id"]][
            "is_instance"
        ]  # type: ignore - dict typing
        is True
        and convert_bitmask_to_polygon(mask_ids == segmentation["id"])
        is not None
    ]


def create_raster_from_bitmask(
    image: dict,
    category_id_to_labels_and_task: Dict[int, Union[TaskType, Dict[str, str]]],
    mask_ids,
) -> List[Annotation]:
    """
    Create raster annotations from COCO annotations.
    """
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
    """
    Create multipolygon annotations from COCO annotations.
    """
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
            raster=convert_bitmask_to_multipolygon_raster(
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
    dtype: AnnotationType = AnnotationType.RASTER,
):
    """
    Create semantic annotations from COCO annotations.
    """

    if dtype not in [AnnotationType.MULTIPOLYGON, AnnotationType.RASTER]:
        raise ValueError(dtype)

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
            raster=(
                Raster.from_numpy(semantic_masks[key][value])
                if dtype == AnnotationType.RASTER
                else convert_bitmask_to_multipolygon_raster(
                    semantic_masks[key][value]
                )
            ),
            is_instance=False,
        )
        for key in semantic_masks
        for value in semantic_masks[key]
    ]


def create_instance_groundtruths_file(
    dtype: AnnotationType,
    filename: str,
    path: Path,
    destination: Path,
    coco_url: str,
    limit: int,
):
    if dtype not in [
        AnnotationType.BOX,
        AnnotationType.POLYGON,
        AnnotationType.MULTIPOLYGON,
        AnnotationType.RASTER,
    ]:
        raise ValueError(dtype)

    # download and unzip coco dataset
    coco_path = Path(path) / Path(destination)
    data = download_coco_panoptic(
        destination=coco_path,
        coco_url=coco_url,
    )

    # path of mask locations
    masks_path = coco_path / Path("annotations/panoptic_val2017/")

    # slice if limited
    if limit > 0:
        data["annotations"] = data["annotations"][:limit]

    # get filepath
    filepath = Path(path) / Path(filename)

    # get creator function
    functions = {
        AnnotationType.BOX: create_bounding_boxes,
        AnnotationType.POLYGON: create_bounding_polygons,
        AnnotationType.MULTIPOLYGON: create_raster_from_multipolygon,
        AnnotationType.RASTER: create_raster_from_bitmask,
    }
    create = functions[dtype]

    with open(filepath, mode="wb") as f:

        # extract labels from categories
        category_id_to_labels_and_task = _parse_categories(data["categories"])

        # create datums
        image_id_to_datum = {
            image["id"]: _parse_image_to_datum(image)
            for image in data["images"]
        }

        # create groundtruths
        for image in tqdm(data["annotations"], "Saving to JSON."):
            # exract masks from annotations
            mask_ids = _create_masks(masks_path / image["filename"])

            # create annotations
            annotations = create(
                image,
                category_id_to_labels_and_task,
                mask_ids,
            )
            gt = GroundTruth(
                datum=image_id_to_datum[image["image_id"]],
                annotations=annotations,
            )
            f.write(json.dumps(gt.encode_value()).encode("utf-8"))
            f.write("\n".encode("utf-8"))


def create_semantic_groundtruths_file(
    dtype: AnnotationType,
    path: Path,
    filename: str,
    destination: Path,
    coco_url: str,
    limit: int,
):
    if dtype not in [
        AnnotationType.MULTIPOLYGON,
        AnnotationType.RASTER,
    ]:
        raise ValueError(dtype)

    # download and unzip coco dataset
    coco_path = path / destination
    data = download_coco_panoptic(
        destination=coco_path,
        coco_url=coco_url,
    )

    # path of mask locations
    masks_path = coco_path / Path("annotations/panoptic_val2017/")

    # slice if limited
    if limit > 0:
        data["annotations"] = data["annotations"][:limit]

    # get filepath
    filepath = path / Path(filename)

    with open(filepath, mode="wb") as f:

        # extract labels from categories
        category_id_to_labels_and_task = _parse_categories(data["categories"])

        # create datums
        image_id_to_datum = {
            image["id"]: _parse_image_to_datum(image)
            for image in data["images"]
        }

        # create groundtruths
        for image in tqdm(data["annotations"], "Saving to JSON."):
            # exract masks from annotations
            mask_ids = _create_masks(masks_path / image["filename"])

            # create semantic segmentations
            semantic_annotations = create_semantic_segmentations(
                image,
                category_id_to_labels_and_task,
                mask_ids,
                dtype=dtype,
            )
            gt = GroundTruth(
                datum=image_id_to_datum[image["image_id"]],
                annotations=semantic_annotations,
            )
            f.write(json.dumps(gt.encode_value()).encode("utf-8"))
            f.write("\n".encode("utf-8"))


def get_instance_groundtruths(
    dtype: AnnotationType,
    chunk_size: int = 1,
    limit: int = 0,
    from_cache: bool = True,
) -> Generator[List[GroundTruth], None, None]:
    """
    Retrieves COCO object detection groundtruths from a variety of sources.

    Parameters
    ----------
    dtype : AnnotationType
        The desired annotation type.
    chunk_size : int, default=1
        The number of groundtruths returned per call.
    limit : int, default=0
        The maximum number of groundtruths returned. Defaults to all.
    from_cache : bool, default=True
        Retrieve cached groundtruths rather than regenerate.
    """

    if dtype not in [
        AnnotationType.BOX,
        AnnotationType.POLYGON,
        AnnotationType.MULTIPOLYGON,
        AnnotationType.RASTER,
    ]:
        raise ValueError(dtype)

    # paths
    path = Path(os.path.dirname(os.path.realpath(__file__)))
    coco_url = "http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip"
    cache_url = "https://pub-fae71003f78140bdaedf32a7c8d331d2.r2.dev/"

    # get filename
    filenames = {
        AnnotationType.BOX: "gt_objdet_coco_bbox.jsonl",
        AnnotationType.POLYGON: "gt_objdet_coco_polygon.jsonl",
        AnnotationType.MULTIPOLYGON: "gt_objdet_coco_raster_multipolygon.jsonl",
        AnnotationType.RASTER: "gt_objdet_coco_raster_bitmask.jsonl",
    }
    filename = filenames[dtype]
    filepath = path / Path(filename)

    if from_cache:
        download_data_if_not_exists(
            filename=filename,
            filepath=filepath,
            url=f"{cache_url}{filename}",
        )
    else:
        create_instance_groundtruths_file(
            dtype=dtype,
            filename=filename,
            path=path,
            destination=Path("coco"),
            coco_url=coco_url,
            limit=limit,
        )

    with open(filepath, "r") as f:
        count = 0
        chunks = []
        for line in f:
            gt_dict = json.loads(line)
            gt = GroundTruth.decode_value(gt_dict)
            chunks.append(gt)
            count += 1
            if count >= limit and limit > 0:
                break
            elif len(chunks) < chunk_size:
                continue

            yield chunks
            chunks = []
        if chunks:
            yield chunks


def get_semantic_groundtruths(
    dtype: AnnotationType,
    chunk_size: int = 1,
    limit: int = 0,
    from_cache: bool = True,
) -> Generator[List[GroundTruth], None, None]:
    """
    Retrieves COCO semantic segmenations groundtruths from a variety of sources.

    Parameters
    ----------
    dtype : AnnotationType
        The desired annotation type.
    chunk_size : int, default=1
        The number of groundtruths returned per call.
    limit : int, default=0
        The maximum number of groundtruths returned. Defaults to all.
    from_cache : bool, default=True
        Retrieve cached groundtruths rather than regenerate.
    """

    if dtype not in [
        AnnotationType.MULTIPOLYGON,
        AnnotationType.RASTER,
    ]:
        raise ValueError(dtype)

    # paths
    path = Path(os.path.dirname(os.path.realpath(__file__)))
    coco_url = "http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip"
    cache_url = "https://pub-fae71003f78140bdaedf32a7c8d331d2.r2.dev/"

    # get filename

    filenames = {
        AnnotationType.MULTIPOLYGON: "gt_semseg_coco_raster_multipolygon.jsonl",
        AnnotationType.RASTER: "gt_semseg_coco_raster_bitmask.jsonl",
    }
    filename = filenames[dtype]
    filepath = path / Path(filename)

    if from_cache:
        download_data_if_not_exists(
            filename=filename,
            filepath=filepath,
            url=f"{cache_url}{filename}",
        )
    else:
        create_semantic_groundtruths_file(
            dtype=dtype,
            filename=filename,
            path=path,
            destination=Path("coco"),
            coco_url=coco_url,
            limit=limit,
        )

    with open(filepath, "r") as f:
        count = 0
        chunks = []
        for line in f:
            gt_dict = json.loads(line)
            gt = GroundTruth.decode_value(gt_dict)
            chunks.append(gt)
            count += 1
            if count >= limit and limit > 0:
                break
            elif len(chunks) < chunk_size:
                continue

            yield chunks
            chunks = []
        if chunks:
            yield chunks


if __name__ == "__main__":

    for chunk in get_instance_groundtruths(
        dtype=AnnotationType.BOX,
        chunk_size=2,
        limit=8,
        from_cache=True,
    ):
        print(chunk[0].datum.uid, chunk[1].datum.uid)
