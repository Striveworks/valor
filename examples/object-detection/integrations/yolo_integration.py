import json
import os
import re
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Generator, List

import numpy as np
import PIL.Image
import requests
import ultralytics
from shapely import geometry, ops
from skimage import measure
from tqdm import tqdm

from valor import Annotation, Datum, Label, Prediction
from valor.enums import AnnotationType
from valor.metatypes import ImageMetadata
from valor.schemas import Box, MultiPolygon, Polygon, Raster


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


def _convert_yolo_segmentation(
    raw,
    height: int,
    width: int,
    resample: PIL.Image.Resampling = PIL.Image.Resampling.BILINEAR,
):
    """Resizes the raw binary mask provided by the YOLO inference to the original image size."""
    mask = np.asarray(raw.cpu())
    mask[mask == 1.0] = 255
    img = PIL.Image.fromarray(np.uint8(mask))
    img = img.resize((width, height), resample=resample)
    mask = np.array(img, dtype=np.uint8) >= 128
    return mask


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


def download_image(url: str) -> PIL.Image.Image:
    """
    Download image using Datum.
    """
    if not isinstance(url, str):
        raise TypeError("datum.metadata['coco_url'] is not type 'str'.")
    img_data = BytesIO(requests.get(url).content)
    return PIL.Image.open(img_data)


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


def create_bounding_box_detection(
    result, datum: Datum, label_key: str = "class"
) -> Prediction:
    """Parses Ultralytic's result for an object detection task."""

    # Extract data
    result = result[0]
    probabilities = [conf.item() for conf in result.boxes.conf]
    labels = [result.names[int(pred.item())] for pred in result.boxes.cls]
    bboxes = [np.asarray(box.cpu()) for box in result.boxes.xyxy]

    # validate dimensions
    image_metadata = ImageMetadata(datum)
    if image_metadata.height != result.orig_shape[0]:
        raise RuntimeError
    if image_metadata.width != result.orig_shape[1]:
        raise RuntimeError

    # Create scored label list
    labels = [
        Label(key=label_key, value=label, score=probability)
        for label, probability in list(zip(labels, probabilities))
    ]

    # Extract Bounding Boxes
    bboxes = [
        Box.from_extrema(
            xmin=int(box[0]),
            ymin=int(box[1]),
            xmax=int(box[2]),
            ymax=int(box[3]),
        )
        for box in bboxes
    ]

    return Prediction(
        datum=datum,
        annotations=[
            Annotation(
                labels=[scored_label],
                bounding_box=bbox,
                is_instance=True,
            )
            for bbox, scored_label in list(zip(bboxes, labels))
        ],
    )


def create_raster_detection(
    result,
    datum: Datum,
    label_key: str = "class",
    resample: PIL.Image.Resampling = PIL.Image.Resampling.BILINEAR,
) -> Prediction:
    """Parses Ultralytic's result for an image segmentation task."""

    result = result[0]

    if result.masks is None:
        return Prediction(
            datum=datum,
            annotations=[],
        )

    # Extract data
    probabilities = [conf.item() for conf in result.boxes.conf]
    labels = [result.names[int(pred.item())] for pred in result.boxes.cls]
    masks = [mask for mask in result.masks.data]

    # validate dimensions
    image_metadata = ImageMetadata(datum)
    if image_metadata.height != result.orig_shape[0]:
        raise RuntimeError
    if image_metadata.width != result.orig_shape[1]:
        raise RuntimeError

    # Create scored label list
    labels = [
        Label(key=label_key, value=label, score=probability)
        for label, probability in list(zip(labels, probabilities))
    ]

    # Extract masks
    masks = [
        _convert_yolo_segmentation(
            raw,
            height=image_metadata.height,
            width=image_metadata.width,
            resample=resample,
        )
        for raw in result.masks.data
    ]

    # create prediction
    return Prediction(
        datum=datum,
        annotations=[
            Annotation(
                labels=[scored_label],
                raster=Raster.from_numpy(mask),
                is_instance=True,
            )
            for mask, scored_label in list(zip(masks, labels))
        ],
    )


def create_multipolygon_detection(
    result,
    datum: Datum,
    label_key: str = "class",
    resample: PIL.Image.Resampling = PIL.Image.Resampling.BILINEAR,
):
    prediction = create_raster_detection(
        result=result, datum=datum, label_key=label_key, resample=resample
    )
    annotations = []
    for annotation in prediction.annotations:
        array = annotation.raster.array
        multipolygon = bitmask_to_multipolygon_raster(array)
        if multipolygon is not None:
            annotation.raster = multipolygon
            annotations.append(annotation)
    prediction.annotations = annotations
    return prediction


def create_polygon_detection(
    result,
    datum: Datum,
    label_key: str = "class",
    resample: PIL.Image.Resampling = PIL.Image.Resampling.BILINEAR,
):
    prediction = create_raster_detection(
        result=result, datum=datum, label_key=label_key, resample=resample
    )
    annotations = []
    for annotation in prediction.annotations:
        array = annotation.raster.array
        polygon = bitmask_to_polygon(array)
        if polygon is not None:
            annotation.polygon = polygon
            annotation.raster = None
            annotations.append(annotation)
    prediction.annotations = annotations
    return prediction


def run_inference(
    path: Path,
    destination: Path,
    coco_url: str,
    limit: int,
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
    limit : int
        Limits the number of datums. Set to 0 for no action.
    """

    coco_path = Path(path) / Path(destination)

    # download and unzip coco dataset
    data = download_coco_panoptic(
        destination=coco_path,
        coco_url=coco_url,
    )

    # create datums
    datums = [_parse_image_to_datum(image) for image in data["images"]]
    if limit > 0 and limit < len(datums):
        datums = datums[:limit]

    inference_engine = ultralytics.YOLO("yolov8n-seg.pt")

    filepath_bbox = Path(path) / Path("pd_objdet_yolo_bbox.jsonl")
    filepath_polygon = Path(path) / Path("pd_objdet_yolo_polygon.jsonl")
    filepath_multipolygon = Path(path) / Path(
        "pd_objdet_yolo_multipolygon.jsonl"
    )
    filepath_raster = Path(path) / Path("pd_objdet_yolo_raster.jsonl")

    with open(filepath_bbox, "w") as fbox:
        with open(filepath_polygon, "w") as fpolygon:
            with open(filepath_multipolygon, "w") as fmultipolygon:
                with open(filepath_raster, "w") as fraster:

                    for datum in tqdm(datums):

                        image = download_image(datum.metadata["coco_url"])

                        results = inference_engine(image, verbose=False)

                        # convert result into Valor Bounding Box prediction
                        prediction = create_bounding_box_detection(
                            results,  # raw inference
                            datum=datum,  # valor datum
                            label_key="name",  # label_key override
                        )
                        fbox.write(json.dumps(prediction.encode_value()))
                        fbox.write("\n")

                        # convert result into Valor Bounding Polygon prediction
                        prediction = create_polygon_detection(
                            results,  # raw inference
                            datum=datum,  # valor datum
                            label_key="name",  # label_key override
                        )
                        fpolygon.write(json.dumps(prediction.encode_value()))
                        fpolygon.write("\n")

                        # convert result into Valor MultiPolygon Raster prediction
                        prediction = create_multipolygon_detection(
                            results,  # raw inference
                            datum=datum,  # valor datum
                            label_key="name",  # label_key override
                        )
                        fmultipolygon.write(
                            json.dumps(prediction.encode_value())
                        )
                        fmultipolygon.write("\n")

                        # convert result into Valor Bitmask Raster prediction
                        prediction = create_raster_detection(
                            results,  # raw inference
                            datum=datum,  # valor datum
                            label_key="name",  # label_key override
                        )
                        fraster.write(json.dumps(prediction.encode_value()))
                        fraster.write("\n")


def get_instance_predictions(
    coco_uids: list[str],
    dtype: AnnotationType,
    chunk_size: int = 1,
    limit: int = 0,
    from_cache: bool = True,
) -> Generator[List[Prediction], None, None]:
    """
    Retrieves YOLO object detection predictions.

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

    # paths
    path = Path(os.path.dirname(os.path.realpath(__file__)))
    coco_url = "http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip"
    cache_url = "https://pub-fae71003f78140bdaedf32a7c8d331d2.r2.dev/"

    # get filename
    filenames = {
        AnnotationType.BOX: "pd_objdet_yolo_bbox.jsonl",
        AnnotationType.POLYGON: "pd_objdet_yolo_polygon.jsonl",
        AnnotationType.MULTIPOLYGON: "pd_objdet_yolo_multipolygon.jsonl",
        AnnotationType.RASTER: "pd_objdet_yolo_raster.jsonl",
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
        run_inference(
            path=path,
            destination=Path("coco"),
            coco_url=coco_url,
            limit=limit,
        )

    with open(filepath, "r") as f:
        pattern = re.compile(r'"uid":\s*"(\d+)"')
        count = 0
        chunks = []
        for line in f:
            match = pattern.search(line)
            if not match:
                continue
            elif match.group(1) not in coco_uids:
                continue
            pd_dict = json.loads(line)
            pd = Prediction.decode_value(pd_dict)
            chunks.append(pd)
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

    for chunk in get_instance_predictions(
        coco_uids=[
            "139",
            "285",
            "632",
            "724",
            "776",
            "785",
            "802",
            "872",
        ],
        dtype=AnnotationType.BOX,
        chunk_size=2,
        limit=8,
        from_cache=True,
    ):
        print(chunk[0].datum.uid, chunk[1].datum.uid)
