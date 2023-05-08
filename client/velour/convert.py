import gzip
import json
import tempfile
from pathlib import Path, PosixPath
from typing import Any, Dict, List, Union

import numpy as np
import PIL.Image
import requests
from chariot.datasets.dataset_version import DatasetVersion
from tqdm import tqdm

from velour.client import Dataset
from velour.data_types import (
    BoundingBox,
    BoundingPolygon,
    GroundTruthDetection,
    GroundTruthImageClassification,
    GroundTruthInstanceSegmentation,
    GroundTruthSemanticSegmentation,
    Image,
    Label,
    Point,
    PolygonWithHole,
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
            bbox=BoundingBox(
                ymin=box[0], xmin=box[1], ymax=box[2], xmax=box[3]
            ),
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

    def _get_segs_for_single_image(
        ann_dict: dict,
    ) -> List[_GroundTruthSegmentation]:
        mask = np.array(
            PIL.Image.open(masks_path / ann_dict["file_name"])
        ).astype(int)
        # convert the colors in the mask to ids
        mask_ids = (
            mask[:, :, 0] + 256 * mask[:, :, 1] + (256**2) * mask[:, :, 2]
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

    for ann in tqdm(annotations["annotations"]):
        segs = _get_segs_for_single_image(ann)
        dataset.add_groundtruth_segmentations(segs)


def retrieve_chariot_ds(manifest_url: str):
    """Retrieves and unpacks Chariot dataset annotations from a manifest url."""

    chariot_dataset = []

    # Create a temporary file
    with tempfile.TemporaryFile(mode="w+b") as f:

        # Download compressed jsonl file
        response = requests.get(manifest_url, stream=True)
        if response.status_code == 200:
            f.write(response.raw.read())
        f.flush()
        f.seek(0)

        # Unzip
        gzf = gzip.GzipFile(mode="rb", fileobj=f)
        jsonl = gzf.read().decode().strip().split("\n")

        # Parse into list of json object(s)
        for line in jsonl:
            chariot_dataset.append(json.loads(line))

    return chariot_dataset


def chariot_parse_image_classification_annotation(
    datum: dict,
) -> GroundTruthImageClassification:
    """Parses Chariot image classification annotation."""

    # Strip UID from URL path
    uid = Path(datum["path"]).stem

    gt_dets = []
    for annotation in datum["annotations"]:
        gt_dets.append(
            GroundTruthImageClassification(
                image=Image(uid=uid, height=None, width=None),
                labels=[
                    Label(key="class_label", value=annotation["class_label"])
                ],
            )
        )
    return gt_dets


def chariot_parse_image_segmentation_annotation(
    datum: dict,
) -> GroundTruthSemanticSegmentation:
    """Parses Chariot image segmentation annotation."""

    # Strip UID from URL path
    uid = Path(datum["path"]).stem

    annotated_regions = {}
    for annotation in datum["annotations"]:

        annotation_label = annotation["class_label"]

        # Create PolygonWithHole
        region = None

        # Contour name changes based on example...
        contour_key = "contours"
        if "contours" not in annotation and "contour" in annotation:
            contour_key = "contour"
        elif "contours" not in annotation:
            raise ("Missing contour key!")

        polygon = None
        hole = None

        # Create Bounding Polygon
        points = []
        for point in annotation[contour_key][0]:
            points.append(
                Point(
                    x=point["x"],
                    y=point["y"],
                )
            )
        polygon = BoundingPolygon(points)

        # Check if hole exists
        if len(annotation[contour_key]) > 1:
            # Create Bounding Polygon for Hole
            points = []
            for point in annotation[contour_key][1]:
                points.append(
                    Point(
                        x=point["x"],
                        y=point["y"],
                    )
                )
            hole = BoundingPolygon(points)

        # Populate PolygonWithHole
        region = PolygonWithHole(polygon=polygon, hole=hole)

        # Add annotated region to list
        if region is not None:
            if annotation_label not in annotated_regions:
                annotated_regions[annotation_label] = []
            annotated_regions[annotation_label].append(region)

    # Create a list of GroundTruths
    gt_dets = []
    for label in annotated_regions.keys():
        gt_dets.append(
            GroundTruthSemanticSegmentation(
                shape=annotated_regions[label],
                labels=[Label(key="class_label", value=label)],
                image=Image(uid=uid, height=None, width=None),
            )
        )
    return gt_dets


def chariot_parse_object_detection_annotation(
    datum: dict,
) -> GroundTruthDetection:
    """Parses Chariot object detection annotation."""

    # Strip UID from URL path
    uid = Path(datum["path"]).stem

    gt_dets = []
    for annotation in datum["annotations"]:
        gt_dets.append(
            GroundTruthDetection(
                bbox=BoundingBox(
                    xmin=annotation["bbox"]["xmin"],
                    ymin=annotation["bbox"]["ymin"],
                    xmax=annotation["bbox"]["xmax"],
                    ymax=annotation["bbox"]["ymax"],
                ),
                labels=[
                    Label(key="class_label", value=annotation["class_label"])
                ],
                image=Image(uid=uid, height=None, width=None),
            )
        )
    return gt_dets


def chariot_parse_text_sentiment_annotation(datum: dict):
    """Parses Chariot text sentiment annotation."""
    return None


def chariot_parse_text_summarization_annotation(datum: dict):
    """Parses Chariot text summarization annotation."""
    return None


def chariot_parse_text_token_classification_annotation(datum: dict):
    """Parses Chariot text token classification annotation."""
    return None


def chariot_parse_text_translation_annotation(datum: dict):
    """Parses Chariot text translation annotation."""
    return None


def chariot_ds_to_velour_ds(
    dsv: DatasetVersion, velour_name: str, use_training_manifest=True
):
    """Converts the annotations of a Chariot dataset to velour's format.

    Parameters
    ----------
    dsv
        Chariot DatasetVerion object
    velour_name
        Name of the new Velour dataset
    using_training_manifest
        (OPTIONAL) Defaults to true, setting false will use the evaluation manifest which is a
        super set of the training manifest. Not recommended as the evaluation manifest may
        contain unlabeled data.

    Returns
    -------
    List of Ground Truth Detections
    """

    manifest_url = (
        dsv.get_training_manifest_url()
        if use_training_manifest
        else dsv.get_evaluation_manifest_url()
    )

    chariot_ds = retrieve_chariot_ds(manifest_url)
    gt_dets = []

    for datum in chariot_ds:

        # Image Classification
        if dsv.supported_task_types.image_classification:
            gt_dets += chariot_parse_image_classification_annotation(datum)

        # Image Segmentation
        if dsv.supported_task_types.image_segmentation:
            gt_dets += chariot_parse_image_segmentation_annotation(datum)

        # Object Detection
        if dsv.supported_task_types.object_detection:
            gt_dets += chariot_parse_object_detection_annotation(datum)

        # Text Sentiment
        if dsv.supported_task_types.text_sentiment:
            pass

        # Text Summarization
        if dsv.supported_task_types.text_summarization:
            pass

        # Text Token Classifier
        if dsv.supported_task_types.text_token_classification:
            pass

        # Text Translation
        if dsv.supported_task_types.text_translation:
            pass

    return gt_dets
