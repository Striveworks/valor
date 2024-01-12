import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from velour import Annotation, GroundTruth, Prediction, enums, schemas
from velour.metatypes import ImageMetadata

# https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
COLOR_MAP = [
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 190),
    (0, 128, 128),
    (230, 190, 255),
    (170, 110, 40),
    (255, 250, 200),
    (128, 0, 0),
    (170, 255, 195),
    (128, 128, 0),
    (255, 215, 180),
    (0, 0, 128),
    (128, 128, 128),
]


def _polygons_to_binary_mask(
    polys: List[schemas.Polygon], img_w, img_h
) -> np.ndarray:
    """note there's some aliasing/areas differences between this
    and e.g. postgis, so this method should only be used for visualization
    """
    mask = Image.new("1", (img_w, img_h), (False,))
    draw = ImageDraw.Draw(mask)
    for poly in polys:
        draw.polygon(poly.boundary.tuple_list(), fill=(True,))
        if poly.holes is not None:
            for hole in poly.holes:
                draw.polygon(hole.tuple_list(), fill=(False,))

    return np.array(mask)


def create_combined_segmentation_mask(
    annotated_datums: List[Union[GroundTruth, Prediction]],
    label_key: str,
    task_type: Union[enums.TaskType, None] = None,
) -> Tuple[Image.Image, Dict[str, Image.Image]]:
    """
    Creates a combined segmentation mask from a list of segmentations.

    Parameters
    -------
    annotated_datums : List[Union[GroundTruth, Prediction]]
        A list of segmentations. These all must have the same `image` attribute.
    label_key : str
        The label key to use.
    task_type : enums.TaskType
        The associated task type.


    Returns
    -------
    tuple
        The first element of the tuple is the combined mask, as an RGB PIL image. The second
        element is a color legend: it's a dict with keys the unique values of the labels and the
        values of the dict are PIL image swatches of the color corresponding to the label value.

    Raises
    ------
    RuntimeError
        If all segmentations don't belong to the same image, or there is a
        segmentation that doesn't have `label_key` as the key of one of its labels.
    ValueError
        If there aren't any segmentations.
    """

    if len(annotated_datums) == 0:
        raise ValueError("`segs` cannot be empty.")

    if (
        len(
            set(
                [
                    annotated_datum.datum.uid
                    for annotated_datum in annotated_datums
                ]
            )
        )
        > 1
    ):
        raise RuntimeError(
            "Expected all segmentation to belong to the same image"
        )

    # Validate task type
    if task_type is not None and task_type not in [
        enums.TaskType.DETECTION,
        enums.TaskType.SEGMENTATION,
    ]:
        raise RuntimeError(
            "Expected either Instance or Semantic segmentation task_type."
        )

    # Create valid task type list
    if task_type is None:
        task_types = [
            enums.TaskType.DETECTION,
            enums.TaskType.SEGMENTATION,
        ]
    else:
        task_types = [task_type]

    # unpack raster annotations
    annotations: List[Annotation] = []
    for annotated_datum in annotated_datums:
        for annotation in annotated_datum.annotations:
            if annotation.task_type in task_types:
                annotations.append(annotation)

    label_values = []
    for annotation in annotations:
        for label in annotation.labels:
            if label.key == label_key:
                label_values.append(label.value)
    if not label_values:
        raise RuntimeError(
            f"Annotation doesn't have a label with key `{label_key}`"
        )

    unique_label_values = list(set(label_values))
    label_value_to_color = {
        v: COLOR_MAP[i] for i, v in enumerate(unique_label_values)
    }
    seg_colors = [label_value_to_color[v] for v in label_values]

    image = ImageMetadata.from_datum(annotated_datums[0].datum)
    img_w, img_h = image.width, image.height

    combined_mask = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    for annotation, color in zip(annotations, seg_colors):
        if annotation.raster is not None:
            mask = annotation.raster.to_numpy()
        elif annotation.multipolygon is not None:
            mask = _polygons_to_binary_mask(
                annotation.multipolygon.polygons,
                img_w=img_w,
                img_h=img_h,
            )
        else:
            continue

        combined_mask[np.where(mask)] = color

    legend = {
        v: Image.new("RGB", (20, 20), color)
        for v, color in label_value_to_color.items()
    }

    return Image.fromarray(combined_mask), legend


def draw_detections_on_image(
    detections: List[Union[GroundTruth, Prediction]],
    img: Image.Image,
) -> Image.Image:
    """
    Draws detections (bounding boxes and labels) on an image.

    Parameters
    -------
    detections : List[Union[GroundTruth, Prediction]]
        A list of `GroundTruths` or `Predictions` to draw on the image.
    img : Image.Image
        The image to draw the detections on.


    Returns
    -------
    img : Image.Image
        An image with the detections drawn on.
    """

    annotations = []
    for datum in detections:
        annotations.extend(datum.annotations)

    for i, detection in enumerate(annotations):
        if detection.task_type in [enums.TaskType.DETECTION]:
            img = _draw_detection_on_image(detection, img, inplace=i != 0)
    return img


def draw_bounding_box_on_image(
    bounding_box: schemas.BoundingBox,
    img: Image.Image,
    color: Tuple[int, int, int] = (255, 0, 0),
) -> Image.Image:
    """Draws a bounding polygon on an image. This operation is not done in-place

    Parameters
    ----------
    bounding_box
        bounding box to draw on the image
    img
        pillow image to draw on
    color
        RGB tuple of the color to use

    Returns
    -------
    img
        pillow image with bounding box drawn on it
    """
    return _draw_bounding_polygon_on_image(
        bounding_box.polygon, img, color=color, inplace=False
    )


def _draw_detection_on_image(
    detection: Annotation, img: Image.Image, inplace: bool
) -> Image.Image:
    """Draw a detection on an image."""
    text = ", ".join(
        [f"{label.key}:{label.value}" for label in detection.labels]
    )
    if detection.polygon is not None:
        img = _draw_bounding_polygon_on_image(
            detection.polygon.boundary,
            img,
            inplace=inplace,
            text=text,
        )
    elif detection.bounding_box is not None:
        img = _draw_bounding_polygon_on_image(
            detection.bounding_box.polygon,
            img,
            inplace=inplace,
            text=text,
        )

    return img


def _draw_bounding_polygon_on_image(
    polygon: schemas.BasicPolygon,
    img: Image.Image,
    color: Tuple[int, int, int] = (255, 0, 0),
    inplace: bool = False,
    text: Optional[str] = None,
    font_size: int = 24,
) -> Image.Image:
    """Draw a bounding polygon on an image."""
    img = img if inplace else img.copy()
    img_draw = ImageDraw.Draw(img)

    img_draw.polygon(
        [(p.x, p.y) for p in polygon.points],
        outline=color,
    )

    if text is not None:
        _write_text(
            font_size=font_size,
            text=text,
            boundary=polygon,
            draw=img_draw,
            color=color,
        )
    return img


def _write_text(
    font_size: int,
    text: str,
    boundary: schemas.BasicPolygon,
    draw: ImageDraw.Draw,
    color: str,
):
    """Write text on an image."""
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        try:
            font = ImageFont.truetype("Arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()

    _, _, text_width, text_height = font.getbbox(text)
    if boundary.ymin > text_height:
        text_bottom = boundary.ymin
    else:
        text_bottom = boundary.ymax + text_height

    margin = math.ceil(0.05 * text_height) - 1
    draw.rectangle(
        [
            (boundary.xmin, text_bottom - text_height - 2 * margin),
            (boundary.xmin + text_width, text_bottom),
        ],
        fill=color,
    )
    draw.text(
        (boundary.xmin + margin, text_bottom - text_height - margin),
        text,
        fill="black",
        font=font,
    )


def draw_raster_on_image(
    raster: schemas.Raster,
    img: Image.Image,
    color: Tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.4,
) -> Image.Image:
    """Draws the raster on top of an image. This operation is not done in-place

    Parameters
    ----------
    img
        pillow image to draw on.
    color
        RGB tuple of the color to use
    alpha
        alpha (transparency) value of the mask. 0 is fully transparent, 1 is fully opaque
    """
    img = img.copy()
    binary_mask = raster.to_numpy()
    mask_arr = np.zeros(
        (binary_mask.shape[0], binary_mask.shape[1], 3), dtype=np.uint8
    )
    mask_arr[binary_mask] = color
    mask_img = Image.fromarray(mask_arr)
    blend = Image.blend(img, mask_img, alpha=alpha)
    img.paste(blend, (0, 0), mask=Image.fromarray(binary_mask))

    return img
