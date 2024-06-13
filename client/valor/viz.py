import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from valor import Annotation, GroundTruth, Prediction, schemas

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
        draw.polygon(poly.boundary, fill=(True,))  # type: ignore
        if poly.holes is not None:
            for hole in poly.holes:
                draw.polygon(hole, fill=(False,))  # type: ignore

    return np.array(mask)


def create_combined_segmentation_mask(
    annotated_datum: Union[GroundTruth, Prediction],
    label_key: str,
    filter_on_instance_segmentations: bool = False,
) -> Tuple[Image.Image, Dict[str, Image.Image]]:
    """
    Creates a combined segmentation mask from a list of segmentations.

    Parameters
    -------
    annotated_datum : Union[GroundTruth, Prediction]
        A list of segmentations. These all must have the same `image` attribute.
    label_key : str
        The label key to use.
    filter_on_instance_segmentations : bool, optional
        Whether to filter on instance segmentations or not.

    Returns
    -------
    tuple
        The first element of the tuple is the combined mask, as an RGB PIL image. The second
        element is a color legend: it's a dict with the unique labels as keys and the
        PIL image swatches as values.

    Raises
    ------
    RuntimeError
        If all segmentations don't belong to the same image or there is a
        segmentation that doesn't have `label_key` as the key of one of its labels.
    ValueError
        If there aren't any segmentations.
    """

    # validate input type
    if not isinstance(annotated_datum, (GroundTruth, Prediction)):
        raise ValueError("Expected either a 'GroundTruth' or 'Prediction'")

    # verify there are a nonzero number of annotations
    if len(annotated_datum.annotations) == 0:
        raise ValueError("annotations cannot be empty.")

    # validate raster size
    img_h = None
    img_w = None
    for annotation in annotated_datum.annotations:
        raster = annotation.raster
        if raster.get_value() is None:
            raise ValueError("No raster exists.")
        if img_h is None:
            img_h = raster.height
        if img_w is None:
            img_w = raster.width
        if (img_h != raster.height) or (img_w != raster.width):
            raise ValueError(
                f"Size mismatch between rasters. {(img_h, img_w)} != {(raster.height, raster.width)}"
            )
    if img_h is None or img_w is None:
        raise ValueError(
            f"Segmentation bounds not properly defined. {(img_h, img_w)}"
        )

    # unpack raster annotations
    annotations: List[Annotation] = []
    for annotation in annotated_datum.annotations:
        if (
            annotation.is_instance or False
        ) == filter_on_instance_segmentations:
            annotations.append(annotation)

    # unpack label values
    label_values = []
    for annotation in annotations:
        for label in annotation.labels:
            if label.key == label_key:
                label_values.append(label.value)
    if not label_values:
        raise RuntimeError(
            f"Annotation doesn't have a label with key `{label_key}`"
        )

    # assign label coloring
    unique_label_values = list(set(label_values))
    label_value_to_color = {
        v: COLOR_MAP[i] for i, v in enumerate(unique_label_values)
    }
    seg_colors = [label_value_to_color[v] for v in label_values]

    # create mask
    combined_mask = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    for annotation, color in zip(annotations, seg_colors):
        raster = annotation.raster
        if raster.get_value() is None:
            raise ValueError("No raster exists.")
        if raster.array is not None:
            if raster.geometry is None:
                mask = raster.array
            elif isinstance(raster.geometry, schemas.MultiPolygon):
                mask = _polygons_to_binary_mask(
                    raster.geometry.to_polygons(),
                    img_w=img_w,
                    img_h=img_h,
                )
            elif isinstance(raster.geometry, (schemas.Box, schemas.Polygon)):
                mask = _polygons_to_binary_mask(
                    [raster.geometry],
                    img_w=img_w,
                    img_h=img_h,
                )
            else:
                continue
            combined_mask[np.where(mask)] = color
        else:
            continue

    legend = {
        v: Image.new("RGB", (20, 20), color)
        for v, color in label_value_to_color.items()
    }

    return Image.fromarray(combined_mask), legend


def draw_bounding_box_on_image(
    bounding_box: schemas.Box,
    img: Image.Image,
    color: Tuple[int, int, int] = (255, 0, 0),
) -> Image.Image:
    """Draws a bounding polygon on an image. This operation is not done in place.

    Parameters
    ----------
    bounding_box
        Bounding box to draw on the image.
    img
        Pillow image to draw on.
    color
        RGB tuple of the color to use.

    Returns
    -------
    img
        Pillow image with bounding box drawn on it.
    """
    coords = bounding_box.get_value()
    return _draw_bounding_polygon_on_image(
        schemas.Polygon(coords), img, color=color, inplace=False
    )


def _draw_detection_on_image(
    detection: Annotation, img: Image.Image, inplace: bool
) -> Image.Image:
    """Draw a detection on an image."""
    text = ", ".join(
        [f"{label.key}:{label.value}" for label in detection.labels]
    )
    box = detection.bounding_box
    polygon = detection.polygon
    if polygon is not None:
        img = _draw_bounding_polygon_on_image(
            polygon,
            img,
            inplace=inplace,
            text=text,
        )
    elif box.get_value() is not None:
        img = _draw_bounding_polygon_on_image(
            box,
            img,
            inplace=inplace,
            text=text,
        )

    return img


def _draw_bounding_polygon_on_image(
    polygon: schemas.Polygon,
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
        [p for p in polygon.boundary],
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
    boundary: schemas.Polygon,
    draw: ImageDraw.ImageDraw,
    color: Union[Tuple[int, int, int], str],
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
        (
            (boundary.xmin, text_bottom - text_height - 2 * margin),
            (boundary.xmin + text_width, text_bottom),
        ),
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
    """Draws the raster on top of an image. This operation is not done in place.

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
    binary_mask = raster.array
    mask_arr = np.zeros(
        (binary_mask.shape[0], binary_mask.shape[1], 3), dtype=np.uint8
    )
    mask_arr[binary_mask] = color
    mask_img = Image.fromarray(mask_arr)

    if mask_img.size != img.size:
        raise ValueError("Input image and raster must be the same size.")
    blend = Image.blend(img, mask_img, alpha=alpha)
    img.paste(blend, (0, 0), mask=Image.fromarray(binary_mask))

    return img


def draw_detections_on_image(
    detections: Sequence[Union[GroundTruth, Prediction]],
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
        if detection.raster and detection.is_instance is True:
            img = _draw_detection_on_image(detection, img, inplace=i != 0)
    return img
