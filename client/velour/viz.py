import math
from typing import Dict, List, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from velour.data_types import (
    BoundingPolygon,
    GroundTruthDetection,
    PolygonWithHole,
    PredictedDetection,
    _GroundTruthSegmentation,
    _PredictedSegmentation,
)

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
    polys: List[PolygonWithHole], img_w, img_h
) -> np.ndarray:
    """note there's some aliasing/areas differences between this
    and e.g. postgis, so this method should only be used for visualization
    """
    mask = Image.new("1", (img_w, img_h), (False,))
    draw = ImageDraw.Draw(mask)
    for poly in polys:
        draw.polygon(poly.polygon.xy_list(), fill=(True,))
        if poly.hole is not None:
            draw.polygon(poly.hole.xy_list(), fill=(False,))

    return np.array(mask)


def combined_segmentation_mask(
    segs: List[Union[_GroundTruthSegmentation, _PredictedSegmentation]],
    label_key: str,
) -> Tuple[Image.Image, Dict[str, Image.Image]]:
    """Creates a combined segmentation mask from a list of segmentations

    segs
        list of segmentations. these all must have the same `image` attribute. this
        must be non-empty
    label_key
        the label key to use.


    Returns
    -------
    tuple
        the first element of the tuple is the combined mask, as an RGB PIL image. the second
        element is a color legend: it's a dict with keys the unique values of the labels and the
        values of the dict are PIL image swatches of the color corresponding to the label value.

    Raises
    ------
    RuntimeError
        if all segmentations don't belong to the same image, or there is a
        segmentation that doesn't have `label_key` as the key of one of its labels
    ValueError
        if segs is empty
    """

    if len(set([seg.image.uid for seg in segs])) > 1:
        raise RuntimeError(
            "Expected all segmentation to belong to the same image"
        )

    if len(segs) == 0:
        raise ValueError("`segs` cannot be empty.")

    label_values = []
    for seg in segs:
        found_label = False
        for label in seg.labels:
            if label.key == label_key:
                found_label = True
                label_values.append(label.value)
        if not found_label:
            raise RuntimeError(
                "Found a segmentation that doesn't have a label with key 'label_key'."
                f" Available label keys are: {[label.key for label in seg.labels]}"
            )

    unique_label_values = list(set(label_values))
    label_value_to_color = {
        v: COLOR_MAP[i] for i, v in enumerate(unique_label_values)
    }
    seg_colors = [label_value_to_color[v] for v in label_values]

    img_w, img_h = segs[0].image.width, segs[0].image.height

    combined_mask = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    for seg, color in zip(segs, seg_colors):
        if isinstance(seg.shape, np.ndarray):
            mask = seg.shape
        else:
            mask = _polygons_to_binary_mask(
                seg.shape, img_w=seg.image.width, img_h=seg.image.height
            )

        combined_mask[np.where(mask)] = color

    legend = {
        v: Image.new("RGB", (20, 20), color)
        for v, color in label_value_to_color.items()
    }

    return Image.fromarray(combined_mask), legend


def draw_detections_on_image(
    detections: List[Union[GroundTruthDetection, PredictedDetection]],
    img: Image.Image,
) -> Image.Image:
    """Draws detections (bounding boxes and labels) on an image"""
    for i, detection in enumerate(detections):
        img = _draw_detection_on_image(detection, img, inplace=i != 0)
    return img


def _draw_detection_on_image(
    detection: Union[GroundTruthDetection, PredictedDetection],
    img: Image.Image,
    inplace: bool,
) -> Image.Image:
    text = ", ".join(
        [f"{label.key}:{label.value}" for label in detection.labels]
    )
    img = _draw_bounding_polygon_on_image(
        detection.boundary,
        img,
        inplace=inplace,
        text=text,
    )

    return img


def _draw_bounding_polygon_on_image(
    polygon: BoundingPolygon,
    img: Image.Image,
    color: Tuple[int, int, int] = (255, 0, 0),
    inplace: bool = False,
    text: str = None,
    font_size: int = 24,
) -> Image.Image:
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
    boundary: BoundingPolygon,
    draw: ImageDraw.Draw,
    color: str,
):
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
