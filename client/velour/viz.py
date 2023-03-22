from typing import Dict, List, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw

from velour.data_types import (
    PolygonWithHole,
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
    label_key: str = None,
) -> Tuple[Image.Image, Dict[str, Image.Image]]:

    if len(set([seg.image.uid for seg in segs])) > 1:
        raise RuntimeError(
            "Expected all segmentation to belong to the same image"
        )

    if len(segs) == 0:
        raise ValueError("`segs` cannot be empty.")

    label_values = []
    for seg in segs:
        if label_key is None:
            if len(seg.labels) > 0:
                raise RuntimeError(
                    "Found segmentation with multiple labels. Pass the desired "
                    "label key to the `label_key` parameter."
                )
            label_values.append(seg.labels[0].value)
        else:
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
