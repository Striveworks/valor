import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont


def rle_to_mask(
    run_length_encoding: List[Tuple[int, int]],
    image_height: int,
    image_width: int,
) -> np.ndarray:
    res = np.zeros((image_height, image_width), dtype=bool)
    idx = 0
    for start, length in run_length_encoding:
        idx += start
        for i in range(idx, idx + length):
            y, x = divmod(i, image_height)
            res[x, y] = True
        idx += length
    return res


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

    return rle_to_mask(
        run_length_encoding=run_length_encoding, image_height=h, image_width=w
    )


@dataclass
class Image:
    uri: str
    height: int
    width: int


@dataclass
class Label:
    key: str
    value: str


@dataclass
class ScoredLabel:
    label: Label
    score: float


@dataclass
class Point:
    x: float
    y: float

    def resize(
        self, og_img_h: int, og_img_w: int, new_img_h: int, new_img_w: int
    ) -> "Point":
        h_factor, w_factor = new_img_h / og_img_h, new_img_w / og_img_w
        return Point(x=w_factor * self.x, y=h_factor * self.y)


@dataclass
class BoundingPolygon:
    """Class for representing a bounding region."""

    points: List[Point]

    def draw_on_image(
        self,
        img: PILImage.Image,
        inplace: bool = False,
        text: str = None,
        font_size: int = 24,
    ) -> PILImage.Image:
        color = (255, 0, 0)
        img = img if inplace else img.copy()
        img_draw = ImageDraw.Draw(img)

        img_draw.polygon(
            [(p.x, p.y) for p in self.points],
            outline=color,
        )

        if text is not None:
            _write_text(
                font_size=font_size,
                text=text,
                boundary=self,
                draw=img_draw,
                color=color,
            )
        return img

    @property
    def xmin(self):
        return min(p.x for p in self.points)

    @property
    def ymin(self):
        return min(p.y for p in self.points)

    @property
    def xmax(self):
        return max(p.x for p in self.points)

    @property
    def ymax(self):
        return max(p.y for p in self.points)


@dataclass
class GroundTruthDetection:
    boundary: BoundingPolygon
    labels: List[Label]
    image: Image

    def draw_on_image(
        self, img: PILImage.Image, inplace: bool = False
    ) -> PILImage.Image:
        text = ", ".join(
            [f"{label.key}:{label.value}" for label in self.labels]
        )
        img = self.boundary.draw_on_image(img, inplace=inplace, text=text)
        return img


@dataclass
class PredictedDetection:
    boundary: BoundingPolygon
    scored_labels: List[ScoredLabel]
    image: Image

    def draw_on_image(
        self, img: PILImage.Image, inplace: bool = False
    ) -> PILImage.Image:
        text = ", ".join(
            [
                f"{scored_label.label.key}:{scored_label.label.value} ({scored_label.score})"
                for scored_label in self.scored_labels
            ]
        )
        img = self.boundary.draw_on_image(img, inplace=inplace, text=text)
        return img


@dataclass
class PolygonWithHole:
    polygon: BoundingPolygon
    hole: BoundingPolygon = None


def _validate_mask(mask: np.ndarray):
    if mask.dtype != bool:
        raise ValueError(
            f"Expecting a binary mask (i.e. of dtype bool) but got dtype {mask.dtype}"
        )


@dataclass
class _GroundTruthSegmentation:
    shape: Union[List[PolygonWithHole], np.ndarray]
    labels: List[Label]
    image: Image

    def __post_init__(self):
        if isinstance(self.shape, np.ndarray):
            _validate_mask(self.shape)


GroundTruthInstanceSegmentation = _GroundTruthSegmentation
GroundTruthSemanticSegmentation = _GroundTruthSegmentation


@dataclass
class PredictedSegmentation:
    mask: np.ndarray
    scored_labels: List[ScoredLabel]
    image: Image

    def __post_init__(self):
        _validate_mask(self.mask)


@dataclass
class GroundTruthImageClassification:
    image: Image
    labels: List[Label]


@dataclass
class PredictedImageClassification:
    image: Image
    scored_labels: List[ScoredLabel]


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

    text_width, text_height = font.getsize(text)
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
