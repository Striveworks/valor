import math
from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

import numpy as np
import PIL.Image
from PIL import ImageDraw, ImageFont


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


@dataclass
class Image:
    uid: str
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
        img: PIL.Image.Image,
        inplace: bool = False,
        text: str = None,
        font_size: int = 24,
    ) -> PIL.Image.Image:
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
        self, img: PIL.Image.Image, inplace: bool = False
    ) -> PIL.Image.Image:
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
        self, img: PIL.Image.Image, inplace: bool = False
    ) -> PIL.Image.Image:
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
class _GroundTruthSegmentation(ABC):
    shape: Union[List[PolygonWithHole], np.ndarray]
    labels: List[Label]
    image: Image
    _is_instance: bool

    def __post_init__(self):
        if self.__class__ == _GroundTruthSegmentation:
            raise TypeError("Cannot instantiate abstract class.")
        if isinstance(self.shape, np.ndarray):
            _validate_mask(self.shape)

    def draw_on_img(self, img: PIL.Image):
        assert isinstance(self.shape, np.ndarray)

        mask = np.zeros(
            (self.shape.shape[0], self.shape.shape[1], 3), dtype=np.uint8
        )
        mask[np.where(self.shape == 1)] = [255, 0, 0]
        blend = PIL.Image.blend(img, PIL.Image.fromarray(mask), alpha=0.4)

        img = img.convert("RGB")
        img.paste(
            blend,
            (0, 0),
            PIL.Image.fromarray(255 * self.shape.astype(np.uint8)),
        )
        return img


@dataclass
class GroundTruthInstanceSegmentation(_GroundTruthSegmentation):
    _is_instance: bool = field(default=True, init=False)


@dataclass
class GroundTruthSemanticSegmentation(_GroundTruthSegmentation):
    _is_instance: bool = field(default=False, init=False)


@dataclass
class _PredictedSegmentation(ABC):
    mask: np.ndarray
    scored_labels: List[ScoredLabel]
    image: Image
    _is_instance: bool

    def __post_init__(self):
        if self.__class__ == _GroundTruthSegmentation:
            raise TypeError("Cannot instantiate abstract class.")
        _validate_mask(self.mask)


@dataclass
class PredictedInstanceSegmentation(_PredictedSegmentation):
    _is_instance: bool = field(default=True, init=False)


@dataclass
class PredictedSemanticSegmentation(_PredictedSegmentation):
    _is_instance: bool = field(default=False, init=False)


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
