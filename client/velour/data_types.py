import math
from dataclasses import dataclass
from typing import List

from PIL import Image, ImageDraw, ImageFont

MaskType = List[List[float]]
ImageInput = List[Image.Image]


@dataclass
class Point:
    x: int
    y: int

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
        img: Image.Image,
        inplace: bool = False,
        text: str = None,
        font_size: int = 24,
    ) -> Image.Image:
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
class DetectionBase:
    """Class representing a single object detection in an image."""

    boundary: BoundingPolygon
    class_label: str

    def draw_on_image(self, img: Image.Image, inplace: bool = False) -> Image.Image:
        img = self.boundary.draw_on_image(
            img, inplace=inplace, text=f"{self.class_label} {self.score:0.2f}"
        )
        return img


@dataclass
class GroundTruthDetection(DetectionBase):
    pass


@dataclass
class PredictedDetection(DetectionBase):
    score: float

    def __post_init__(self):
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be between 0.0 and 1.0 but got {self.score}.")


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
