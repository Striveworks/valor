import math
from dataclasses import dataclass
from typing import List

from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont


@dataclass
class Image:
    uri: str


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
class DetectionBase:
    """Class representing a single object detection in an image."""

    boundary: BoundingPolygon
    labels: List[Label]
    image: Image

    def draw_on_image(
        self, img: PILImage.Image, inplace: bool = False
    ) -> PILImage.Image:
        img = self.boundary.draw_on_image(
            img, inplace=inplace, text=f"{self.class_label} {self.score:0.2f}"
        )
        return img


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


@dataclass
class GroundTruthSegmentation:
    shape: List[PolygonWithHole]
    labels: List[Label]
    image: Image


@dataclass
class PredictedSegmentation:
    shape: List[PolygonWithHole]
    scored_labels: List[ScoredLabel]
    image: Image


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
