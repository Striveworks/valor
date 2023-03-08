import io
from base64 import b64decode

import PIL.Image
from pydantic import BaseModel, Extra, Field, validator
from typing import Optional


def validate_single_polygon(poly: list[tuple[float, float]]):
    if len(poly) < 3:
        raise ValueError("Polygon must be composed of at least three points.")
    return poly


class Dataset(BaseModel):
    name: str
    draft: bool
    from_video: bool = False


class DatasetCreate(BaseModel):
    name: str


class Model(BaseModel):
    name: str


class Image(BaseModel):
    uri: str
    height: int
    width: int
    frame: Optional[int] = None
    timestamp: Optional[float] = None


class Label(BaseModel):
    key: str
    value: str


class ScoredLabel(Label):
    score: float


class DetectionBase(BaseModel):
    # should enforce beginning and ending points are the same? or no?
    boundary: list[tuple[float, float]]
    image: Image

    @validator("boundary")
    def enough_pts(cls, v):
        return validate_single_polygon(v)


class GroundTruthDetection(DetectionBase):
    labels: list[Label]


class PredictedDetection(DetectionBase):
    scored_labels: list[ScoredLabel]


class PredictedDetectionsCreate(BaseModel):
    model_name: str
    detections: list[PredictedDetection]


class GroundTruthDetectionsCreate(BaseModel):
    dataset_name: str
    detections: list[GroundTruthDetection]


class ImageClassificationBase(BaseModel):
    image: Image
    labels: list[Label]


class PredictedImageClassification(BaseModel):
    image: Image
    scored_labels: list[ScoredLabel]


class GroundTruthImageClassificationsCreate(BaseModel):
    dataset_name: str
    classifications: list[ImageClassificationBase]


class PredictedImageClassificationsCreate(BaseModel):
    model_name: str
    classifications: list[PredictedImageClassification]


class PolygonWithHole(BaseModel):
    polygon: list[tuple[float, float]]
    hole: list[tuple[float, float]] = None

    @validator("polygon")
    def enough_pts_outer(cls, v):
        return validate_single_polygon(v)


class GroundTruthSegmentation(BaseModel):
    # multipolygon or base64 mask
    shape: str | list[PolygonWithHole] = Field(allow_mutation=False)
    image: Image
    labels: list[Label]
    is_instance: bool

    class Config:
        extra = Extra.allow
        validate_assignment = True

    @validator("shape")
    def non_empty(cls, v):
        if len(v) == 0:
            raise ValueError("shape must have at least one element.")
        return v

    @property
    def is_poly(self) -> bool:
        return isinstance(self.shape, list)

    @property
    def mask_bytes(self) -> bytes:
        if self.is_poly:
            raise RuntimeError(
                "`mask_bytes` can only be called for `GroundTruthSegmentation`'s defined"
                " by masks, not polygons"
            )
        if not hasattr(self, "_mask_bytes"):
            self._mask_bytes = b64decode(self.shape)
        return self._mask_bytes


class PredictedSegmentation(BaseModel):
    base64_mask: str = Field(allow_mutation=False)
    image: Image
    scored_labels: list[ScoredLabel]
    is_instance: bool

    class Config:
        extra = Extra.allow
        validate_assignment = True

    @property
    def mask_bytes(self) -> bytes:
        if not hasattr(self, "_mask_bytes"):
            self._mask_bytes = b64decode(self.base64_mask)
        return self._mask_bytes

    @validator("base64_mask")
    def check_png_and_mode(cls, v):
        """Check that the bytes are for a png file and is binary"""
        f = io.BytesIO(b64decode(v))
        img = PIL.Image.open(f)
        f.close()
        if img.format != "PNG":
            raise ValueError(
                f"Expected image format PNG but got {img.format}."
            )
        if img.mode != "1":
            raise ValueError(
                f"Expected image mode to be binary but got mode {img.mode}."
            )
        return v


class GroundTruthSegmentationsCreate(BaseModel):
    dataset_name: str
    segmentations: list[GroundTruthSegmentation]


class PredictedSegmentationsCreate(BaseModel):
    model_name: str
    segmentations: list[PredictedSegmentation]


class User(BaseModel):
    email: str = None
