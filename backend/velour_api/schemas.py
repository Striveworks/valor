from pydantic import BaseModel, validator


def validate_single_polygon(poly: list[tuple[float, float]]):
    if len(poly) < 3:
        raise ValueError("Polygon must be composed of at least three points.")
    return poly


class Dataset(BaseModel):
    name: str
    draft: bool


class DatasetCreate(BaseModel):
    name: str


class Model(BaseModel):
    name: str


class Image(BaseModel):
    uri: str


class Label(BaseModel):
    key: str
    value: str


class ScoredLabel(BaseModel):
    label: Label
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


class PolygonWithHoles(BaseModel):
    outer: list[tuple[float, float]]
    inner: list[tuple[float, float]]

    @validator("outer")
    def enough_pts_outer(cls, v):
        return validate_single_polygon(v)

    @validator("inner")
    def enough_pts_inner(cls, v):
        return validate_single_polygon(v)


class SegmentationBase(BaseModel):
    shape: list[PolygonWithHoles]
    image: Image

    @validator("shape")
    def non_empty(cls, v):
        if len(v) == 0:
            raise ValueError("shape must have at least one element.")
        return v


class GroundTruthSegmentation(SegmentationBase):
    labels: list[Label]


class PredictedSegmentation(SegmentationBase):
    scored_labels: list[ScoredLabel]


class GroundTruthSegmentationsCreate(BaseModel):
    dataset_name: str
    segmentations: list[GroundTruthSegmentation]


class PredictedSegmentationsCreate(BaseModel):
    model_name: str
    segmentations: list[PredictedSegmentation]
