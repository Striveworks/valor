from pydantic import BaseModel, validator


class Dataset(BaseModel):
    name: str
    draft: bool


class DatasetCreate(BaseModel):
    name: str


class ModelCreate(BaseModel):
    name: str


class Image(BaseModel):
    uri: str


class Label(BaseModel):
    key: str
    value: str


class DetectionBase(BaseModel):
    # should enforce beginning and ending points are the same? or no?
    boundary: list[tuple[float, float]]
    labels: list[Label]
    image: Image

    @validator("boundary")
    def enough_pts(cls, v):
        if len(v) < 3:
            raise ValueError(
                "Boundary must be composed of at least three points."
            )
        return v


class PredictedDetection(DetectionBase):
    score: float


class PredictedDetectionsCreate(BaseModel):
    model_name: str
    detections: list[PredictedDetection]


class GroundTruthDetectionsCreate(BaseModel):
    dataset_name: str
    detections: list[DetectionBase]
