from pydantic import BaseModel, Extra, validator


# extra=Extra.allow allows us to attach additional attributes to these objects later
class Image(BaseModel, extra=Extra.allow):
    uri: str


class Label(BaseModel, extra=Extra.allow):
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


class PredictedDetectionCreate(DetectionBase):
    score: float


class GroundTruthDetectionCreate(DetectionBase):
    pass
