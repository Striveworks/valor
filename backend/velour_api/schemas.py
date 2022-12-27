from pydantic import BaseModel, validator


class DetectionBase(BaseModel):
    # should enforce beginning and ending points are the same? or no?
    boundary: list[tuple[int, int]]
    class_label: str

    @validator("boundary")
    def enough_pts(cls, v):
        if len(v) < 3:
            raise ValueError(
                "Boundary must be composed of at least three points."
            )
        return v


class PredictedDetectionBase(DetectionBase):
    score: float


class GroundTruthDetectionCreate(DetectionBase):
    pass


class PredictedDetectionCreate(PredictedDetectionBase):
    pass


class GroundTruthDetection(DetectionBase):
    id: int


class PredictedDetection(PredictedDetectionBase):
    id: int
