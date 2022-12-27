from pydantic import BaseModel


class DetectionBase(BaseModel):
    # should enforce beginning and ending points are the same? or no?
    boundary: list[tuple[int, int]]
    class_label: str


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
