import io
from base64 import b64decode
from typing import Optional
from uuid import uuid4

import PIL.Image
from pydantic import BaseModel, Extra, Field, root_validator, validator

from velour_api.enums import JobStatus, Task


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
    uid: str
    height: int
    width: int
    frame: Optional[int] = None


class Label(BaseModel):
    key: str
    value: str

    @classmethod
    def from_key_value_tuple(cls, kv_tuple: tuple[str, str]):
        return cls(key=kv_tuple[0], value=kv_tuple[1])


class ScoredLabel(BaseModel):
    label: Label
    score: float


class DetectionBase(BaseModel):
    # list of (x, y) points
    boundary: list[tuple[float, float]] = None
    # (xmin, ymin, xmax, ymax)
    bbox: tuple[float, float, float, float] = None
    image: Image

    @root_validator(skip_on_failure=True)
    def boundary_or_bbox(cls, values):
        if (values["boundary"] is None) == (values["bbox"] is None):
            raise ValueError("Must have exactly one of boundary or bbox")

        return values

    @validator("boundary")
    def enough_pts(cls, v):
        if v is not None:
            return validate_single_polygon(v)
        return v

    @property
    def is_bbox(self):
        return self.bbox is not None


class GroundTruthDetection(DetectionBase):
    labels: list[Label]


class PredictedDetection(DetectionBase):
    scored_labels: list[ScoredLabel]


class PredictedDetectionsCreate(BaseModel):
    model_name: str
    dataset_name: str
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
    dataset_name: str
    classifications: list[PredictedImageClassification]


class PolygonWithHole(BaseModel):
    polygon: list[tuple[float, float]]
    hole: list[tuple[float, float]] = None

    @validator("polygon")
    def enough_pts_outer(cls, v):
        return validate_single_polygon(v)


def _mask_bytes_to_pil(mask_bytes: bytes) -> PIL.Image.Image:
    with io.BytesIO(mask_bytes) as f:
        return PIL.Image.open(f)


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

    @root_validator
    def correct_mask_shape(cls, values):
        if isinstance(values["shape"], list):
            return values
        mask_size = _mask_bytes_to_pil(b64decode(values["shape"])).size
        image_size = (values["image"].width, values["image"].height)
        if mask_size != image_size:
            raise ValueError(
                f"Expected mask and image to have the same size, but got size {mask_size} for the mask and {image_size} for image."
            )
        return values

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

    @property
    def pil_mask(self) -> PIL.Image:
        return _mask_bytes_to_pil(self.mask_bytes)


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
    dataset_name: str
    segmentations: list[PredictedSegmentation]


class User(BaseModel):
    email: str = None


class MetricParameters(BaseModel):
    """General parameters defining any filters of the data such
    as model, dataset, groundtruth and prediction type, model, dataset,
    size constraints, coincidence/intersection constraints, etc.
    """

    model_name: str
    dataset_name: str
    model_pred_task_type: Task
    dataset_gt_task_type: Task
    # TODO: add things here for filtering, prediction
    # and dataset label mappings (e.g. man, boy -> person)


class APRequest(BaseModel):
    """Request to compute average precision"""

    parameters: MetricParameters
    labels: list[Label] = None
    # (mutable defaults are ok for pydantic models)
    iou_thresholds: list[float] = [round(0.5 + 0.05 * i, 2) for i in range(10)]


class APMetric(BaseModel):
    iou: float
    value: float
    label: Label


class mAPMetric(BaseModel):
    iou: float
    value: float
    labels: list[Label]


class MetricResponse(BaseModel):
    """Used for REST responses sending a metric"""

    metric_name: str
    parameters: MetricParameters
    metric: APMetric | mAPMetric


class CreateMetricsResponse(BaseModel):
    missing_pred_labels: list[Label]
    ignored_pred_labels: list[Label]
    job_id: str


class EvalJob(BaseModel):
    uid: str = Field(default_factory=lambda: str(uuid4()))
    status: JobStatus = JobStatus.PENDING
    created_metrics_ids: list[int] = None
