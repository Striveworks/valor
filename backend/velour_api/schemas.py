import io
import json
from base64 import b64decode
from typing import Optional
from uuid import uuid4

import numpy as np
import PIL.Image
from pydantic import BaseModel, Extra, Field, root_validator, validator

from velour_api.enums import DatumTypes, JobStatus, Task


def validate_single_polygon(poly: list[tuple[float, float]]):
    if len(poly) < 3:
        raise ValueError("Polygon must be composed of at least three points.")
    return poly


def _validate_href(v: str | None):
    if v is None:
        return v
    if not (v.startswith("http://") or v.startswith("https://")):
        raise ValueError("`href` must start with http:// or https://")
    return v


class _BaseDataset(BaseModel):
    name: str
    from_video: bool = False
    href: str = None
    description: str = None
    type: DatumTypes

    @validator("href")
    def validate_href(cls, v):
        return _validate_href(v)


class Dataset(_BaseDataset):
    draft: bool


class DatasetCreate(_BaseDataset):
    pass


class Model(BaseModel):
    name: str
    href: str = None
    description: str = None
    type: DatumTypes

    @validator("href")
    def validate_href(cls, v):
        return _validate_href(v)


class DatumMetadatum(BaseModel):
    name: str
    value: float | str | dict

    @validator("value")
    def check_json(cls, v):
        # TODO: add more validation that the dict is valid geoJSON?
        if isinstance(v, dict):
            json.dumps(v)
        return v


class Datum(BaseModel):
    uid: str
    metadata: list[DatumMetadatum] = []


class Image(Datum):
    height: int = None
    width: int = None
    frame: Optional[int] = None


class Label(BaseModel):
    key: str
    value: str

    @classmethod
    def from_key_value_tuple(cls, kv_tuple: tuple[str, str]):
        return cls(key=kv_tuple[0], value=kv_tuple[1])

    def __eq__(self, other):
        if hasattr(other, "key") and hasattr(other, "value"):
            return self.key == other.key and self.value == other.value
        return False

    def __hash__(self) -> int:
        return hash(f"key:{self.key},value:{self.value}")


class ScoredLabel(BaseModel):
    label: Label
    score: float


class LabelDistribution(BaseModel):
    label: Label
    count: int


class ScoredLabelDistribution(BaseModel):
    label: Label
    scores: list[float]
    count: int


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


class GroundTruthClassification(BaseModel):
    datum: Datum
    labels: list[Label]


class PredictedClassification(BaseModel):
    datum: Datum
    scored_labels: list[ScoredLabel]

    @validator("scored_labels")
    def check_sum_to_one(cls, v: list[ScoredLabel]):
        label_keys_to_sum = {}
        for scored_label in v:
            label_key = scored_label.label.key
            if label_key not in label_keys_to_sum:
                label_keys_to_sum[label_key] = 0.0
            label_keys_to_sum[label_key] += scored_label.score

        for k, total_score in label_keys_to_sum.items():
            if abs(total_score - 1) > 1e-5:
                raise ValueError(
                    "For each label key, prediction scores must sum to 1, but"
                    f" for label key {k} got scores summing to {total_score}."
                )
        return v


class GroundTruthClassificationsCreate(BaseModel):
    dataset_name: str
    classifications: list[GroundTruthClassification]


class PredictedClassificationsCreate(BaseModel):
    model_name: str
    dataset_name: str
    classifications: list[PredictedClassification]


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


class _PredictedSegmentation(BaseModel):
    base64_mask: str = Field(allow_mutation=False)
    image: Image

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


class PredictedInstanceSegmentation(_PredictedSegmentation):
    scored_labels: list[ScoredLabel]


class PredictedSemanticSegmentation(_PredictedSegmentation):
    labels: list[Label]


class GroundTruthSegmentationsCreate(BaseModel):
    dataset_name: str
    segmentations: list[GroundTruthSegmentation]


class PredictedSegmentationsCreate(BaseModel):
    model_name: str
    dataset_name: str
    segmentations: list[
        PredictedInstanceSegmentation | PredictedSemanticSegmentation
    ]


class User(BaseModel):
    email: str = None


class EvaluationSettings(BaseModel):
    """General parameters defining any filters of the data such
    as model, dataset, groundtruth and prediction type, model, dataset,
    size constraints, coincidence/intersection constraints, etc.
    """

    model_name: str
    dataset_name: str
    model_pred_task_type: Task = None
    dataset_gt_task_type: Task = None
    min_area: float = None
    max_area: float = None
    group_by: str = None
    label_key: str = None
    id: int = None


class APRequest(BaseModel):
    """Request to compute average precision"""

    settings: EvaluationSettings
    # (mutable defaults are ok for pydantic models)
    iou_thresholds: list[float] = [round(0.5 + 0.05 * i, 2) for i in range(10)]
    ious_to_keep: set[float] = {0.5, 0.75}

    @root_validator
    def check_ious(cls, values):
        for iou in values["ious_to_keep"]:
            if iou not in values["iou_thresholds"]:
                raise ValueError(
                    "`ious_to_keep` must be contained in `iou_thresholds`"
                )
        return values


class CreateAPMetricsResponse(BaseModel):
    missing_pred_labels: list[Label]
    ignored_pred_labels: list[Label]
    job_id: str


class CreateClfMetricsResponse(BaseModel):
    missing_pred_keys: list[str]
    ignored_pred_keys: list[str]
    job_id: str


class Job(BaseModel):
    uid: str = Field(default_factory=lambda: str(uuid4()))
    status: JobStatus = JobStatus.PENDING

    class Config:
        extra = Extra.allow


class ClfMetricsRequest(BaseModel):
    settings: EvaluationSettings


class Metric(BaseModel):
    """This is used for responses from the API"""

    type: str
    parameters: dict | None
    value: float | dict | None
    label: Label = None
    group: DatumMetadatum = None


class APMetric(BaseModel):
    iou: float
    value: float
    label: Label

    def db_mapping(self, label_id: int, evaluation_settings_id: int) -> dict:
        return {
            "value": self.value,
            "label_id": label_id,
            "type": "AP",
            "evaluation_settings_id": evaluation_settings_id,
            "parameters": {"iou": self.iou},
        }


class APMetricAveragedOverIOUs(BaseModel):
    ious: set[float]
    value: float
    label: Label

    def db_mapping(self, label_id: int, evaluation_settings_id: int) -> dict:
        return {
            "value": self.value,
            "label_id": label_id,
            "type": "APAveragedOverIOUs",
            "evaluation_settings_id": evaluation_settings_id,
            "parameters": {"ious": list(self.ious)},
        }


class mAPMetric(BaseModel):
    iou: float
    value: float

    def db_mapping(self, evaluation_settings_id: int) -> dict:
        return {
            "value": self.value,
            "type": "mAP",
            "evaluation_settings_id": evaluation_settings_id,
            "parameters": {"iou": self.iou},
        }


class mAPMetricAveragedOverIOUs(BaseModel):
    ious: set[float]
    value: float

    def db_mapping(self, evaluation_settings_id: int) -> dict:
        return {
            "value": self.value,
            "type": "mAPAveragedOverIOUs",
            "evaluation_settings_id": evaluation_settings_id,
            "parameters": {"ious": list(self.ious)},
        }


class ConfusionMatrixEntry(BaseModel):
    prediction: str
    groundtruth: str
    count: int

    class Config:
        allow_mutation = False


class _BaseConfusionMatrix(BaseModel):
    label_key: str
    entries: list[ConfusionMatrixEntry]
    group: DatumMetadatum = None
    group_id: int = None


class ConfusionMatrix(_BaseConfusionMatrix, extra=Extra.allow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        label_values = set(
            [entry.prediction for entry in self.entries]
            + [entry.groundtruth for entry in self.entries]
        )
        self.label_map = {
            label_value: i
            for i, label_value in enumerate(sorted(label_values))
        }
        n_label_values = len(self.label_map)

        matrix = np.zeros((n_label_values, n_label_values), dtype=int)
        for entry in self.entries:
            matrix[
                self.label_map[entry.groundtruth],
                self.label_map[entry.prediction],
            ] = entry.count

        self.matrix = matrix

    def db_mapping(self, evaluation_settings_id: int) -> dict:
        return {
            "label_key": self.label_key,
            "value": [entry.dict() for entry in self.entries],
            "evaluation_settings_id": evaluation_settings_id,
        }


class ConfusionMatrixResponse(_BaseConfusionMatrix):
    """used for http response since it won't have the matrix and
    label map attributes
    """

    pass


class AccuracyMetric(BaseModel):
    label_key: str
    value: float
    group: DatumMetadatum = None
    group_id: int = None

    def db_mapping(self, evaluation_settings_id: int) -> dict:
        return {
            "value": self.value,
            "type": "Accuracy",
            "evaluation_settings_id": evaluation_settings_id,
            "parameters": {"label_key": self.label_key},
            "group_id": self.group_id,
        }


class _PrecisionRecallF1Base(BaseModel):
    label: Label
    value: float | None
    group: DatumMetadatum = None
    group_id: int = None

    @validator("value")
    def replace_nan_with_neg_1(cls, v):
        if np.isnan(v):
            return -1
        return v

    def db_mapping(self, label_id: int, evaluation_settings_id: int) -> dict:
        return {
            "value": self.value,
            "label_id": label_id,
            "type": self.__type__,
            "evaluation_settings_id": evaluation_settings_id,
            "group_id": self.group_id,
        }


class PrecisionMetric(_PrecisionRecallF1Base):
    __type__ = "Precision"


class RecallMetric(_PrecisionRecallF1Base):
    __type__ = "Recall"


class F1Metric(_PrecisionRecallF1Base):
    __type__ = "F1"


class ROCAUCMetric(BaseModel):
    label_key: str
    value: float
    group: DatumMetadatum = None
    group_id: int = None

    def db_mapping(self, evaluation_settings_id: int) -> dict:
        return {
            "value": self.value,
            "type": "ROCAUC",
            "parameters": {"label_key": self.label_key},
            "evaluation_settings_id": evaluation_settings_id,
            "group_id": self.group_id,
        }


class Info(BaseModel):
    annotation_type: list[str]
    number_of_classifications: int
    number_of_bounding_boxes: int
    number_of_bounding_polygons: int
    number_of_segmentation_rasters: int
    associated: list[str]
