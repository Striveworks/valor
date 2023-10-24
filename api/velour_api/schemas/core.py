import io
import re
from base64 import b64decode

import PIL.Image
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from velour_api.enums import TaskType
from velour_api.schemas.geometry import (
    BoundingBox,
    MultiPolygon,
    Polygon,
    Raster,
)
from velour_api.schemas.label import Label
from velour_api.schemas.metadata import Metadatum


def _format_name(name: str):
    allowed_special = ["-", "_"]
    pattern = re.compile(f"[^a-zA-Z0-9{''.join(allowed_special)}]")
    return re.sub(pattern, "", name)


def _format_uid(uid: str):
    allowed_special = ["-", "_", "/", "."]
    pattern = re.compile(f"[^a-zA-Z0-9{''.join(allowed_special)}]")
    return re.sub(pattern, "", uid)


class Dataset(BaseModel):
    id: int | None = None
    name: str
    metadata: list[Metadatum] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def check_name_valid(cls, v):
        if v != _format_name(v):
            raise ValueError("name includes illegal characters.")
        if not v:
            raise ValueError("invalid string")
        return v


class Model(BaseModel):
    id: int | None = None
    name: str
    metadata: list[Metadatum] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def check_name_valid(cls, v):
        if v != _format_name(v):
            raise ValueError("name includes illegal characters.")
        if not v:
            raise ValueError("invalid string")
        return v


class Datum(BaseModel):
    uid: str
    dataset: str
    metadata: list[Metadatum] = Field(default_factory=list)

    @field_validator("uid")
    @classmethod
    def format_uid(cls, v):
        if v != _format_uid(v):
            raise ValueError("uid includes illegal characters.")
        if not v:
            raise ValueError("invalid string")
        return v

    @field_validator("dataset")
    @classmethod
    def check_name_valid(cls, v):
        if v != _format_name(v):
            raise ValueError("name includes illegal characters.")
        if not v:
            raise ValueError("invalid string")
        return v

    def __eq__(self, other):
        if (
            not hasattr(other, "uid")
            or not hasattr(other, "dataset")
            or not hasattr(other, "metadata")
        ):
            return False

        return self.uid == other.uid and self.dataset == other.dataset

    def __hash__(self) -> int:
        return hash(f"uid:{self.uid},dataset:{self.dataset}")


class Annotation(BaseModel):
    task_type: TaskType
    labels: list[Label]
    metadata: list[Metadatum] = Field(default_factory=list)

    # Geometric types
    bounding_box: BoundingBox | None = None
    polygon: Polygon | None = None
    multipolygon: MultiPolygon | None = None
    raster: Raster | None = None
    jsonb: dict[str, str] | None = None

    model_config = ConfigDict(use_enum_values=True)

    @field_validator("labels")
    @classmethod
    def check_labels_not_empty(cls, v):
        if not v:
            raise ValueError("`labels` cannot be empty.")
        return v


def _check_semantic_segmentations_single_label(
    annotations: list[Annotation],
) -> None:
    # check that a label on appears once in the annotations for semenatic segmentations
    labels = []
    indices = dict()
    for index, annotation in enumerate(annotations):
        if annotation.task_type == TaskType.SEGMENTATION:
            for label in annotation.labels:
                if label in labels:
                    raise ValueError(
                        f"Label {label} appears in both annotation {index} and {indices[label]}, but semantic segmentation "
                        "tasks can only have one annotation per label."
                    )
                labels.append(label)
                indices[label] = index


class GroundTruth(BaseModel):
    datum: Datum
    annotations: list[Annotation]

    @field_validator("annotations")
    @classmethod
    def check_annotations_not_empty(cls, v: list[Annotation]):
        if not v:
            raise ValueError("annotations is empty")
        return v

    @field_validator("annotations")
    @classmethod
    def check_semantic_segmentation_annotations(cls, v: list[Annotation]):
        # make sure a label doesn't appear more than once
        _check_semantic_segmentations_single_label(v)
        return v

    @model_validator(mode="after")
    @classmethod
    def validate_annotation_rasters(cls, values):
        return _validate_rasters(values)


class Prediction(BaseModel):
    model: str
    datum: Datum
    annotations: list[Annotation]

    @field_validator("model")
    @classmethod
    def check_name_valid(cls, v):
        if v != _format_name(v):
            raise ValueError("name includes illegal characters.")
        if not v:
            raise ValueError("invalid string")
        return v

    @field_validator("annotations")
    @classmethod
    def check_annotations(cls, v):
        if not v:
            raise ValueError("annotations is empty")
        return v

    @model_validator(mode="after")
    @classmethod
    def validate_annotation_rasters(cls, values):
        return _validate_rasters(values)

    @field_validator("annotations")
    @classmethod
    def validate_annotation_scores(cls, v: list[Annotation]):
        # check that we have scores for all the labels if
        # the task type requires it
        for annotation in v:
            if annotation.task_type in [
                TaskType.CLASSIFICATION,
                TaskType.DETECTION,
            ]:
                for label in annotation.labels:
                    if label.score is None:
                        raise ValueError(
                            f"Missing score for label in {annotation.task_type} task."
                        )
            elif annotation.task_type == TaskType.SEGMENTATION:
                for label in annotation.labels:
                    if label.score is not None:
                        raise ValueError(
                            "Semantic segmentation tasks cannot have scores; only metrics with "
                            "hard predictions are supported."
                        )

        return v

    @field_validator("annotations")
    @classmethod
    def check_label_scores(cls, v: list[Annotation]):
        # check that for classification tasks, the label scores
        # sum to 1
        for annotation in v:
            if annotation.task_type == TaskType.CLASSIFICATION:
                label_keys_to_sum = {}
                for scored_label in annotation.labels:
                    label_key = scored_label.key
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

    @field_validator("annotations")
    @classmethod
    def check_semantic_segmentation_annotations(cls, v: list[Annotation]):
        # make sure a label doesn't appear more than once
        _check_semantic_segmentations_single_label(v)
        return v


def _validate_rasters(d: GroundTruth | Prediction):
    def _mask_bytes_to_pil(mask_bytes):
        with io.BytesIO(mask_bytes) as f:
            return PIL.Image.open(f)

    for annotation in d.annotations:
        if annotation.raster is not None:
            # unpack datum metadata
            metadata = {
                metadatum.key: metadatum.value
                for metadatum in filter(
                    lambda x: x.key in ["height", "width"], d.datum.metadata
                )
            }
            if "height" not in metadata or "width" not in metadata:
                raise RuntimeError(
                    "Attempted raster validation but image dimensions are missing."
                )

            # validate raster wrt datum metadata
            mask_size = _mask_bytes_to_pil(
                b64decode(annotation.raster.mask)
            ).size
            image_size = (metadata["width"], metadata["height"])
            if mask_size != image_size:
                raise ValueError(
                    f"Expected raster and image to have the same size, but got size {mask_size} for the mask and {image_size} for image."
                )
    return d
