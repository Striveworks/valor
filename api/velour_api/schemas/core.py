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
from velour_api.schemas.geojson import (
    GeoJSONMultiPolygon,
    GeoJSONPoint,
    GeoJSONPolygon,
)
from velour_api.schemas.geometry import (
    BoundingBox,
    MultiPolygon,
    Polygon,
    Raster,
)
from velour_api.schemas.label import Label


def _validate_name_format(name: str):
    """Validate that a name doesn't contain any forbidden characters"""
    allowed_special = ["-", "_"]
    pattern = re.compile(f"^[a-zA-Z0-9{''.join(allowed_special)}]+$")
    if not pattern.match(name):
        raise ValueError(
            "The provided string contains illegal characters. Please ensure your input consists of only alphanumeric characters, hyphens, and underscores."
        )
    return name


def _validate_uid_format(uid: str):
    """Validate that a UID doesn't contain any forbidden characters."""
    allowed_special = ["-", "_", "/", "."]
    pattern = re.compile(f"^[a-zA-Z0-9{''.join(allowed_special)}]+$")
    if not pattern.match(uid):
        raise ValueError(
            "The provided string contains illegal characters. Please ensure your input consists of only alphanumeric characters, hyphens, underscores, forward slashes, and periods."
        )
    return uid


class Dataset(BaseModel):
    """
    A class describing a given dataset.

    Attributes
    ----------
    id : int
        The ID of the dataset.
    name : str
        The name of the dataset.
    metadata :  dict
        A dictionary of metadata that describes the dataset.
    geospatial : dict
        A GeoJSON-style dictionary describing the geospatial coordinates of the dataset.

    Raises
    ----------
    ValueError
        If the name is invalid.
    """

    id: int | None = None
    name: str
    metadata: dict[str, float | str | dict[str, str]] = {}
    geospatial: GeoJSONPoint | GeoJSONPolygon | GeoJSONMultiPolygon | None = (
        None
    )
    model_config = ConfigDict(extra="forbid")

    @field_validator("name")
    @classmethod
    def _check_name_valid(cls, v):
        """Validate the name field."""
        if not v:
            raise ValueError("invalid string")

        _validate_name_format(v)
        return v


class Model(BaseModel):
    """
    A class describing a model that was trained on a particular dataset.

    Attributes
    ----------
    id : int
        The ID of the model.
    name : str
        The name of the model.
    metadata :  dict
        A dictionary of metadata that describes the model.
    geospatial : dict
        A GeoJSON-style dictionary describing the geospatial metadata of the model.

    Raises
    ----------
    ValueError
        If the name is invalid.
    """

    id: int | None = None
    name: str
    metadata: dict[str, float | str | dict[str, str]] = {}
    geospatial: GeoJSONPoint | GeoJSONPolygon | GeoJSONMultiPolygon | None = (
        None
    )
    model_config = ConfigDict(extra="forbid")

    @field_validator("name")
    @classmethod
    def _check_name_valid(cls, v):
        """Validate the name field."""
        if not v:
            raise ValueError("invalid string")

        _validate_name_format(v)
        return v


class Datum(BaseModel):
    """
    A class used to store datum about `GroundTruths` and `Predictions`.

    Attributes
    ----------
    uid : str
        The UID of the `Datum`.
    metadata : dict
        A dictionary of metadata that describes the `Datum`.
    geospatial :  dict
        A GeoJSON-style dictionary describing the geospatial coordinates of the `Datum`.
    dataset : str
        The name of the dataset to associate the `Datum` with.

    Raises
    ----------
    ValueError
        If the dataset or UID is invalid.
    """

    uid: str
    dataset: str
    metadata: dict[str, float | str | dict[str, str]] = {}
    geospatial: GeoJSONPoint | GeoJSONPolygon | GeoJSONMultiPolygon | None = (
        None
    )
    model_config = ConfigDict(extra="forbid")

    @field_validator("uid")
    @classmethod
    def _check_uid_valid(cls, v):
        """Validate the UID field."""

        if not v:
            raise ValueError("invalid string")

        _validate_uid_format(v)
        return v

    @field_validator("dataset")
    @classmethod
    def _check_name_valid(cls, v):
        """Validate the dataset field."""

        if not v:
            raise ValueError("invalid string")

        _validate_name_format(v)
        return v

    def __eq__(self, other):
        """
        Defines how `Datums` are compared to one another

        Parameters
        ----------
        other : Datum
            The object to compare with the `Datum`.

        Returns
        ----------
        boolean
            A boolean describing whether the two objects are equal.
        """
        if not isinstance(other, Datum):
            raise TypeError
        return (
            self.uid == other.uid
            and self.dataset == other.dataset
            and self.metadata == other.metadata
            and self.geospatial == other.geospatial
        )


class Annotation(BaseModel):
    """
    A class used to annotate `GroundTruths` and `Predictions`.

    Attributes
    ----------
    task_type: TaskType
        The task type associated with the `Annotation`.
    labels: List[Label]
        A list of labels to use for the `Annotation`.
    metadata: Dict[str, Union[int, float, str]]
        A dictionary of metadata that describes the `Annotation`.
    bounding_box: BoundingBox
        A bounding box to assign to the `Annotation`.
    polygon: Polygon
        A polygon to assign to the `Annotation`.
    multipolygon: MultiPolygon
        A multipolygon to assign to the `Annotation`.
    raster: Raster
        A raster to assign to the `Annotation`.
    jsonb: Dict
        A jsonb to assign to the `Annotation`.

    Raises
    ----------
    ValueError
        If no labels are passed.
        If the same label appears in two annotations.
    """

    task_type: TaskType
    labels: list[Label]
    metadata: dict[str, float | str | dict[str, str]] = Field(
        default_factory=dict
    )

    # Geometric types
    bounding_box: BoundingBox | None = None
    polygon: Polygon | None = None
    multipolygon: MultiPolygon | None = None
    raster: Raster | None = None
    jsonb: dict[str, str] | None = None

    model_config = ConfigDict(use_enum_values=True, extra="forbid")

    @field_validator("labels")
    @classmethod
    def _check_labels_not_empty(cls, v):
        """Validate that labels aren't empty."""
        if not v:
            raise ValueError("`labels` cannot be empty.")
        return v


def _check_semantic_segmentations_single_label(
    annotations: list[Annotation],
) -> None:
    """Check that a label appears once in the annotations for semenatic segmentations"""
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
    """
    An object describing a groundtruth (e.g., a human-drawn bounding box on an image).

    Attributes
    ----------
    datum : Datum
        The `Datum` associated with the `GroundTruth`.
    annotations : List[Annotation]
        The list of `Annotations` associated with the `GroundTruth`.

    Raises
    ----------
    ValueError
        If the same label appears in two annotations.
        If any rasters don't match their metadata.
    """

    datum: Datum
    annotations: list[Annotation]
    model_config = ConfigDict(extra="forbid")

    @field_validator("annotations")
    @classmethod
    def _check_semantic_segmentation_annotations(cls, v: list[Annotation]):
        """Validate that only one label exists."""
        _check_semantic_segmentations_single_label(v)
        return v

    @model_validator(mode="after")
    @classmethod
    def _validate_annotation_rasters(cls, values):
        """Validate any rasters on the groundtruth."""
        return _validate_rasters(values)


class Prediction(BaseModel):
    """
    An object describing a prediction (e.g., a machine-drawn bounding box on an image).

    Attributes
    ----------
    model : str
        The name of the model that produced the `Prediction`.
    datum : Datum
        The `Datum` associated with the `Prediction`.
    annotations : List[Annotation]
        The list of `Annotations` associated with the `Prediction`.

    Raises
    ----------
    ValueError
        If the model name is invalid.
        If no annotations are passed.
        If the same label appears in two annotations.
        If we're missing scores for any label.
        If semantic segmentations contain a score.
        If label scores for any key sum to more than 1.
    """

    model: str
    datum: Datum
    annotations: list[Annotation]
    model_config = ConfigDict(extra="forbid")

    @field_validator("model")
    @classmethod
    def _check_name_valid(cls, v):
        """Validate the model field."""

        if not v:
            raise ValueError("invalid string")
        _validate_name_format(v)
        return v

    @model_validator(mode="after")
    @classmethod
    def _validate_annotation_rasters(cls, values):
        """Validate any rasters on the annotation."""

        return _validate_rasters(values)

    @field_validator("annotations")
    @classmethod
    def _validate_annotation_scores(cls, v: list[Annotation]):
        """Check that we have scores for all the labels if the task type requires it."""
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
    def _check_label_scores(cls, v: list[Annotation]):
        """Check that for classification tasks, the label scores sum to 1."""
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
    def _check_semantic_segmentation_annotations(cls, v: list[Annotation]):
        """Validate that a label doesn't appear more than once."""
        _check_semantic_segmentations_single_label(v)
        return v


def _mask_bytes_to_pil(mask_bytes):
    """Convert a byte mask to a PIL.Image."""
    with io.BytesIO(mask_bytes) as f:
        return PIL.Image.open(f)


def _validate_rasters(annotated_datum: GroundTruth | Prediction):
    """Validate that the Annotation metadata matches what's described in a raster."""
    for annotation in annotated_datum.annotations:
        if annotation.raster is not None:
            # unpack datum metadata
            metadata = annotated_datum.datum.metadata
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
    return annotated_datum
