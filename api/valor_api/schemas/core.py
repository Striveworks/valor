import re

from pydantic import BaseModel, ConfigDict, field_validator

from valor_api.enums import TaskType
from valor_api.schemas.geometry import (
    BoundingBox,
    MultiPolygon,
    Polygon,
    Raster,
)
from valor_api.schemas.label import Label
from valor_api.schemas.metadata import DateTimeType, GeoJSONType

MetadataType = dict[
    str, float | str | bool | dict[str, DateTimeType | GeoJSONType]
]


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
    metadata :  MetadataType
        A dictionary of metadata that describes the dataset.

    Raises
    ----------
    ValueError
        If the name is invalid.
    """

    id: int | None = None
    name: str
    metadata: MetadataType = {}
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
    metadata :  MetadataType
        A dictionary of metadata that describes the model.

    Raises
    ----------
    ValueError
        If the name is invalid.
    """

    id: int | None = None
    name: str
    metadata: MetadataType = {}
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
    dataset_name : str
        The name of the dataset to associate the `Datum` with.
    metadata : MetadataType
        A dictionary of metadata that describes the `Datum`.

    Raises
    ----------
    ValueError
        If the dataset or UID is invalid.
    """

    uid: str
    dataset_name: str
    metadata: MetadataType = {}
    model_config = ConfigDict(extra="forbid")

    @field_validator("uid")
    @classmethod
    def _check_uid_valid(cls, v):
        """Validate the UID field."""

        if not v:
            raise ValueError("invalid string")

        _validate_uid_format(v)
        return v

    @field_validator("dataset_name")
    @classmethod
    def _check_name_valid(cls, v):
        """Validate the dataset field."""

        if not v:
            raise ValueError("invalid string")

        _validate_name_format(v)
        return v

    def __eq__(self, other):
        """
        Defines how `Datums` are compared to one another.

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
            and self.dataset_name == other.dataset_name
            and self.metadata == other.metadata
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
    metadata: MetadataType
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
    metadata: MetadataType = {}

    # Geometric types
    bounding_box: BoundingBox | None = None
    polygon: Polygon | None = None
    multipolygon: MultiPolygon | None = None
    raster: Raster | None = None

    model_config = ConfigDict(use_enum_values=True, extra="forbid")


def _check_semantic_segmentations_single_label(
    annotations: list[Annotation],
) -> None:
    """Check that a label appears once in the annotations for semenatic segmentations"""
    labels = []
    indices = dict()
    for index, annotation in enumerate(annotations):
        if annotation.task_type == TaskType.SEMANTIC_SEGMENTATION:
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
    An object describing a ground truth (e.g., a human-drawn bounding box on an image).

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


class Prediction(BaseModel):
    """
    An object describing a prediction (e.g., a machine-drawn bounding box on an image).

    Attributes
    ----------
    model_name : str
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

    model_name: str
    datum: Datum
    annotations: list[Annotation]
    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=("protected_",),
    )

    @field_validator("model_name")
    @classmethod
    def _check_name_valid(cls, v):
        """Validate the model field."""

        if not v:
            raise ValueError("invalid string")
        _validate_name_format(v)
        return v

    @field_validator("annotations")
    @classmethod
    def _validate_annotation_scores(cls, v: list[Annotation]):
        """Check that we have scores for all the labels if the task type requires it."""
        for annotation in v:
            if annotation.task_type in [
                TaskType.CLASSIFICATION,
                TaskType.OBJECT_DETECTION,
            ]:
                for label in annotation.labels:
                    if label.score is None:
                        raise ValueError(
                            f"Missing score for label in {annotation.task_type} task."
                        )
            elif annotation.task_type == TaskType.SEMANTIC_SEGMENTATION:
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
