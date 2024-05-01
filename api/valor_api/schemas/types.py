import math
from typing import Any, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    field_serializer,
    field_validator,
    model_validator,
)

from valor_api.enums import TaskType
from valor_api.schemas.geometry import Box, Polygon, Raster
from valor_api.schemas.validators import (
    validate_metadata,
    validate_type_string,
)

GeometryType = Union[
    tuple[float, float],
    list[tuple[float, float]],
    list[list[tuple[float, float]]],
    list[list[list[tuple[float, float]]]],
]
GeoJSONType = dict[str, str | GeometryType]
DateTimeType = str | float
MetadataType = dict[
    str, bool | int | float | str | dict[str, DateTimeType | GeoJSONType]
]


def _check_if_empty_annotation(annotation: "Annotation") -> bool:
    """
    Checks if the annotation is empty.

    Parameters
    ----------
    annotation : Annotation
        The annotation to check.

    Returns
    -------
    bool
        Whether the annotation is empty.
    """
    return (
        not annotation.labels
        and annotation.bounding_box is None
        and annotation.polygon is None
        and annotation.raster is None
        and annotation.embedding is None
    )


def _validate_annotation_by_task_type(
    annotation: "Annotation",
) -> "Annotation":
    """
    Validates the contents of an annotation by task type.

    Parameters
    ----------
    annotation: Annotation
        The annotation to validate.

    Raises
    ------
    ValueError
        If the contents of the annotation do not match the task type.
    NotImplementedError
        If the task type is not recognized.
    """
    if _check_if_empty_annotation(annotation):
        if annotation.task_type != TaskType.SKIP:
            annotation.task_type = TaskType.EMPTY
    match annotation.task_type:
        case TaskType.CLASSIFICATION:
            if not (
                annotation.labels
                and annotation.bounding_box is None
                and annotation.polygon is None
                and annotation.raster is None
                and annotation.embedding is None
            ):
                raise ValueError(
                    "Annotations with task type `classification` do not support geometries or embeddings."
                )
        case TaskType.OBJECT_DETECTION:
            if not (
                annotation.labels
                and (
                    annotation.bounding_box is not None
                    or annotation.polygon is not None
                    or annotation.raster is not None
                )
                and annotation.embedding is None
            ):
                raise ValueError(
                    "Annotations with task type `object-detection` do not support embeddings."
                )
        case TaskType.SEMANTIC_SEGMENTATION:
            if not (
                annotation.labels
                and annotation.raster is not None
                and annotation.bounding_box is None
                and annotation.polygon is None
                and annotation.embedding is None
            ):
                raise ValueError(
                    "Annotations with task type `semantic-segmentation` only supports rasters."
                )
        case TaskType.EMBEDDING:
            if not (
                annotation.embedding is not None
                and not annotation.labels
                and annotation.bounding_box is None
                and annotation.polygon is None
                and annotation.raster is None
            ):
                raise ValueError(
                    "Annotations with task type `embedding` do not support labels or geometries."
                )
        case TaskType.RANKING:
            if not (
                annotation.labels
                and annotation.ranking is not None
                and (
                    annotation.bounding_box is None
                    and annotation.polygon is None
                    and annotation.embedding is None
                    and annotation.raster is None
                    and annotation.embedding is None
                )
            ):
                raise ValueError(
                    "Annotations with task type `ranking` should only contain labels and rankings."
                )
        case TaskType.EMPTY | TaskType.SKIP:
            if not _check_if_empty_annotation(annotation):
                raise ValueError("Annotation is not empty.")
        case _:
            raise NotImplementedError(
                f"Task type `{annotation.task_type}` is not supported."
            )
    return annotation


def _validate_groundtruth_annotations(annotations: list["Annotation"]) -> None:
    """
    Check that a label appears once in the annotations for semenatic segmentations.

    Parameters
    ----------
    annotations: list[Annotation]
        The annotations to validate.

    Raises
    ------
    ValueError
        If the contents of an annotation does not match the task type.
    """
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


def _validate_prediction_annotations(annotations: list["Annotation"]) -> None:
    """
    Validate prediction annotations by task type.

    Parameters
    ----------
    annotations: list[Annotation]
        The annotations to validate.

    Raises
    ------
    ValueError
        If the contents of an annotation does not match the task type.
    """
    labels = []
    indices = dict()
    for index, annotation in enumerate(annotations):
        if annotation.task_type == TaskType.CLASSIFICATION:
            # Check that the label scores sum to 1.
            label_keys_to_sum = {}
            for scored_label in annotation.labels:
                if scored_label.score is None:
                    raise ValueError(
                        f"Missing score for label in {annotation.task_type} task."
                    )
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

        elif annotation.task_type == TaskType.OBJECT_DETECTION:
            # Check that we have scores for all the labels.
            for label in annotation.labels:
                if label.score is None:
                    raise ValueError(
                        f"Missing score for label in {annotation.task_type} task."
                    )

        elif annotation.task_type == TaskType.SEMANTIC_SEGMENTATION:
            for label in annotation.labels:
                # Check that score is not defined.
                if label.score is not None:
                    raise ValueError(
                        "Semantic segmentation tasks cannot have scores; only metrics with "
                        "hard predictions are supported."
                    )
                # Check that a label appears once in the annotations.
                if label in labels:
                    raise ValueError(
                        f"Label {label} appears in both annotation {index} and {indices[label]}, but semantic segmentation "
                        "tasks can only have one annotation per label."
                    )
                labels.append(label)
                indices[label] = index


class Label(BaseModel):
    """
    An object for labeling datasets, models, and annotations.

    Attributes
    ----------
    key : str
        The label key. (e.g. 'class', 'category')
    value : str
        The label's value. (e.g. 'dog', 'cat')
    score : float, optional
        A score assigned to the label in the case of a prediction.
    """

    key: str
    value: str
    score: float | None = None
    model_config = ConfigDict(extra="forbid")

    def __eq__(self, other):
        """
        Defines how labels are compared to one another.

        Parameters
        ----------
        other : Label
            The object to compare with the label.

        Returns
        ----------
        bool
            A boolean describing whether the two objects are equal.
        """
        if (
            not hasattr(other, "key")
            or not hasattr(other, "key")
            or not hasattr(other, "score")
        ):
            return False

        # if the scores aren't the same type return False
        if (other.score is None) != (self.score is None):
            return False

        if self.score is None or other.score is None:
            scores_equal = other.score is None and self.score is None
        else:
            scores_equal = math.isclose(self.score, other.score)

        return (
            scores_equal
            and self.key == other.key
            and self.value == other.value
        )

    def __hash__(self) -> int:
        """
        Defines how a 'Label' is hashed.

        Returns
        ----------
        int
            The hashed 'Label'.
        """
        return hash(f"key:{self.key},value:{self.value},score:{self.score}")


class Annotation(BaseModel):
    """
    A class used to annotate 'GroundTruths' and 'Predictions'.

    Attributes
    ----------
    task_type: TaskType
        The task type associated with the 'Annotation'.
    metadata: dict, optional
        A dictionary of metadata that describes the 'Annotation'.
    labels: List[Label], optional
        A list of labels to use for the 'Annotation'.
    bounding_box: BoundingBox, optional
        A bounding box to assign to the 'Annotation'.
    polygon: Polygon, optional
        A polygon to assign to the 'Annotation'.
    raster: Raster, optional
        A raster to assign to the 'Annotation'.
    embedding: list[float], optional
        A jsonb to assign to the 'Annotation'.
    ranking: list[str], optional
        A list of strings or a list of floats representing an ordered ranking.

    """

    task_type: TaskType
    metadata: MetadataType = dict()
    labels: list[Label] = list()
    bounding_box: Box | None = None
    polygon: Polygon | None = None
    raster: Raster | None = None
    embedding: list[float] | None = None
    ranking: list[str] | None = None
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    @classmethod
    def validate_by_task_type(cls, values: Any) -> Any:
        """Validates the annotation by task type."""
        return _validate_annotation_by_task_type(values)

    @field_validator("metadata")
    @classmethod
    def validate_metadata_values(cls, v: dict) -> dict:
        """Validates the 'metadata' field."""
        validate_metadata(v)
        return v

    @field_serializer("bounding_box")
    def serialize_bounding_box(bounding_box: Box | None) -> dict | None:  # type: ignore - pydantic field_serializer
        """Serializes the 'bounding_box' attribute."""
        if bounding_box is None:
            return None
        return bounding_box.model_dump()["value"]

    @field_serializer("polygon")
    def serialize_polygon(polygon: Polygon | None) -> dict | None:  # type: ignore - pydantic field_serializer
        """Serializes the 'polygon' attribute."""
        if polygon is None:
            return None
        return polygon.model_dump()["value"]

    @field_serializer("raster")
    def serialize_raster(raster: Raster | None) -> dict | None:  # type: ignore - pydantic field_serializer
        """Serializes the 'raster' attribute."""
        if raster is None:
            return None
        return raster.model_dump()


class Datum(BaseModel):
    """
    A class used to store datum information about 'GroundTruths' and 'Predictions'.

    Attributes
    ----------
    uid : str
        The UID of the datum.
    metadata : dict, optional
        A dictionary of metadata that describes the datum.
    """

    uid: str
    metadata: MetadataType = dict()
    model_config = ConfigDict(extra="forbid")

    @field_validator("uid")
    @classmethod
    def validate_uid(cls, v: str) -> str:
        """Validates the 'uid' field."""
        validate_type_string(v)
        return v

    @field_validator("metadata")
    @classmethod
    def validate_metadata_values(cls, v: dict) -> dict:
        """Validates the 'metadata' field."""
        validate_metadata(v)
        return v


class GroundTruth(BaseModel):
    """
    An object describing a ground truth (e.g., a human-drawn bounding box on an image).

    Attributes
    ----------
    dataset_name: str
        The name of the dataset this ground truth belongs to.
    datum : Datum
        The datum this ground truth annotates.
    annotations : List[Annotation]
        The list of annotations that this ground truth applies.
    """

    dataset_name: str
    datum: Datum
    annotations: list[Annotation]
    model_config = ConfigDict(extra="forbid")

    @field_validator("dataset_name")
    @classmethod
    def validate_dataset_name(cls, v: str) -> str:
        """Validates the 'dataset_name' field."""
        validate_type_string(v)
        return v

    @field_validator("annotations")
    @classmethod
    def validate_annotations(cls, v: list[Annotation]) -> list[Annotation]:
        """Validates the 'annotations' attribute."""
        if not v:
            v = [Annotation(task_type=TaskType.EMPTY)]
        _validate_groundtruth_annotations(v)
        return v


class Prediction(BaseModel):
    """
    An object describing a prediction (e.g., a machine-drawn bounding box on an image).

    Attributes
    ----------
    dataset_name: str
        The name of the dataset this ground truth belongs to.
    model_name : str
        The name of the model that produced the prediction.
    datum : Datum
        The datum this ground truth annotates.
    annotations : List[Annotation]
        The list of annotations that this ground truth applies.
    """

    dataset_name: str
    model_name: str
    datum: Datum
    annotations: list[Annotation]
    model_config = ConfigDict(extra="forbid")

    @field_validator("dataset_name")
    @classmethod
    def validate_dataset_name(cls, v: str) -> str:
        """Validates the 'dataset_name' attribute."""
        validate_type_string(v)
        return v

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validates the 'model_name' attribute."""
        validate_type_string(v)
        return v

    @field_validator("annotations")
    @classmethod
    def validate_annotations(cls, v: list[Annotation]) -> list[Annotation]:
        """Validates the 'annotations' attribute."""
        if not v:
            v = [Annotation(task_type=TaskType.EMPTY)]
        _validate_prediction_annotations(v)
        return v


class Dataset(BaseModel):
    """
    A class describing a given dataset.

    Attributes
    ----------
    name : str
        The name of the dataset.
    metadata : dict, optional
        A dictionary of metadata that describes the dataset.
    """

    name: str
    metadata: MetadataType = dict()
    model_config = ConfigDict(extra="forbid")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validates the 'name' field."""
        validate_type_string(v)
        return v

    @field_validator("metadata")
    @classmethod
    def validate_metadata_values(cls, v: dict) -> dict:
        """Validates the 'metadata' field."""
        validate_metadata(v)
        return v


class Model(BaseModel):
    """
    A class describing a model that was trained on a particular dataset.

    Attributes
    ----------
    name : str
        The name of the model.
    metadata : dict, optional
        A dictionary of metadata that describes the model.
    """

    name: str
    metadata: MetadataType = dict()
    model_config = ConfigDict(extra="forbid")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validates the 'name' field."""
        validate_type_string(v)
        return v

    @field_validator("metadata")
    @classmethod
    def validate_metadata_values(cls, v: dict) -> dict:
        """Validates the 'metadata' field."""
        validate_metadata(v)
        return v
