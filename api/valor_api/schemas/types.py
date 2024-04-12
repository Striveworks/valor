import math
from typing import Union

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
    validate_annotation_by_task_type,
    validate_groundtruth_annotations,
    validate_metadata,
    validate_prediction_annotations,
    validate_type_string,
)

GeometryType = Union[
    tuple[float, float],
    list[tuple[float, float]],
    list[list[tuple[float, float]]],
    list[list[list[tuple[float, float]]]],
]
GeoJSONType = dict[str, str | GeometryType]
MetadataType = dict[str, dict[str, bool | int | float | str | GeoJSONType]]


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
    """

    task_type: TaskType
    metadata: MetadataType = dict()
    labels: list[Label] = list()
    bounding_box: Box | None = None
    polygon: Polygon | None = None
    raster: Raster | None = None
    embedding: list[float] | None = None
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    @classmethod
    def validate_by_task_type(cls, values):
        """Validates the annotation by task type."""
        return validate_annotation_by_task_type(values)

    @field_validator("metadata")
    @classmethod
    def validate_metadata_values(cls, v: dict) -> dict:
        """Validates the 'metadata' field."""
        validate_metadata(v)
        return v

    @field_serializer("bounding_box")
    def serialize_bounding_box(bounding_box: Box | None):  # type: ignore - pydantic field_serializer
        """Serializes the 'bounding_box' attribute."""
        if bounding_box is None:
            return None
        return bounding_box.model_dump()["value"]

    @field_serializer("polygon")
    def serialize_polygon(polygon: Polygon | None):  # type: ignore - pydantic field_serializer
        """Serializes the 'polygon' attribute."""
        if polygon is None:
            return None
        return polygon.model_dump()["value"]

    @field_serializer("raster")
    def serialize_raster(raster: Raster | None):  # type: ignore - pydantic field_serializer
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
    def validate_uid(cls, v):
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
    def validate_dataset_name(cls, v):
        """Validates the 'dataset_name' field."""
        validate_type_string(v)
        return v

    @field_validator("annotations")
    @classmethod
    def validate_annotations(cls, v: list[Annotation]):
        """Validates the 'annotations' attribute."""
        if not v:
            v = [Annotation(task_type=TaskType.EMPTY)]
        validate_groundtruth_annotations(v)
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
    def validate_dataset_name(cls, v):
        """Validates the 'dataset_name' attribute."""
        validate_type_string(v)
        return v

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v):
        """Validates the 'model_name' attribute."""
        validate_type_string(v)
        return v

    @field_validator("annotations")
    @classmethod
    def validate_annotations(cls, v: list[Annotation]):
        """Validates the 'annotations' attribute."""
        if not v:
            v = [Annotation(task_type=TaskType.EMPTY)]
        validate_prediction_annotations(v)
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
    def validate_name(cls, v):
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
    def validate_name(cls, v):
        """Validates the 'name' field."""
        validate_type_string(v)
        return v

    @field_validator("metadata")
    @classmethod
    def validate_metadata_values(cls, v: dict) -> dict:
        """Validates the 'metadata' field."""
        validate_metadata(v)
        return v
