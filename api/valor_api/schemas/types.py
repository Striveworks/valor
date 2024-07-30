import math
from typing import Any, Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    field_serializer,
    field_validator,
    model_validator,
)
from typing_extensions import Self

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


def _match_annotation_to_implied_task_type(
    annotation: "Annotation",
) -> list[str]:
    """
    Match an annotation to an implied task type based on the arguments that were passed to the Annotation constructor.

    Parameters
    ----------
    annotation: Annotation
        The annotation to validate.

    Raises
    ------
    ValueError
        If the contents of the annotation do not match an expected pattern.
    """
    implied_type = None
    # classification annotations have labels, but not anything else
    if (
        annotation.labels
        and annotation.bounding_box is None
        and annotation.polygon is None
        and annotation.raster is None
        and annotation.embedding is None
        and annotation.text is None
        and annotation.context_list is None
    ):
        implied_type = ["classification"]
    # object detection annotations have bounding boxes, polygons, and/or rasters
    elif (
        annotation.labels
        and (
            annotation.bounding_box is not None
            or annotation.polygon is not None
            or annotation.raster is not None
        )
        and annotation.is_instance is True
        and annotation.embedding is None
        and annotation.text is None
        and annotation.context_list is None
    ):
        implied_type = ["object-detection"]
    # semantic segmentation tasks only support rasters
    elif (
        annotation.labels
        and annotation.raster is not None
        and annotation.is_instance is not True
        and annotation.bounding_box is None
        and annotation.polygon is None
        and annotation.embedding is None
        and annotation.text is None
        and annotation.context_list is None
    ):
        implied_type = ["semantic-segmentation"]
    # embedding tasks only support enbeddings
    elif (
        annotation.embedding is not None
        and not annotation.labels
        and annotation.bounding_box is None
        and annotation.polygon is None
        and annotation.raster is None
        and annotation.text is None
        and annotation.context_list is None
    ):
        implied_type = ["embedding"]
    # text generation tasks only support text and optionally context_list
    elif (
        annotation.text is not None
        and not annotation.labels
        and annotation.bounding_box is None
        and annotation.polygon is None
        and annotation.raster is None
        and annotation.embedding is None
    ):
        implied_type = ["text-generation"]
    # empty annotations shouldn't contain anything
    elif (
        not annotation.labels
        and annotation.embedding is None
        and annotation.bounding_box is None
        and annotation.polygon is None
        and annotation.raster is None
        and annotation.text is None
        and annotation.context_list is None
    ):
        implied_type = ["empty"]
    else:
        raise ValueError(
            "Input didn't match any known patterns. Classification tasks should only contain labels. Object detection tasks should contain labels and polygons, bounding boxes, or rasters with is_instance == True. Segmentation tasks should contain labels and rasters with is_instance != True. Text generation tasks should only contain text and optionally context_list."
        )

    return implied_type


def _validate_groundtruth_annotations(annotations: list["Annotation"]) -> None:
    """
    Validate all of the annotations that are passed into a Groundtruth constructor.

    Parameters
    ----------
    annotations: list[Annotation]
        The annotations to validate.

    Raises
    ------
    ValueError
        If the contents of an annotation does not match expected patterns.
    """
    labels = []
    indices = dict()
    for index, annotation in enumerate(annotations):
        # handle type error
        if not isinstance(annotation.implied_task_types, list):
            raise ValueError("implied_task_types should be a list.")

        if "semantic-segmentation" in annotation.implied_task_types:
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
    Validate all of the annotations that are passed into a Prediction constructor.

    Parameters
    ----------
    annotations: list[Annotation]
        The annotations to validate.

    Raises
    ------
    ValueError
        If the contents of an annotation does not match expected patterns.
    """
    labels = []
    indices = dict()
    for index, annotation in enumerate(annotations):
        # handle type error
        if not isinstance(annotation.implied_task_types, list):
            raise ValueError("implied_task_types should be a list.")

        # Check that the label scores sum to 1.
        if "classification" in annotation.implied_task_types:
            label_keys_to_sum = {}
            for scored_label in annotation.labels:
                if scored_label.score is None:
                    raise ValueError(
                        "Prediction labels must have scores for classification tasks."
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
        elif "object-detection" in annotation.implied_task_types:
            # Check that we have scores for all the labels.
            for label in annotation.labels:
                if label.score is None:
                    raise ValueError(
                        "Prediction labels must have scores for object detection tasks."
                    )
        elif "semantic-segmentation" in annotation.implied_task_types:
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
    is_instance: bool, optional
        A boolean describing whether we should treat the Raster attached to an annotation as an instance segmentation or not. If set to true, then the Annotation will be validated for use in object detection tasks. If set to false, then the Annotation will be validated for use in semantic segmentation tasks.
    implied_task_types: list[str], optional
        The validated task types that are applicable to each Annotation. Doesn't need to be set by the user.
    text: str, optional
        A piece of text to assign to the 'Annotation'.
    context_list: list[str], optional
        A list of context to assign to the 'Annotation'.

    """

    metadata: MetadataType = dict()
    labels: list[Label] = list()
    bounding_box: Box | None = None
    polygon: Polygon | None = None
    raster: Raster | None = None
    embedding: list[float] | None = None
    is_instance: bool | None = None
    model_config = ConfigDict(extra="forbid")
    implied_task_types: list[str] | None = None
    text: str | None = None
    context_list: list[str] | None = None

    @field_validator("implied_task_types")
    @classmethod
    def _validate_implied_task_types(
        cls, implied_task_types: list[str]
    ) -> None:
        """Raise error if user tries to pass in an improper value into implied_task_types."""
        if implied_task_types and any(
            [
                x
                not in [
                    "classification",
                    "semantic-segmentation",
                    "object-detection",
                    "embedding",
                    "text-generation",
                    "empty",
                ]
                for x in implied_task_types
            ]
        ):
            raise ValueError(
                "Invalid value in implied_task_types. implied_task_types should not be set by the user; it will be determined automatically based on the user's supplied inputs to Annotation."
            )

    @model_validator(mode="after")
    def _set_implied_task_types(self) -> Self:
        """Set implied_task_types."""
        self.implied_task_types = _match_annotation_to_implied_task_type(self)
        return self

    @field_validator("is_instance")
    @classmethod
    def _validate_is_instance(
        cls, is_instance: bool | None, values: Any
    ) -> Optional[bool]:
        """Validates that is_instance was used correctly."""
        if is_instance is True and (
            values.data["raster"] is None
            and values.data["polygon"] is None
            and values.data["bounding_box"] is None
        ):
            raise ValueError(
                "is_instance=True currently only supports bounding_box, polygon and raster."
            )
        return is_instance

    @field_validator("metadata")
    @classmethod
    def validate_metadata_values(cls, v: dict) -> dict:
        """Validates the 'metadata' field."""
        validate_metadata(v)
        return v

    @field_serializer("bounding_box")
    @staticmethod
    def serialize_bounding_box(bounding_box: Box | None) -> Optional[dict]:
        """Serializes the 'bounding_box' attribute."""
        if bounding_box is None:
            return None
        return bounding_box.model_dump()["value"]

    @field_serializer("polygon")
    @staticmethod
    def serialize_polygon(polygon: Polygon | None) -> Optional[dict]:
        """Serializes the 'polygon' attribute."""
        if polygon is None:
            return None
        return polygon.model_dump()["value"]

    @field_serializer("raster")
    @staticmethod
    def serialize_raster(raster: Raster | None) -> Optional[dict]:
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
    text : str, optional
        If the datum is a piece of text, then this field should contain the text.
    metadata : dict, optional
        A dictionary of metadata that describes the datum.
    """

    uid: str
    text: str | None = None
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
            v = [Annotation()]
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
            v = [Annotation()]
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
