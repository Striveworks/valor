import math

from dataclasses import dataclass, fields, is_dataclass
from typing import Optional, List, Tuple, Union, Any

from velour.types import MetadataType, GeoJSONType
from velour.enums import TaskType
from velour.schemas.constraints import (
    _DeclarativeMapper,
    NumericMapper, 
    StringMapper, 
    GeometryMapper, 
    GeospatialMapper, 
    DictionaryMapper,
    LabelMapper,
)
from velour.schemas.geometry import BoundingBox, Polygon, MultiPolygon, Raster
from velour.schemas.metadata import validate_metadata, load_metadata


def _reset_mapped_fields(obj):
    """Sets values that inherit from _DeclarativeMapper to `None`."""
    if not is_dataclass:
        raise TypeError
    for field in fields(obj):
        obj_type = type(getattr(obj, field.name))
        if issubclass(obj_type, _DeclarativeMapper):
            if field.name == "metadata":
                setattr(obj, field.name, {})
            else:
                setattr(obj, field.name, None)
            



@dataclass
class Label:
    """
    An object for labeling datasets, models, and annotations.

    Attributes
    ----------
    key : str
        The class key of the label.
    value : str
        The class value of the label.
    score : float, optional
        The score associated with the label (if applicable).
    """
    value: str
    key: Union[str, StringMapper] = StringMapper("label_keys")
    score: Union[Optional[float], NumericMapper] = NumericMapper("prediction_scores")

    def __post_init__(self):
        _reset_mapped_fields(self)

        if not isinstance(self.key, str):
            raise TypeError(f"Attribute `key` should have type `str`.")
        
        if not isinstance(self.value, str):
            raise TypeError(f"Attribute `value` should have type `str`.")

        if (
            not isinstance(self.score, float)
            and self.score is not None
        ):
            raise TypeError(f"Attribute `score` should be of type `float` or `None`.")

    def __str__(self):
        return str(self.tuple())

    def __eq__(self, other):
        """
        Defines how `Labels` are compared to one another

        Parameters
        ----------
        other : Label
            The object to compare with the `Label`.

        Returns
        ----------
        boolean
            A boolean describing whether the two objects are equal.
        """
        if type(other) is not type(self):
            return False

        # if the scores aren't the same type return False
        if (other.score is None) != (self.score is None):
            return False

        scores_equal = (other.score is None and self.score is None) or (
            math.isclose(self.score, other.score)
        )

        return (
            scores_equal
            and self.key == other.key
            and self.value == other.value
        )

    def __hash__(self) -> int:
        """
        Defines how a `Label` is hashed.

        Returns
        ----------
        int
            The hashed 'Label`.
        """
        return hash(f"key:{self.key},value:{self.value},score:{self.score}")
    
    def tuple(self) -> Tuple[str, str, Optional[float]]:
        """
        Defines how the `Label` is turned into a tuple.

        Returns
        ----------
        tuple
            A tuple of the `Label's` arguments.
        """
        return (self.key, self.value, self.score)


@dataclass
class Annotation:
    task_type: Union[TaskType, StringMapper] = StringMapper(name="task_types")
    labels: Union[List[Label], LabelMapper] = LabelMapper(name="labels")
    metadata: Union[MetadataType, DictionaryMapper] = DictionaryMapper("annotation_metadata")
    bounding_box: Union[Optional[BoundingBox], GeometryMapper] = GeometryMapper("annotation_bounding_box")
    polygon: Union[Optional[Polygon], GeometryMapper] = GeometryMapper("annotation_polygon")
    multipolygon: Union[Optional[MultiPolygon], GeometryMapper] = GeometryMapper("annotation_multipolygon")
    raster: Union[Optional[Raster], GeometryMapper] = GeometryMapper("annotation_raster")

    def __post_init__(self):
        _reset_mapped_fields(self)

        # task_type
        if not isinstance(self.task_type, TaskType):
            self.task_type = TaskType(self.task_type)

        # labels
        if not isinstance(self.labels, list):
            raise TypeError("Attribute `labels` should be of type `List[Label]`.")
        for idx, label in enumerate(self.labels):
            if isinstance(self.labels[idx], dict):
                self.labels[idx] = Label(**label)
            if not isinstance(self.labels[idx], Label):
                raise TypeError("Attribute `labels` should be of type `List[Label]`.")

        # bounding box
        if self.bounding_box:
            if isinstance(self.bounding_box, dict):
                self.bounding_box = BoundingBox(**self.bounding_box)
            if not isinstance(self.bounding_box, BoundingBox):
                raise TypeError("Attribute `bounding_box` should be of type `BoundingBox`.")
        
        # polygon
        if self.polygon:
            if isinstance(self.polygon, dict):
                self.polygon = Polygon(**self.polygon)
            if not isinstance(self.polygon, Polygon):
                raise TypeError("Attribute `polygon` should be of type `Polygon`.")
            
        # multipolygon
        if self.multipolygon:
            if isinstance(self.multipolygon, dict):
                self.multipolygon = MultiPolygon(**self.multipolygon)
            if not isinstance(self.multipolygon, MultiPolygon):
                raise TypeError("Attribute `multipolygon` should be of type `MultiPolygon`.")
            
        # raster
        if self.raster:
            if isinstance(self.raster, dict):
                self.raster = Raster(**self.raster)
            if not isinstance(self.raster, Raster):
                raise TypeError("Attribute `raster` should be of type `Raster`.")

        # metadata
        if not isinstance(self.metadata, dict):
            raise TypeError("Attribute `metadata` should be of type `dict`.")
        validate_metadata(self.metadata)
        self.metadata = load_metadata(self.metadata)


@dataclass
class Datum:
    uid : Union[str, StringMapper] = StringMapper(name="datum_uids")
    metadata : Union[MetadataType, DictionaryMapper] = DictionaryMapper(name="datum_metadata")
    geospatial : Union[Optional[GeoJSONType], GeospatialMapper] = GeospatialMapper(name="datum_geospatial")
    dataset_name: Optional[str] = None

    def __post_init__(self):
        _reset_mapped_fields(self)

        validate_metadata(self.metadata)
        self.metadata = load_metadata(self.metadata)


@dataclass
class GroundTruth:
    datum: Datum
    annotations: List[Annotation]

    def __post_init__(self):
        # validate datum
        if isinstance(self.datum, dict):
            self.datum = Datum(**self.datum)

        # validate annotations
        if not isinstance(self.annotations, list):
            raise TypeError("Annotations should be given as a `list`.")
        for idx, annotation in enumerate(self.annotations):
            if isinstance(self.annotations[idx], dict):
                self.annotations[idx] = Annotation(**annotation)


@dataclass
class Prediction:
    model_name: str
    datum: Datum
    annotations: List[Annotation]

    def __post_init__(self):

        # validate datum
        if isinstance(self.datum, dict):
            self.datum = Datum(**self.datum)

        # validate annotations
        if not isinstance(self.annotations, list):
            raise TypeError("Annotations should be given as a `list`.")
        for idx, annotation in enumerate(self.annotations):
            if isinstance(self.annotations[idx], dict):
                self.annotations[idx] = Annotation(**annotation)

        # TaskType-specific validations
        for annotation in self.annotations:
            if annotation.task_type in [
                TaskType.CLASSIFICATION,
                TaskType.DETECTION,
            ]:
                for label in annotation.labels:
                    if label.score is None:
                        raise ValueError(
                            f"For task type `{annotation.task_type}` prediction labels must have scores, but got `None`"
                        )
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
