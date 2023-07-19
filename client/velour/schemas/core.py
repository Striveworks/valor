import json
from abc import ABC
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np

from velour import enums
from velour.schemas.geometry import BoundingBox, MultiPolygon, Polygon, Raster
from velour.schemas.metadata import Metadatum


@dataclass
class Dataset:
    name: str
    id: int = None
    metadata: List[Metadatum] = field(default_factory=list)

    def __post_init__(self):
        assert isinstance(self.name, str)
        assert isinstance(self.id, int | None)
        assert isinstance(self.metadata, list)
        for metadatum in self.metadata:
            assert isinstance(metadatum, Metadatum)


@dataclass
class Model:
    name: str
    id: int = None
    metadata: List[Metadatum] = field(default_factory=list)

    def __post_init__(self):
        assert isinstance(self.name, str)
        assert isinstance(self.id, int | None)
        assert isinstance(self.metadata, list)
        for metadatum in self.metadata:
            assert isinstance(metadatum, Metadatum)


@dataclass
class Info:
    type: enums.DataType = None
    annotation_types: list[enums.AnnotationType] = field(default_factory=list)
    associated_datasets: list[str] = field(default_factory=list)


@dataclass
class Datum:
    uid: str
    metadata: List[Metadatum] = field(default_factory=list)

    def __post_init__(self):
        assert isinstance(self.uid, str)
        assert isinstance(self.metadata, list)
        for metadatum in self.metadata:
            assert isinstance(metadatum, Metadatum)


@dataclass
class Annotation:
    task_type: enums.TaskType
    metadata: List[Metadatum] = field(default_factory=list)

    # geometric types
    bounding_box: BoundingBox = None
    polygon: Polygon = None
    multipolygon: MultiPolygon = None
    raster: Raster = None

    def __post_init__(self):
        if not isinstance(self.task_type, enums.TaskType):
            raise TypeError("Value for task_type is not of type `enums.TaskType`")

        try:
            if self.bounding_box:
                assert isinstance(self.bounding_box, BoundingBox)            
            if self.polygon:
                assert isinstance(self.polygon, Polygon)
            if self.multipolygon:
                assert isinstance(self.multipolygon, MultiPolygon)
            if self.raster:
                assert isinstance(self.raster, Raster)
        except AssertionError:
            raise TypeError("Value does not match intended type.")
        
        try:
            assert isinstance(self.metadata, list)
            for metadatum in self.metadata:
                assert isinstance(metadatum, Metadatum)
        except AssertionError:
            raise TypeError("Metadata contains incorrect type.")


@dataclass
class Label:
    key: str
    value: str

    def __post_init__(self):
        if not isinstance(self.key, str):
            raise TypeError("Label key is not a `str`.")
        if not isinstance(self.value, str):
            raise TypeError("Label value is not a `str`.")

    def tuple(self) -> Tuple[str, str]:
        return (self.key, self.value)

    def __eq__(self, other):
        if hasattr(other, "key") and hasattr(other, "value"):
            return self.key == other.key and self.value == other.value
        return False

    def __hash__(self) -> int:
        return hash(f"key:{self.key},value:{self.value}")


@dataclass
class ScoredLabel:
    label: Label
    score: float

    def __post_init__(self):
        if not isinstance(self.label, Label):
            raise TypeError("label should be of type `velour.schemas.Label`.")
        if not isinstance(self.score, float | int):
            raise TypeError("score should be of type `float`")

    @property
    def key(self):
        return self.label.key

    @property
    def value(self):
        return self.label.value

    def __eq__(self, other):
        if hasattr(other, "label") and hasattr(other, "score"):
            return self.score == other.score and self.label == other.label
        return False
    
    def __neq__(self, other):
        if hasattr(other, "label") and hasattr(other, "score"):
            return not self == other
        return True

    def __hash__(self) -> int:
        return hash(f"key:{self.key},value:{self.value},score:{self.score}")


@dataclass
class LabelDistribution:
    label: Label
    count: int


@dataclass
class ScoredLabelDistribution:
    label: Label
    count: int
    scores: list[float] = field(default_factory=list)


@dataclass
class AnnotationDistribution:
    annotation_type: enums.AnnotationType
    count: int


@dataclass
class GroundTruthAnnotation:
    annotation: Annotation
    labels: List[Label] = field(default_factory=list)

    def __post_init__(self):
        if not isinstance(self.annotation, Annotation):
            raise TypeError("annotation should be type `velour.schemas.Annotation`")
        if not isinstance(self.labels, list):
            raise TypeError("labels should be a list of `velour.schemas.Label`.")
        for label in self.labels:
            if not isinstance(label, Label):
                raise TypeError("labels list should contain only `velour.schemas.Label`.")


@dataclass
class PredictedAnnotation:
    annotation: Annotation
    scored_labels: List[ScoredLabel] = field(default_factory=list)

    def __post_init__(self):
        if not isinstance(self.annotation, Annotation):
            raise TypeError("annotation should be type `velour.schemas.Annotation`.")
        if not isinstance(self.scored_labels, list):
            raise TypeError("scored_labels should be a list of `velour.schemas.ScoredLabel`.")
        for label in self.scored_labels:
            if not isinstance(label, ScoredLabel):
                raise TypeError("scored_labels list should contain only `velour.schemas.ScoredLabel`.")


        # check that for each label key all the predictions sum to ~1
        # the backend should also do this validation but its good to do
        # it here on creation, before sending to backend
        label_keys_to_sum = {}
        for scored_label in self.scored_labels:
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


@dataclass
class GroundTruth:
    datum: Datum
    annotations: list[GroundTruthAnnotation] = field(default_factory=list)
    dataset_name: str = field(default="")

    def __post_init__(self):
        if not isinstance(self.datum, Datum):
            raise TypeError("datum should be type `velour.schemas.Datum`.")
        if not isinstance(self.annotations, list):
            raise TypeError("annotations should be a list of `velour.schemas.GroundTruthAnnotation`.")
        for annotation in self.annotations:
            if not isinstance(annotation, GroundTruthAnnotation):
                raise TypeError("annotations list should contain only `velour.schemas.GroundTruthAnnotation`.")
        if not isinstance(self.dataset_name, str):
            raise TypeError("dataset_name should be type `str`.")


@dataclass
class Prediction:
    datum: Datum
    annotations: list[PredictedAnnotation] = field(default_factory=list)
    model_name: str = field(default="")

    def __post_init__(self):
        if not isinstance(self.datum, Datum):
            raise TypeError("datum should be type `velour.schemas.Datum`.")
        if not isinstance(self.annotations, list):
            raise TypeError("annotations should be a list of `velour.schemas.PredictedAnnotation`.")
        for annotation in self.annotations:
            if not isinstance(annotation, PredictedAnnotation):
                raise TypeError("annotations list should contain only `velour.schemas.PredictedAnnotation`.")
        if not isinstance(self.model_name, str):
            raise TypeError("model_name should be type `str`.")