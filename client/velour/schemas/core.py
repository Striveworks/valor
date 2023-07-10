import json
from abc import ABC
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np

from velour import enums
from velour.schemas.geometry import (
    Box,
    MultiPolygon,
    Polygon,
    Raster,
)
from velour.schemas.metadata import Metadatum


@dataclass
class Dataset:
    name: str
    id: int = None
    metadata: List[Metadatum] = field(default_factory=list)
    

@dataclass
class Model:
    name: str
    id: int = None
    metadata: List[Metadatum] = field(default_factory=list)


@dataclass
class Info:
    href: str = None
    description: str = None
    type: enums.DataType = None
    annotation_types: list[enums.AnnotationType] = field(default_factory=list)
    associated_datasets: list[str] = field(default_factory=list)


@dataclass
class Datum:
    dataset: Dataset
    uid: str
    metadata: List[Metadatum] = field(default_factory=list)


@dataclass
class Annotation:
    geometry: Optional[
        Union[Box, Polygon, MultiPolygon, Raster]
    ] = None
    # other type of annotation
    metadata: List[Metadatum] = field(default_factory=list)

    def __post_init__(self):
        # check at least one attribute is not None
        if self.geometry is None:
            raise ValueError


@dataclass
class Label:
    key: str
    value: str

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
class GroundTruth:
    datum: Datum
    labels: List[Label] = field(default_factory=list)
    annotation: Annotation = None


@dataclass
class AnnotationDistribution:
    annotation_type: enums.AnnotationType
    count: int


@dataclass
class Prediction:
    datum: Datum
    scored_labels: List[ScoredLabel] = field(default_factory=list)
    annotation: Annotation = None

    def __post_init__(self):
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
