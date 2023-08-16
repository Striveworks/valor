from dataclasses import dataclass, field
from typing import List

from velour import enums
from velour.schemas.core import Label


@dataclass
class LabelDistribution:
    label: Label
    count: int


@dataclass
class ScoredLabelDistribution:
    label: Label
    count: int
    scores: List[float] = field(default_factory=list)


@dataclass
class AnnotationDistribution:
    annotation_type: enums.AnnotationType
    count: int
