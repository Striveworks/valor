from dataclasses import dataclass, field
from velour.schemas.core import Label

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