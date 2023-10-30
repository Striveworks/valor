from dataclasses import dataclass, field
from typing import List

from velour import schemas
from velour.enums import AnnotationType, TaskType


@dataclass
class StringFilter:
    value: str
    operator: str = "=="

    def __post_init__(self):
        allowed_operators = ["==", "!="]
        if self.operator not in allowed_operators:
            raise ValueError(
                f"Invalid comparison operator '{self.operator}'. Allowed operators are {', '.join(allowed_operators)}."
            )


@dataclass
class NumericFilter:
    value: float
    operator: str = "=="

    def __post_init__(self):
        allowed_operators = [">", "<", ">=", "<=", "==", "!="]
        if self.operator not in allowed_operators:
            raise ValueError(
                f"Invalid comparison operator '{self.operator}'. Allowed operators are {', '.join(allowed_operators)}."
            )


@dataclass
class GeospatialFilter:
    operator: str = "=="

    def __post_init__(self):
        allowed_operators = [">", "<", ">=", "<=", "==", "!="]
        if self.operator not in allowed_operators:
            raise ValueError(
                f"Invalid comparison operator '{self.operator}'. Allowed operators are {', '.join(allowed_operators)}."
            )


@dataclass
class GeometricFilter:
    type: AnnotationType
    area: NumericFilter | None = None


@dataclass
class MetadatumFilter:
    key: str
    comparison: NumericFilter | StringFilter | GeospatialFilter


@dataclass
class DatasetFilter:
    ids: List[int] = field(default_factory=list)
    names: List[str] = field(default_factory=list)
    metadata: List[MetadatumFilter] = field(default_factory=list)


@dataclass
class ModelFilter:
    ids: List[int] = field(default_factory=list)
    names: List[str] = field(default_factory=list)
    metadata: List[MetadatumFilter] = field(default_factory=list)


@dataclass
class DatumFilter:
    ids: List[int] = field(default_factory=list)
    uids: List[str] = field(default_factory=list)
    metadata: List[MetadatumFilter] = field(default_factory=list)


@dataclass
class AnnotationFilter:
    task_types: List[TaskType] = field(default_factory=list)
    annotation_types: List[AnnotationType] = field(default_factory=list)
    geometry: List[GeometricFilter] = field(default_factory=list)
    metadata: List[MetadatumFilter] = field(default_factory=list)
    allow_conversion: bool = False


@dataclass
class LabelFilter:
    ids: List[int] = field(default_factory=list)
    labels: List[schemas.Label] = field(default_factory=list)
    keys: List[str] = field(default_factory=list)


@dataclass
class PredictionFilter:
    score: NumericFilter | None = None


@dataclass
class Filter:
    datasets: DatasetFilter = field(default=DatasetFilter())
    models: ModelFilter = field(default=ModelFilter())
    datums: DatumFilter = field(default=DatumFilter())
    annotations: AnnotationFilter = field(default=AnnotationFilter())
    predictions: PredictionFilter = field(default=PredictionFilter())
    labels: LabelFilter = field(default=LabelFilter())
