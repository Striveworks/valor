from dataclasses import dataclass, field
from typing import List, Union

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
class KeyValueFilter:
    key: str
    comparison: Union[NumericFilter, StringFilter, GeospatialFilter]


@dataclass
class DatasetFilter:
    ids: List[int] = field(default_factory=list)
    names: List[str] = field(default_factory=list)
    metadata: List[KeyValueFilter] = field(default_factory=list)
    geo: List[GeospatialFilter] = field(default_factory=list)


@dataclass
class ModelFilter:
    ids: List[int] = field(default_factory=list)
    names: List[str] = field(default_factory=list)
    metadata: List[KeyValueFilter] = field(default_factory=list)
    geo: List[GeospatialFilter] = field(default_factory=list)


@dataclass
class DatumFilter:
    ids: List[int] = field(default_factory=list)
    uids: List[str] = field(default_factory=list)
    metadata: List[KeyValueFilter] = field(default_factory=list)
    geo: List[GeospatialFilter] = field(default_factory=list)


@dataclass
class GeometricAnnotationFilter:
    annotation_type: AnnotationType
    area: List[NumericFilter] = field(default_factory=list)


@dataclass
class JSONAnnotationFilter:
    keys: List[str] = field(default_factory=list)
    values: List[KeyValueFilter] = field(default_factory=list)


@dataclass
class AnnotationFilter:
    task_types: List[TaskType] = field(default_factory=list)
    annotation_types: List[AnnotationType] = field(default_factory=list)
    geometry: List[GeometricAnnotationFilter] = field(default_factory=list)
    json_: List[JSONAnnotationFilter] = field(default_factory=list)
    metadata: List[KeyValueFilter] = field(default_factory=list)
    geo: List[GeospatialFilter] = field(default_factory=list)
    allow_conversion: bool = False


@dataclass
class LabelFilter:
    ids: List[int] = field(default_factory=list)
    labels: List[schemas.Label] = field(default_factory=list)
    keys: List[str] = field(default_factory=list)


@dataclass
class PredictionFilter:
    scores: List[NumericFilter] = field(default_factory=list)


@dataclass
class Filter:
    datasets: DatasetFilter = field(default=DatasetFilter())
    models: ModelFilter = field(default=ModelFilter())
    datums: DatumFilter = field(default=DatumFilter())
    annotations: AnnotationFilter = field(default=AnnotationFilter())
    predictions: PredictionFilter = field(default=PredictionFilter())
    labels: LabelFilter = field(default=LabelFilter())
