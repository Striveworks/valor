from pydantic import BaseModel, ConfigDict, field_validator

from velour_api.enums import AnnotationType, TaskType
from velour_api.schemas.geojson import GeoJSON


class StringFilter(BaseModel):
    value: str
    operator: str = "=="

    @field_validator("operator")
    @classmethod
    def validate_comparison_operator(cls, op: str) -> str:
        allowed_operators = ["==", "!="]
        if op not in allowed_operators:
            raise ValueError(
                f"Invalid comparison operator '{op}'. Allowed operators are {', '.join(allowed_operators)}."
            )
        return op

    model_config = ConfigDict(extra="forbid")

    def __eq__(self, other) -> bool:
        if not isinstance(other, StringFilter):
            raise TypeError("expected StringFilter")
        return (
            self.value == other.value,
            self.operator == other.operator,
        )

    def __hash__(self):
        return hash(f"Value:{self.value},Op:{self.operator}")


class NumericFilter(BaseModel):
    value: float
    operator: str = "=="

    @field_validator("operator")
    @classmethod
    def validate_comparison_operator(cls, op: str) -> str:
        allowed_operators = [">", "<", ">=", "<=", "==", "!="]
        if op not in allowed_operators:
            raise ValueError(
                f"Invalid comparison operator '{op}'. Allowed operators are {', '.join(allowed_operators)}."
            )
        return op

    model_config = ConfigDict(extra="forbid")

    def __eq__(self, other) -> bool:
        if not isinstance(other, NumericFilter):
            raise TypeError("expected NumericFilter")
        return (
            self.value == other.value,
            self.operator == other.operator,
        )

    def __hash__(self):
        return hash(f"Value:{self.value},Op:{self.operator}")


class GeospatialFilter(BaseModel):
    value: GeoJSON
    operator: str = "intersect"

    @field_validator("operator")
    @classmethod
    def validate_comparison_operator(cls, op: str) -> str:
        allowed_operators = ["inside", "outside", "intersect"]
        if op not in allowed_operators:
            raise ValueError(
                f"Invalid comparison operator '{op}'. Allowed operators are {', '.join(allowed_operators)}."
            )
        return op

    model_config = ConfigDict(extra="forbid")

    def __eq__(self, other) -> bool:
        if not isinstance(other, GeospatialFilter):
            raise TypeError("expected GeospatialFilter")
        return (
            self.value == other.value,
            self.operator == other.operator,
        )

    def __hash__(self):
        return hash(f"Value:{self.value.model_dump_json},Op:{self.operator}")


class Filter(BaseModel):
    # datasets
    dataset_names: list[str] | None = None
    dataset_metadata: dict[
        str, StringFilter | list[NumericFilter]
    ] | None = None
    dataset_geospatial: list[GeospatialFilter] | None = None

    # models
    models_names: list[str] | None = None
    models_metadata: dict[
        str, StringFilter | list[NumericFilter]
    ] | None = None
    models_geospatial: list[GeospatialFilter] | None = None

    # datums
    datum_ids: list[
        int
    ] | None = None  # This should be used sparingly and with small lists.
    datum_uids: list[str] | None = None
    datum_metadata: dict[str, StringFilter | list[NumericFilter]] | None = None
    datum_geospatial: list[GeospatialFilter] | None = None

    # annotations
    task_types: list[TaskType] | None = None
    annotation_types: list[AnnotationType] | None = None
    annotation_geometric_area: list[NumericFilter] | None = None
    annotation_metadata: dict[
        str, StringFilter | list[NumericFilter]
    ] | None = None
    annotation_geospatial: list[GeospatialFilter] | None = None

    # predictions
    prediction_scores: list[NumericFilter] | None = None

    # labels
    labels: list[dict[str, str]] | None = None
    label_ids: list[int] | None = None
    label_keys: list[str] | None = None

    # pydantic settings
    model_config = ConfigDict(extra="forbid")
