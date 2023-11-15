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
    operator: str = "=="

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


def _merge(lhs, rhs):
    if isinstance(lhs, list) and isinstance(rhs, list):
        return list(set(lhs).union(set(rhs)))
    elif isinstance(lhs, dict) and isinstance(rhs, dict):
        retval = {}
        for key in set(lhs.keys()).union(set(rhs.keys())):
            if key in lhs and key in rhs:
                retval[key] = _merge(lhs[key], rhs[key])
            elif key in lhs:
                retval[key] = lhs[key]
            else:
                retval[key] = rhs[key]
        return retval
    else:
        raise TypeError("lhs, rhs must both be `list` or `dict`.")


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

    def merge(self, other) -> "Filter":
        if not isinstance(other, Filter):
            raise TypeError("Filter can only merge with other filter objects.")
        filter = Filter()
        # datasets
        filter.dataset_names = _merge(self.dataset_names, other.dataset_names)
        filter.dataset_metadata = _merge(
            self.dataset_metadata, other.dataset_metadata
        )
        filter.dataset_geospatial = _merge(
            self.dataset_geospatial, other.dataset_geospatial
        )
        # models
        filter.models_names = _merge(self.models_names, other.models_names)
        filter.models_metadata = _merge(
            self.models_metadata, other.models_metadata
        )
        filter.models_geospatial = _merge(
            self.models_geospatial, other.models_geospatial
        )
        # datums
        filter.datum_uids = _merge(self.datum_uids, other.datum_uids)
        filter.datum_metadata = _merge(
            self.datum_metadata, other.datum_metadata
        )
        filter.datum_geospatial = _merge(
            self.datum_geospatial, other.datum_geospatial
        )
        # annotations
        filter.task_types = _merge(self.task_types, other.task_types)
        filter.annotation_types = _merge(
            self.annotation_types, other.annotation_types
        )
        filter.annotation_geometric_area = _merge(
            self.annotation_geometric_area, other.annotation_geometric_area
        )
        filter.annotation_metadata = _merge(
            self.annotation_metadata, other.annotation_metadata
        )
        filter.annotation_geospatial = _merge(
            self.annotation_geospatial, other.annotation_geospatial
        )
        # predictions
        filter.prediction_scores = _merge(
            self.prediction_scores, other.prediction_scores
        )
        # labels
        filter.label_ids = _merge(self.label_ids, other.label_ids)
        filter.labels = _merge(self.labels, other.labels)
        filter.label_keys = _merge(self.label_keys, other.label_keys)
        return filter
