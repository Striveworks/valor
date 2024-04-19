import json

from pydantic import (
    BaseModel,
    ConfigDict,
    create_model,
    field_validator,
    model_validator,
)

from valor_api.enums import TaskType
from valor_api.schemas.geometry import GeoJSON
from valor_api.schemas.timestamp import Date, DateTime, Duration, Time


class StringFilter(BaseModel):
    """
    Used to filter on string values that meet some user-defined condition.

    Attributes
    ----------
    value : str
        The value to compare the specific field against.
    operator : str
        The operator to use for comparison. Should be one of `["==", "!="]`.

    Raises
    ------
    ValueError
        If the `operator` doesn't match one of the allowed patterns.
    """

    value: str
    operator: str = "=="

    @field_validator("operator")
    @classmethod
    def _validate_comparison_operator(cls, op: str) -> str:
        """Validate the operator."""
        allowed_operators = ["==", "!="]
        if op not in allowed_operators:
            raise ValueError(
                f"Invalid comparison operator '{op}'. Allowed operators are {', '.join(allowed_operators)}."
            )
        return op

    model_config = ConfigDict(extra="forbid")


class NumericFilter(BaseModel):
    """
    Used to filter on numeric values that meet some user-defined condition.

    Attributes
    ----------
    value : float
        The value to compare the specific field against.
    operator : str
        The operator to use for comparison. Should be one of `[">", "<", ">=", "<=", "==", "!="]`.

    Raises
    ------
    ValueError
        If the `operator` doesn't match one of the allowed patterns.
    """

    value: float
    operator: str = "=="

    @field_validator("operator")
    @classmethod
    def _validate_comparison_operator(cls, op: str) -> str:
        """Validate the operator."""
        allowed_operators = [">", "<", ">=", "<=", "==", "!="]
        if op not in allowed_operators:
            raise ValueError(
                f"Invalid comparison operator '{op}'. Allowed operators are {', '.join(allowed_operators)}."
            )
        return op

    model_config = ConfigDict(extra="forbid")


class BooleanFilter(BaseModel):
    """
    Used to filter on boolean values that meet some user-defined condition.

    Attributes
    ----------
    value : bool
        The value to compare the specific field against.
    operator : str
        The operator to use for comparison. Should be one of `["==", "!="]`.

    Raises
    ------
    ValueError
        If the `operator` doesn't match one of the allowed patterns.
    """

    value: bool
    operator: str = "=="
    model_config = ConfigDict(extra="forbid")

    @field_validator("operator")
    @classmethod
    def _validate_comparison_operator(cls, op: str) -> str:
        """Validate the operator."""
        allowed_operators = ["==", "!="]
        if op not in allowed_operators:
            raise ValueError(
                f"Invalid comparison operator '{op}'. Allowed operators are {', '.join(allowed_operators)}."
            )
        return op


class GeospatialFilter(BaseModel):
    """
    Used to filter on geospatial coordinates.

    Attributes
    ----------
    value : GeoJSON
        A dictionary containing a Point, Polygon, or MultiPolygon. Mirrors `shapely's` `GeoJSON` format.
    operator : str
        The operator to use for comparison. Should be one of `intersect`, `inside`, or `outside`.

    """

    value: GeoJSON
    operator: str = "intersect"
    model_config = ConfigDict(extra="forbid")

    @field_validator("operator")
    @classmethod
    def _validate_comparison_operator(cls, op: str) -> str:
        """Validate the operator."""
        allowed_operators = ["inside", "outside", "intersect"]
        if op not in allowed_operators:
            raise ValueError(
                f"Invalid comparison operator '{op}'. Allowed operators are {', '.join(allowed_operators)}."
            )
        return op


class DateTimeFilter(BaseModel):
    """
    Used to filter on datetime values that meet some user-defined condition.

    Attributes
    ----------
    value : DateTime
        The value to compare the specific field against.
    operator : str
        The operator to use for comparison. Should be one of `[">", "<", ">=", "<=", "==", "!="]`.

    Raises
    ------
    ValueError
        If the `operator` doesn't match one of the allowed patterns.
    """

    value: DateTime | Date | Time | Duration
    operator: str = "=="

    @model_validator(mode="before")
    @classmethod
    def _unpack_timestamp_value(cls, values):
        # TODO - This will be addressed in Issue #526
        if isinstance(values, dict) and (value := values.get("value")):
            if isinstance(value, dict) and (
                "datetime" in value
                or "date" in value
                or "time" in value
                or "duration" in value
            ):
                k, v = list(value.items())[0]
                types = {
                    "datetime": DateTime,
                    "date": Date,
                    "time": Time,
                    "duration": Duration,
                }
                values["value"] = types[k](value=v)
        return values

    @field_validator("operator")
    @classmethod
    def _validate_comparison_operator(cls, op: str) -> str:
        """Validate the operator."""
        allowed_operators = [">", "<", ">=", "<=", "==", "!="]
        if op not in allowed_operators:
            raise ValueError(
                f"Invalid comparison operator '{op}'. Allowed operators are {', '.join(allowed_operators)}."
            )
        return op

    model_config = ConfigDict(extra="forbid")


class Filter(BaseModel):
    """
    Used to filter Evaluations according to specific, user-defined criteria.

    Attributes
    ----------
    dataset_names: List[str], default=None
        A list of `Dataset` names to filter on.
    dataset_metadata: Dict[str, list[StringFilter | NumericFilter | DateTimeFilter | BooleanFilter | GeospatialFilter]], default=None
        A dictionary of `Dataset` metadata to filter on.
    model_names: List[str], default=None
        A list of `Model` names to filter on.
    model_metadata: Dict[str, list[StringFilter | NumericFilter | DateTimeFilter | BooleanFilter | GeospatialFilter]], default=None
        A dictionary of `Model` metadata to filter on.
    datum_metadata: Dict[str, list[StringFilter | NumericFilter | DateTimeFilter | BooleanFilter | GeospatialFilter]], default=None
        A dictionary of `Datum` metadata to filter on.
    task_types: List[TaskType], default=None
        A list of task types to filter on.
    annotation_metadata: Dict[str, list[StringFilter | NumericFilter | DateTimeFilter | BooleanFilter | GeospatialFilter]], default=None
        A dictionary of `Annotation` metadata to filter on.
    require_bounding_box : bool, optional
        A toggle for filtering by bounding boxes.
    bounding_box_area : bool, optional
        An optional constraint to filter by bounding box area.
    require_polygon : bool, optional
        A toggle for filtering by polygons.
    polygon_area : bool, optional
        An optional constraint to filter by polygon area.
    require_raster : bool, optional
        A toggle for filtering by rasters.
    raster_area : bool, optional
        An optional constraint to filter by raster area.
    labels: List[Dict[str, str]], default=None
        A dictionary of `Labels' to filter on.
    label_ids: List[int], default=None
        A list of `Label` IDs to filter on.
    label_keys: List[str] = None, default=None
        A list of `Label` keys to filter on.
    label_scores: List[ValueFilter], default=None
        A list of `ValueFilters` which are used to filter `Evaluations` according to the `Model`'s prediction scores.
    """

    # datasets
    dataset_names: list[str] | None = None
    dataset_metadata: (
        dict[
            str,
            list[
                StringFilter
                | NumericFilter
                | DateTimeFilter
                | BooleanFilter
                | GeospatialFilter
            ],
        ]
        | None
    ) = None

    # models
    model_names: list[str] | None = None
    model_metadata: (
        dict[
            str,
            list[
                StringFilter
                | NumericFilter
                | DateTimeFilter
                | BooleanFilter
                | GeospatialFilter
            ],
        ]
        | None
    ) = None

    # datums
    datum_uids: list[str] | None = None
    datum_metadata: (
        dict[
            str,
            list[
                StringFilter
                | NumericFilter
                | DateTimeFilter
                | BooleanFilter
                | GeospatialFilter
            ],
        ]
        | None
    ) = None

    # annotations
    task_types: list[TaskType] | None = None
    annotation_metadata: (
        dict[
            str,
            list[
                StringFilter
                | NumericFilter
                | DateTimeFilter
                | BooleanFilter
                | GeospatialFilter
            ],
        ]
        | None
    ) = None
    require_bounding_box: bool | None = None
    bounding_box_area: list[NumericFilter] | None = None
    require_polygon: bool | None = None
    polygon_area: list[NumericFilter] | None = None
    require_raster: bool | None = None
    raster_area: list[NumericFilter] | None = None

    # labels
    labels: list[dict[str, str]] | None = None
    label_ids: list[int] | None = None
    label_keys: list[str] | None = None

    # predictions
    label_scores: list[NumericFilter] | None = None

    # pydantic settings
    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=("protected_",),
    )


# we want to pass a Filter as a query parameters instead of a body
# so we make a new model `FilterQueryParams` where every value is a JSON string
model_fields = Filter.model_fields
model_def_dict = {kwarg: (str | None, None) for kwarg in model_fields}
FilterQueryParams = create_model(
    "FilterQueryParams",
    __config__=ConfigDict(extra="forbid"),
    **model_def_dict,  # type: ignore
)


def convert_filter_query_params_to_filter_obj(filter_query_params) -> Filter:
    """Converts a `FilterQueryParams` object to a `Filter` object by
    loading from JSON strings.

    Parameters
    ----------
    filter_query_params : FilterQueryParams
        The `FilterQueryParams` object to convert.

    Returns
    -------
    Filter
        The converted `Filter` object.
    """
    return Filter(
        **{
            k: json.loads(v if v is not None else "null")
            for k, v in filter_query_params.model_dump().items()
        }
    )
