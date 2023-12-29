from pydantic import BaseModel, ConfigDict, field_validator

from velour_api.enums import AnnotationType, TaskType
from velour_api.schemas.geojson import GeoJSONPoint, GeoJSONPolygon, GeoJSONMultiPolygon
from velour_api.schemas.metadata import DateTime, Date, Time, Duration


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

    def __eq__(self, other) -> bool:
        """
        Checks that two `StringFilters` are equivalent.

        Parameters
        ----------
        other : StringFilter
            The object to compare against.

        Returns
        ----------
        boolean
            A boolean describing whether the two objects are equal.

        Raises
        ----------
        TypeError
            When comparing against an object that isn't a `StringFilter`.
        """
        if not isinstance(other, StringFilter):
            raise TypeError("expected StringFilter")
        return (
            self.value == other.value,
            self.operator == other.operator,
        )

    def __hash__(self) -> int:
        """
        Hashes a `StringFilter`.

        Returns
        ----------
        int
            A hashed integer.
        """
        return hash(f"Value:{self.value},Op:{self.operator}")


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

    def __eq__(self, other) -> bool:
        """
        Checks that two `NumericFilters` are equivalent.

        Parameters
        ----------
        other : NumericFilter
            The object to compare against.

        Returns
        ----------
        boolean
            A boolean describing whether the two objects are equal.

        Raises
        ----------
        TypeError
            When comparing against an object that isn't a `NumericFilter`.
        """
        if not isinstance(other, NumericFilter):
            raise TypeError("expected NumericFilter")
        return (
            self.value == other.value,
            self.operator == other.operator,
        )

    def __hash__(self) -> int:
        """
        Hashes a `StringFilter`.

        Returns
        ----------
        int
            A hashed integer.
        """
        return hash(f"Value:{self.value},Op:{self.operator}")


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

    value: GeoJSONPoint | GeoJSONPolygon | GeoJSONMultiPolygon
    operator: str = "intersect"

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

    model_config = ConfigDict(extra="forbid")

    def __eq__(self, other) -> bool:
        """
        Checks that two `GeospatialFilters` are equivalent.

        Parameters
        ----------
        other : GeospatialFilter
            The object to compare against.

        Returns
        ----------
        boolean
            A boolean describing whether the two objects are equal.

        Raises
        ----------
        TypeError
            When comparing against an object that isn't a `GeospatialFilter`.
        """
        if not isinstance(other, GeospatialFilter):
            raise TypeError("expected GeospatialFilter")
        return (
            self.value == other.value,
            self.operator == other.operator,
        )

    def __hash__(self) -> int:
        """
        Hashes a `GeospatialFilter`.

        Returns
        ----------
        int
            A hashed integer.
        """
        return hash(f"Value:{self.value.model_dump_json},Op:{self.operator}")
    

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

    def __eq__(self, other) -> bool:
        """
        Checks that `DateTimeFilter` objects are equivalent.

        Parameters
        ----------
        other : DateTimeFilter
            The object to compare against.

        Returns
        ----------
        boolean
            A boolean describing whether the two objects are equal.

        Raises
        ----------
        TypeError
            When comparing against an object that isn't a `DateTimeFilter`.
        """
        if not isinstance(other, DateTimeFilter):
            raise TypeError("expected DateTimeFilter")
        return (
            self.value == other.value,
            self.operator == other.operator,
        )

    def __hash__(self) -> int:
        """
        Hashes a `DateTimeFilter`.

        Returns
        ----------
        int
            A hashed integer.
        """
        return hash(f"Value:{self.value.value},Pattern:{self.value.pattern},Op:{self.operator}")


class Filter(BaseModel):
    """
    Used to filter Evaluations according to specific, user-defined criteria.

    Attributes
    ----------
    dataset_names: List[str], default=None
        A list of `Dataset` names to filter on.
    dataset_metadata: Dict[str, list[StringFilter | NumericFilter | DateTimeFilter]], default=None
        A dictionary of `Dataset` metadata to filter on.
    dataset_geospatial: List[GeospatialFilter]., default=None
        A list of `Dataset` geospatial filters to filter on.
    models_names: List[str], default=None
        A list of `Model` names to filter on.
    models_metadata: Dict[str, list[StringFilter | NumericFilter | DateTimeFilter]], default=None
        A dictionary of `Model` metadata to filter on.
    models_geospatial: List[GeospatialFilter], default=None
        A list of `Model` geospatial filters to filter on.
    datum_ids: List[str], default=None
        A list of `Datum` UIDs to filter on.
    datum_metadata: Dict[str, list[StringFilter | NumericFilter | DateTimeFilter]], default=None
        A dictionary of `Datum` metadata to filter on.
    datum_geospatial: List[GeospatialFilter], default=None
        A list of `Datum` geospatial filters to filter on.
    task_types: List[TaskType], default=None
        A list of task types to filter on.
    annotation_types: List[AnnotationType], default=None
        A list of `Annotation` types to filter on.
    annotation_geometric_area: List[ValueFilter], default=None
        A list of `ValueFilters` which are used to filter `Evaluations` according to the `Annotation`'s geometric area.
    annotation_metadata: Dict[str, list[StringFilter | NumericFilter | DateTimeFilter]], default=None
        A dictionary of `Annotation` metadata to filter on.
    annotation_geospatial: List[GeospatialFilter], default=None
        A list of `Annotation` geospatial filters to filter on.
    prediction_scores: List[ValueFilter], default=None
        A list of `ValueFilters` which are used to filter `Evaluations` according to the `Model`'s prediction scores.
    labels: List[Dict[str, str]], default=None
        A dictionary of `Labels' to filter on.
    label_ids: List[int], default=None
        A list of `Label` IDs to filter on.
    label_keys: List[str] = None, default=None
        A list of `Label` keys to filter on.
    """

    # datasets
    dataset_names: list[str] | None = None
    dataset_metadata: dict[
        str, 
        list[StringFilter | NumericFilter | DateTimeFilter]
    ] | None = None
    dataset_geospatial: list[GeospatialFilter] | None = None

    # models
    models_names: list[str] | None = None
    models_metadata: dict[
        str, 
        list[StringFilter | NumericFilter | DateTimeFilter]
    ] | None = None
    models_geospatial: list[GeospatialFilter] | None = None

    # datums
    datum_ids: list[
        int
    ] | None = None  # This should be used sparingly and with small lists.
    datum_uids: list[str] | None = None
    datum_metadata: dict[
        str, 
        list[StringFilter | NumericFilter | DateTimeFilter]
    ] | None = None
    datum_geospatial: list[GeospatialFilter] | None = None

    # annotations
    task_types: list[TaskType] | None = None
    annotation_types: list[AnnotationType] | None = None
    annotation_geometric_area: list[NumericFilter] | None = None
    annotation_metadata: dict[
        str, 
        list[StringFilter | NumericFilter | DateTimeFilter]
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
