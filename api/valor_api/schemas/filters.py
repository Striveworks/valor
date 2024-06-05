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
from valor_api.schemas.validators import (
    validate_type_bool,
    validate_type_box,
    validate_type_date,
    validate_type_datetime,
    validate_type_duration,
    validate_type_float,
    validate_type_integer,
    validate_type_linestring,
    validate_type_multilinestring,
    validate_type_multipoint,
    validate_type_multipolygon,
    validate_type_point,
    validate_type_polygon,
    validate_type_string,
    validate_type_time,
)


def validate_type_symbol(x):
    if not isinstance(x, Symbol):
        raise TypeError


filterable_types_to_validator = {
    "symbol": validate_type_symbol,
    "bool": validate_type_bool,
    "string": validate_type_string,
    "integer": validate_type_integer,
    "float": validate_type_float,
    "datetime": validate_type_datetime,
    "date": validate_type_date,
    "time": validate_type_time,
    "duration": validate_type_duration,
    "point": validate_type_point,
    "multipoint": validate_type_multipoint,
    "linestring": validate_type_linestring,
    "multilinestring": validate_type_multilinestring,
    "polygon": validate_type_polygon,
    "box": validate_type_box,
    "multipolygon": validate_type_multipolygon,
    "tasktypeenum": validate_type_string,
    "label": None,
    "embedding": None,
    "raster": None,
}


class Symbol(BaseModel):
    type: str
    name: str
    key: str | None = None
    attribute: str | None = None


class Value(BaseModel):
    type: str
    value: bool | int | float | str | list | dict
    model_config = ConfigDict(extra="forbid")


class Operands(BaseModel):
    lhs: Symbol
    rhs: Value
    model_config = ConfigDict(extra="forbid")


class And(BaseModel):
    logical_and: list["FunctionType"]
    model_config = ConfigDict(extra="forbid")

    @property
    def op(self) -> str:
        return type(self).__name__.lower()

    @property
    def args(self):
        return self.logical_and


class Or(BaseModel):
    logical_or: list["FunctionType"]
    model_config = ConfigDict(extra="forbid")

    @property
    def op(self) -> str:
        return type(self).__name__.lower()

    @property
    def args(self):
        return self.logical_or


class Not(BaseModel):
    logical_not: "FunctionType"
    model_config = ConfigDict(extra="forbid")

    @property
    def op(self) -> str:
        return type(self).__name__.lower()

    @property
    def arg(self):
        return self.logical_not


class IsNull(BaseModel):
    isnull: Symbol
    model_config = ConfigDict(extra="forbid")

    @property
    def op(self) -> str:
        return type(self).__name__.lower()

    @property
    def arg(self):
        return self.isnull


class IsNotNull(BaseModel):
    isnotnull: Symbol
    model_config = ConfigDict(extra="forbid")

    @property
    def op(self) -> str:
        return type(self).__name__.lower()

    @property
    def arg(self):
        return self.isnotnull


class Equal(BaseModel):
    eq: Operands
    model_config = ConfigDict(extra="forbid")

    @property
    def op(self) -> str:
        return type(self).__name__.lower()

    @property
    def lhs(self):
        return self.eq.lhs

    @property
    def rhs(self):
        return self.eq.rhs


class NotEqual(BaseModel):
    ne: Operands
    model_config = ConfigDict(extra="forbid")

    @property
    def op(self) -> str:
        return type(self).__name__.lower()

    @property
    def lhs(self):
        return self.ne.lhs

    @property
    def rhs(self):
        return self.ne.rhs


class GreaterThan(BaseModel):
    gt: Operands
    model_config = ConfigDict(extra="forbid")

    @property
    def op(self) -> str:
        return type(self).__name__.lower()

    @property
    def lhs(self):
        return self.gt.lhs

    @property
    def rhs(self):
        return self.gt.rhs


class GreaterThanEqual(BaseModel):
    ge: Operands
    model_config = ConfigDict(extra="forbid")

    @property
    def op(self) -> str:
        return type(self).__name__.lower()

    @property
    def lhs(self):
        return self.ge.lhs

    @property
    def rhs(self):
        return self.ge.rhs


class LessThan(BaseModel):
    lt: Operands
    model_config = ConfigDict(extra="forbid")

    @property
    def op(self) -> str:
        return type(self).__name__.lower()

    @property
    def lhs(self):
        return self.lt.lhs

    @property
    def rhs(self):
        return self.lt.rhs


class LessThanEqual(BaseModel):
    le: Operands
    model_config = ConfigDict(extra="forbid")

    @property
    def op(self) -> str:
        return type(self).__name__.lower()

    @property
    def lhs(self):
        return self.le.lhs

    @property
    def rhs(self):
        return self.le.rhs


class Intersects(BaseModel):
    intersects: Operands
    model_config = ConfigDict(extra="forbid")

    @property
    def op(self) -> str:
        return type(self).__name__.lower()

    @property
    def lhs(self):
        return self.intersects.lhs

    @property
    def rhs(self):
        return self.intersects.rhs


class Inside(BaseModel):
    inside: Operands
    model_config = ConfigDict(extra="forbid")

    @property
    def op(self) -> str:
        return type(self).__name__.lower()

    @property
    def lhs(self):
        return self.inside.lhs

    @property
    def rhs(self):
        return self.inside.rhs


class Outside(BaseModel):
    outside: Operands
    model_config = ConfigDict(extra="forbid")

    @property
    def op(self) -> str:
        return type(self).__name__.lower()

    @property
    def lhs(self):
        return self.outside.lhs

    @property
    def rhs(self):
        return self.outside.rhs


class Contains(BaseModel):
    contains: Operands
    model_config = ConfigDict(extra="forbid")

    @property
    def op(self) -> str:
        return type(self).__name__.lower()

    @property
    def lhs(self):
        return self.contains.lhs

    @property
    def rhs(self):
        return self.contains.rhs


NArgFunction = And | Or
OneArgFunction = Not | IsNull | IsNotNull
TwoArgFunction = (
    Equal
    | NotEqual
    | GreaterThan
    | GreaterThanEqual
    | LessThan
    | LessThanEqual
    | Intersects
    | Inside
    | Outside
    | Contains
)
FunctionType = OneArgFunction | TwoArgFunction | NArgFunction


#


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

    def to_function(
        self, name: str, key: str | None = None, attribute: str | None = None
    ) -> FunctionType:
        operands = Operands(
            lhs=Symbol(type="string", name=name, key=key, attribute=attribute),
            rhs=Value(type="string", value=self.value),
        )
        match self.operator:
            case "==":
                return Equal(eq=operands)
            case "!=":
                return NotEqual(ne=operands)
            case _:
                raise RuntimeError


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

    def to_function(
        self,
        name: str,
        key: str | None = None,
        attribute: str | None = None,
        type_str: str = "float",
    ) -> FunctionType:
        operands = Operands(
            lhs=Symbol(type=type_str, name=name, key=key, attribute=attribute),
            rhs=Value(type="float", value=self.value),
        )
        match self.operator:
            case "==":
                return Equal(eq=operands)
            case "!=":
                return NotEqual(ne=operands)
            case ">":
                return GreaterThan(gt=operands)
            case ">=":
                return GreaterThanEqual(ge=operands)
            case "<":
                return LessThan(lt=operands)
            case "<=":
                return LessThanEqual(le=operands)
            case _:
                raise RuntimeError


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

    def to_function(
        self, name: str, key: str | None = None, attribute: str | None = None
    ) -> FunctionType:
        operands = Operands(
            lhs=Symbol(
                type="boolean", name=name, key=key, attribute=attribute
            ),
            rhs=Value(type="boolean", value=self.value),
        )
        match self.operator:
            case "==":
                return Equal(eq=operands)
            case "!=":
                return NotEqual(ne=operands)
            case _:
                raise RuntimeError


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

    def to_function(
        self, name: str, key: str | None = None, attribute: str | None = None
    ) -> FunctionType:
        operands = Operands(
            lhs=Symbol(
                type="geojson", name=name, key=key, attribute=attribute
            ),
            rhs=Value(type="geojson", value=self.value.geometry.to_json()),
        )
        match self.operator:
            case "inside":
                return Inside(inside=operands)
            case "outside":
                return Outside(outside=operands)
            case "intersect":
                return Intersects(intersects=operands)
            case _:
                raise RuntimeError


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

    def to_function(
        self, name: str, key: str | None = None, attribute: str | None = None
    ) -> FunctionType:
        type_str = type(self.value).__name__.lower()
        operands = Operands(
            lhs=Symbol(type=type_str, name=name, key=key, attribute=attribute),
            rhs=Value(type=type_str, value=self.value.value),
        )
        match self.operator:
            case "==":
                return Equal(eq=operands)
            case "!=":
                return NotEqual(ne=operands)
            case ">":
                return GreaterThan(gt=operands)
            case ">=":
                return GreaterThanEqual(ge=operands)
            case "<":
                return LessThan(lt=operands)
            case "<=":
                return LessThanEqual(le=operands)
            case _:
                raise RuntimeError


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


#


class AdvancedFilter(BaseModel):
    """
    New filter schema to replace the ab

    The intent is for this object to replace 'Filter' in a future PR.
    """

    datasets: FunctionType | None = None
    models: FunctionType | None = None
    datums: FunctionType | None = None
    annotations: FunctionType | None = None
    groundtruths: FunctionType | None = None
    predictions: FunctionType | None = None
    labels: FunctionType | None = None
    embeddings: FunctionType | None = None

    @classmethod
    def from_simple_filter(
        cls,
        filter_: Filter,
        ignore_groundtruths: bool = False,
        ignore_predictions: bool = False,
    ):
        def filter_equatable(
            name: str,
            values: list[str] | list[TaskType] | list[int],
            type_str: str = "string",
        ) -> FunctionType | None:
            if len(values) > 1:
                return Or(
                    logical_or=[
                        Equal(
                            eq=Operands(
                                lhs=Symbol(type=type_str, name=name),
                                rhs=Value(
                                    type=type_str,
                                    value=value.value
                                    if isinstance(value, TaskType)
                                    else value,
                                ),
                            )
                        )
                        for value in values
                    ]
                )
            elif len(values) == 1:
                value = (
                    values[0].value
                    if isinstance(values[0], TaskType)
                    else values[0]
                )
                return Equal(
                    eq=Operands(
                        lhs=Symbol(type=type_str, name=name),
                        rhs=Value(type=type_str, value=value),
                    )
                )
            else:
                return None

        def filter_metadata(
            name: str, values: dict[str, list]
        ) -> FunctionType | None:
            filter_expressions = [
                f.to_function(name=name, key=key)
                for key, filters in values.items()
                for f in filters
            ]
            if len(filter_expressions) > 1:
                return And(logical_and=filter_expressions)
            elif len(filter_expressions) == 1:
                return filter_expressions[0]
            else:
                return None

        def annotation_geometry_exist(
            type_str: str, name: str, exists: bool
        ) -> IsNull | IsNotNull:
            if exists:
                return IsNotNull(isnotnull=Symbol(type=type_str, name=name))
            else:
                return IsNull(isnull=Symbol(type=type_str, name=name))

        def filter_numerics(
            type_str: str,
            name: str,
            values: list[NumericFilter],
            attribute: str | None = None,
        ) -> FunctionType | None:
            expressions = [
                f.to_function(
                    name=name, attribute=attribute, type_str=type_str
                )
                for f in values
            ]
            if len(expressions) > 1:
                return And(logical_and=expressions)
            elif len(expressions) == 1:
                return expressions[0]
            else:
                return None

        def filter_labels(
            values: list[dict[str, str]],
        ) -> FunctionType | None:
            if len(values) > 1:
                return Or(
                    logical_or=[
                        And(
                            logical_and=[
                                Equal(
                                    eq=Operands(
                                        lhs=Symbol(
                                            type="string", name="label.key"
                                        ),
                                        rhs=Value(type="string", value=key),
                                    )
                                ),
                                Equal(
                                    eq=Operands(
                                        lhs=Symbol(
                                            type="string", name="label.value"
                                        ),
                                        rhs=Value(type="string", value=value),
                                    )
                                ),
                            ]
                        )
                        for label in values
                        for key, value in label.items()
                    ]
                )
            elif len(values) == 1:
                key = list(values[0].keys())[0]
                value = list(values[0].values())[0]
                return And(
                    logical_and=[
                        Equal(
                            eq=Operands(
                                lhs=Symbol(type="string", name="label.key"),
                                rhs=Value(type="string", value=key),
                            )
                        ),
                        Equal(
                            eq=Operands(
                                lhs=Symbol(type="string", name="label.value"),
                                rhs=Value(type="string", value=value),
                            )
                        ),
                    ]
                )
            else:
                return None

        dataset_names = None
        dataset_metadata = None
        model_names = None
        model_metadata = None
        datum_uids = None
        datum_metadata = None
        annotation_task_types = None
        annotation_metadata = None
        annotation_box = None
        annotation_box_area = None
        annotation_polygon = None
        annotation_polygon_area = None
        annotation_raster = None
        annotation_raster_area = None
        labels = None
        label_keys = None
        label_scores = None
        label_ids = None

        if filter_.dataset_names:
            dataset_names = filter_equatable(
                name="dataset.name", values=filter_.dataset_names
            )
        if filter_.dataset_metadata:
            dataset_metadata = filter_metadata(
                name="dataset.metadata", values=filter_.dataset_metadata
            )
        if filter_.model_names:
            model_names = filter_equatable(
                name="model.name", values=filter_.model_names
            )
        if filter_.model_metadata:
            model_metadata = filter_metadata(
                name="model.metadata", values=filter_.model_metadata
            )
        if filter_.datum_uids:
            datum_uids = filter_equatable(
                name="datum.uid", values=filter_.datum_uids
            )
        if filter_.datum_metadata:
            datum_metadata = filter_metadata(
                name="datum.metadata", values=filter_.datum_metadata
            )
        if filter_.task_types:
            annotation_task_types = filter_equatable(
                name="annotation.task_type", values=filter_.task_types
            )
        if filter_.annotation_metadata:
            annotation_metadata = filter_metadata(
                name="annotation.metadata", values=filter_.annotation_metadata
            )
        if filter_.require_bounding_box is not None:
            annotation_box = annotation_geometry_exist(
                type_str="box",
                name="annotation.bounding_box",
                exists=filter_.require_bounding_box,
            )
        if filter_.bounding_box_area:
            annotation_box_area = filter_numerics(
                type_str="box",
                name="annotation.bounding_box",
                attribute="area",
                values=filter_.bounding_box_area,
            )
        if filter_.require_polygon is not None:
            annotation_polygon = annotation_geometry_exist(
                type_str="polygon",
                name="annotation.polygon",
                exists=filter_.require_polygon,
            )
        if filter_.polygon_area:
            annotation_polygon_area = filter_numerics(
                type_str="polygon",
                name="annotation.polygon",
                attribute="area",
                values=filter_.polygon_area,
            )
        if filter_.require_raster is not None:
            annotation_raster = annotation_geometry_exist(
                type_str="raster",
                name="annotation.raster",
                exists=filter_.require_raster,
            )
        if filter_.raster_area:
            annotation_raster_area = filter_numerics(
                type_str="raster",
                name="annotation.raster",
                attribute="area",
                values=filter_.raster_area,
            )
        if filter_.labels:
            labels = filter_labels(filter_.labels)
        if filter_.label_keys:
            label_keys = filter_equatable(
                name="label.key", values=filter_.label_keys
            )
        if filter_.label_scores:
            label_scores = filter_numerics(
                type_str="float",
                name="label.score",
                values=filter_.label_scores,
            )
        if filter_.label_ids:
            label_ids = filter_equatable(
                name="label.id", values=filter_.label_ids, type_str="integer"
            )

        def and_if_list(values: list[FunctionType]) -> FunctionType | None:
            if len(values) > 1:
                return And(logical_and=values)
            elif len(values) == 1:
                return values[0]
            else:
                return None

        groundtruth_filter = and_if_list(
            [
                expr
                for expr in [
                    dataset_names,
                    dataset_metadata,
                    datum_uids,
                    datum_metadata,
                    annotation_task_types,
                    annotation_metadata,
                    annotation_box,
                    annotation_box_area,
                    annotation_polygon,
                    annotation_polygon_area,
                    annotation_raster,
                    annotation_raster_area,
                    labels,
                    label_keys,
                    label_ids,
                ]
                if expr is not None
            ]
        )
        prediction_filter = and_if_list(
            [
                expr
                for expr in [
                    dataset_names,
                    dataset_metadata,
                    datum_uids,
                    datum_metadata,
                    model_names,
                    model_metadata,
                    annotation_task_types,
                    annotation_metadata,
                    annotation_box,
                    annotation_box_area,
                    annotation_polygon,
                    annotation_polygon_area,
                    annotation_raster,
                    annotation_raster_area,
                    labels,
                    label_keys,
                    label_ids,
                    label_scores,
                ]
                if expr is not None
            ]
        )
        annotation_filter = and_if_list(
            [
                expr
                for expr in [
                    datum_uids,
                    datum_metadata,
                    annotation_task_types,
                    annotation_metadata,
                    annotation_box,
                    annotation_box_area,
                    annotation_polygon,
                    annotation_polygon_area,
                    annotation_raster,
                    annotation_raster_area,
                ]
                if expr is not None
            ]
        )
        label_filter = and_if_list(
            [
                expr
                for expr in [
                    labels,
                    label_keys,
                    label_ids,
                    label_scores,
                ]
                if expr is not None
            ]
        )
        dataset_filter = and_if_list(
            [
                expr
                for expr in [
                    dataset_names,
                    dataset_metadata,
                ]
                if expr is not None
            ]
        )
        model_filter = and_if_list(
            [
                expr
                for expr in [
                    model_names,
                    model_metadata,
                ]
                if expr is not None
            ]
        )

        f = cls()
        if ignore_groundtruths:
            f.predictions = prediction_filter
        elif ignore_predictions:
            f.groundtruths = groundtruth_filter
        else:
            f.annotations = annotation_filter
            f.labels = label_filter
            f.models = model_filter
            f.datasets = dataset_filter

        return f
