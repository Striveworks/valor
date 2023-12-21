from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Union

from velour.enums import AnnotationType, TaskType


@dataclass
class ValueFilter:
    """
    Used to filter on string or numeric values that meet some user-defined condition.

    Attributes
    ----------
    value : Union[int, float, str]
        The value to compare the specific field against.
    operator : str
        The operator to use for comparison. Should be one of `[">", "<", ">=", "<=", "==", "!="]` if the value is an int or float, otherwise should be one of `["==", "!="]`.

    Raises
    ------
    TypeError
        If `value` isn't of the correct type.
    ValueError
        If the `operator` doesn't match one of the allowed patterns.
    """

    value: Union[int, float, str]
    operator: str = "=="

    def __post_init__(self):
        if isinstance(self.value, int) or isinstance(self.value, float):
            allowed_operators = [">", "<", ">=", "<=", "==", "!="]
        elif isinstance(self.value, str):
            allowed_operators = ["==", "!="]
        else:
            raise TypeError(
                "`value` should be of type `int`, `float` or `str`"
            )
        if self.operator not in allowed_operators:
            raise ValueError(
                f"Invalid comparison operator '{self.operator}'. Allowed operators are {', '.join(allowed_operators)}."
            )


@dataclass
class GeospatialFilter:
    """
    Used to filter on geospatial coordinates.

    Attributes
    ----------
    geodict : Dict[str, Union[List[List[List[List[Union[float, int]]]]], List[List[List[Union[float, int]]]], List[Union[float, int]], str]]
        A dictionary containing a Point, Polygon, or MultiPolygon. Mirrors `shapely's` `GeoJSON` format.
    operator : str
        The operator to use for comparison. Should be one of `intersect`, `inside`, or `outside`.

    """

    value: Dict[
        str,
        Union[
            List[List[List[List[Union[float, int]]]]],
            List[List[List[Union[float, int]]]],
            List[Union[float, int]],
            str,
        ],
    ]
    operator: str = "intersect"

    def __post_init__(self):
        allowed_operators = ["inside", "outside", "intersect"]
        if self.operator not in allowed_operators:
            raise ValueError(
                f"Invalid comparison operator '{self.operator}'. Allowed operators are {', '.join(allowed_operators)}."
            )


@dataclass
class BinaryExpression:
    name: str
    value: object
    operator: str
    key: Union[str, Enum, None] = None


class DeclarativeMapper:
    def __init__(self, name: str, object_type: object, key: str = None):
        self.name = name
        self.key = key
        self.object_type = object_type

    def _validate_operator(self, value):
        if self.object_type == float and isinstance(value, int):
            return  # edge case
        if not isinstance(value, self.object_type):
            raise ValueError(
                f"`{self.name}` should be of type `{self.object_type}`"
            )

    def _validate_geospatial_operator(self, value):
        if (
            not isinstance(value, dict)
            or not value.get("geometry")
            or not value["geometry"].get("type")
            or not value["geometry"].get("coordinates")
        ):
            raise ValueError(
                "Geospatial filters should be a GeoJSON-style dictionary containing the keys `type` and `coordinates`."
            )

    def __getitem__(self, key: str):
        return DeclarativeMapper(
            name=self.name,
            object_type=self.object_type,
            key=key,
        )

    def __eq__(self, __value: object) -> BinaryExpression:
        self._validate_operator(
            __value,
        )
        return BinaryExpression(
            name=self.name,
            key=self.key,
            value=__value,
            operator="==",
        )

    def __ne__(self, __value: object) -> BinaryExpression:
        self._validate_operator(__value)
        return BinaryExpression(
            name=self.name,
            key=self.key,
            value=__value,
            operator="!=",
        )

    def __lt__(self, __value: object) -> BinaryExpression:
        self._validate_operator(__value)
        return BinaryExpression(
            name=self.name,
            key=self.key,
            value=__value,
            operator="<",
        )

    def __gt__(self, __value: object) -> BinaryExpression:
        if isinstance(__value, str):
            raise TypeError("`__gt__` does not support type `str`")
        self._validate_operator(__value)
        return BinaryExpression(
            name=self.name,
            key=self.key,
            value=__value,
            operator=">",
        )

    def __le__(self, __value: object) -> BinaryExpression:
        if isinstance(__value, str):
            raise TypeError("`__le__` does not support type `str`")
        self._validate_operator(__value)
        return BinaryExpression(
            name=self.name,
            key=self.key,
            value=__value,
            operator="<=",
        )

    def __ge__(self, __value: object) -> BinaryExpression:
        if isinstance(__value, str):
            raise TypeError("`__ge__` does not support type `str`")
        self._validate_operator(__value)
        return BinaryExpression(
            name=self.name,
            key=self.key,
            value=__value,
            operator=">=",
        )

    def in_(self, __values: List[object]) -> List[BinaryExpression]:
        if not isinstance(__values, list):
            raise TypeError("`in_` takes a list as input.")
        for value in __values:
            if not isinstance(value, self.object_type):
                raise TypeError(
                    f"All elements must be of type `{self.object_type}`, got '{type(value)}'"
                )
        return [self == value for value in __values]

    def intersect(self, __value: dict) -> BinaryExpression:
        self._validate_geospatial_operator(
            __value,
        )
        return BinaryExpression(
            name=self.name,
            key=self.key,
            value=__value,
            operator="intersect",
        )

    def inside(self, __value: object) -> BinaryExpression:
        self._validate_geospatial_operator(
            __value,
        )
        return BinaryExpression(
            name=self.name,
            key=self.key,
            value=__value,
            operator="inside",
        )

    def outside(self, __value: object) -> BinaryExpression:
        self._validate_geospatial_operator(
            __value,
        )
        return BinaryExpression(
            name=self.name,
            key=self.key,
            value=__value,
            operator="outside",
        )


@dataclass
class Filter:
    """
    Used to filter Evaluations according to specific, user-defined criteria.

    Attributes
    ----------
    dataset_names: List[str]
        A list of `Dataset` names to filter on.
    dataset_metadata: Dict[str, List[ValueFilter]]
        A dictionary of `Dataset` metadata to filter on.
    dataset_geospatial: List[GeospatialFilter].
        A list of `Dataset` geospatial filters to filter on.
    models_names: List[str]
        A list of `Model` names to filter on.
    models_metadata: Dict[str, List[ValueFilter]]
        A dictionary of `Model` metadata to filter on.
    models_geospatial: List[GeospatialFilter]
        A list of `Model` geospatial filters to filter on.
    datum_uids: List[str]
        A list of `Datum` UIDs to filter on.
    datum_metadata: Dict[str, List[ValueFilter]] = None
        A dictionary of `Datum` metadata to filter on.
    datum_geospatial: List[GeospatialFilter]
        A list of `Datum` geospatial filters to filter on.
    task_types: List[TaskType]
        A list of task types to filter on.
    annotation_types: List[AnnotationType]
        A list of `Annotation` types to filter on.
    annotation_geometric_area: List[ValueFilter]
        A list of `ValueFilters` which are used to filter `Evaluations` according to the `Annotation`'s geometric area.
    annotation_metadata: Dict[str, List[ValueFilter]]
        A dictionary of `Annotation` metadata to filter on.
    annotation_geospatial: List[GeospatialFilter]
        A list of `Annotation` geospatial filters to filter on.
    prediction_scores: List[ValueFilter]
        A list of `ValueFilters` which are used to filter `Evaluations` according to the `Model`'s prediction scores.
    label_ids: List[int]
        A list of `Label` IDs to filter on.
    label_keys: List[str]
        A list of `Label` keys to filter on.
    label_values: List[str]
        A list of `Label` values to filter on.


    Raises
    ------
    TypeError
        If `value` isn't of the correct type.
    ValueError
        If the `operator` doesn't match one of the allowed patterns.
    """

    # datasets
    dataset_names: List[str] = None
    dataset_metadata: Dict[str, List[ValueFilter]] = None
    dataset_geospatial: List[GeospatialFilter] = None

    # models
    models_names: List[str] = None
    models_metadata: Dict[str, List[ValueFilter]] = None
    models_geospatial: List[GeospatialFilter] = None

    # datums
    datum_uids: List[str] = None
    datum_metadata: Dict[str, List[ValueFilter]] = None
    datum_geospatial: List[GeospatialFilter] = None

    # annotations
    task_types: List[TaskType] = None
    annotation_types: List[AnnotationType] = None
    annotation_geometric_area: List[ValueFilter] = None
    annotation_metadata: Dict[str, List[ValueFilter]] = None
    annotation_geospatial: List[GeospatialFilter] = None

    # predictions
    prediction_scores: List[ValueFilter] = None

    # labels
    label_ids: List[int] = None
    label_keys: List[str] = None
    label_values: List[str] = None

    @classmethod
    def create(cls, expressions: List[BinaryExpression]) -> "Filter":
        """
        Parses a list of `BinaryExpression` to create a `schemas.Filter` object.

        Parameters
        ----------
        expressions: List[BinaryExpression]
            A list of `BinaryExpressions' to parse into a `Filter` object.
        """

        # expand nested expressions
        expression_list = [
            expr for expr in expressions if isinstance(expr, BinaryExpression)
        ] + [
            expr_
            for expr in expressions
            if isinstance(expr, list)
            for expr_ in expr
            if isinstance(expr_, BinaryExpression)
        ]

        # create dict using expr names as keys
        expression_dict = {}
        for expr in expression_list:
            if expr.name not in expression_dict:
                expression_dict[expr.name] = []
            expression_dict[expr.name].append(expr)

        # create filter
        filter_request = cls()

        # datasets
        if "dataset_names" in expression_dict:
            filter_request.dataset_names = [
                expr.value for expr in expression_dict["dataset_names"]
            ]
        if "dataset_metadata" in expression_dict:
            filter_request.dataset_metadata = {
                expr.key: ValueFilter(
                    value=expr.value,
                    operator=expr.operator,
                )
                for expr in expression_dict["dataset_metadata"]
            }
        if "dataset_geospatial" in expression_dict:
            filter_request.dataset_geospatial = [
                GeospatialFilter(
                    value=expr.value,
                    operator=expr.operator,
                )
                for expr in expression_dict["dataset_geospatial"]
            ]
        # models
        if "models_names" in expression_dict:
            filter_request.models_names = [
                expr.value for expr in expression_dict["models_names"]
            ]
        if "models_metadata" in expression_dict:
            filter_request.models_metadata = {
                expr.key: ValueFilter(
                    value=expr.value,
                    operator=expr.operator,
                )
                for expr in expression_dict["models_metadata"]
            }
        if "model_geospatial" in expression_dict:
            filter_request.model_geospatial = [
                GeospatialFilter(
                    value=expr.value,
                    operator=expr.operator,
                )
                for expr in expression_dict["model_geospatial"]
            ]
        # datums
        if "datum_uids" in expression_dict:
            filter_request.datum_uids = [
                expr.value for expr in expression_dict["datum_uids"]
            ]
        if "datum_metadata" in expression_dict:
            filter_request.datum_metadata = {
                expr.key: ValueFilter(
                    value=expr.value,
                    operator=expr.operator,
                )
                for expr in expression_dict["datum_metadata"]
            }
        if "datum_geospatial" in expression_dict:
            filter_request.datum_geospatial = [
                GeospatialFilter(
                    value=expr.value,
                    operator=expr.operator,
                )
                for expr in expression_dict["datum_geospatial"]
            ]

        # annotations
        if "task_types" in expression_dict:
            filter_request.task_types = [
                expr.value for expr in expression_dict["task_types"]
            ]
        if "annotation_types" in expression_dict:
            filter_request.annotation_types = [
                expr.value for expr in expression_dict["annotation_types"]
            ]
        if "annotation_geometric_area" in expression_dict:
            filter_request.annotation_geometric_area = [
                ValueFilter(
                    value=expr.value,
                    operator=expr.operator,
                )
                for expr in expression_dict["annotation_geometric_area"]
            ]
        if "annotation_metadata" in expression_dict:
            filter_request.annotation_metadata = {
                expr.key: ValueFilter(
                    value=expr.value,
                    operator=expr.operator,
                )
                for expr in expression_dict["annotation_metadata"]
            }
        if "annotation_geospatial" in expression_dict:
            filter_request.annotation_geospatial = [
                GeospatialFilter(
                    value=expr.value,
                    operator=expr.operator,
                )
                for expr in expression_dict["annotation_geospatial"]
            ]
        # predictions
        if "prediction_scores" in expression_dict:
            filter_request.prediction_scores = [
                ValueFilter(
                    value=expr.value,
                    operator=expr.operator,
                )
                for expr in expression_dict["prediction_scores"]
            ]

        # labels
        if "label_ids" in expression_dict:
            filter_request.label_ids = [
                expr.value for expr in expression_dict["label_ids"]
            ]
        if "label_keys" in expression_dict:
            filter_request.label_keys = [
                expr.value for expr in expression_dict["label_keys"]
            ]
        if "label_values" in expression_dict:
            filter_request.label_values = [
                expr.value for expr in expression_dict["label_values"]
            ]

        return filter_request
