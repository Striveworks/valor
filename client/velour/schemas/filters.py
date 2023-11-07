from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Union

from velour.enums import AnnotationType, TaskType


@dataclass
class ValueFilter:
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
    operator: str = "=="

    def __post_init__(self):
        allowed_operators = [">", "<", ">=", "<=", "==", "!="]
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

    def _validate(self, value, exclude=None, name: str = None):
        # if isinstance(value, str):
        #     raise TypeError("f`{name}` does not support type `str`")
        if self.object_type == float and isinstance(value, int):
            return  # edge case
        if not isinstance(value, self.object_type):
            raise ValueError(
                f"`{self.name}` should be of type `{self.object_type}`"
            )

    def __getitem__(self, key: str):
        return DeclarativeMapper(
            name=self.name,
            object_type=self.object_type,
            key=key,
        )

    def __eq__(self, __value: object) -> BinaryExpression:
        self._validate(
            __value,
        )
        return BinaryExpression(
            name=self.name,
            key=self.key,
            value=__value,
            operator="==",
        )

    def __ne__(self, __value: object) -> BinaryExpression:
        self._validate(__value)
        return BinaryExpression(
            name=self.name,
            key=self.key,
            value=__value,
            operator="!=",
        )

    def __lt__(self, __value: object) -> BinaryExpression:

        self._validate(__value)
        return BinaryExpression(
            name=self.name,
            key=self.key,
            value=__value,
            operator="<",
        )

    def __gt__(self, __value: object) -> BinaryExpression:
        if isinstance(__value, str):
            raise TypeError("`__gt__` does not support type `str`")
        self._validate(__value)
        return BinaryExpression(
            name=self.name,
            key=self.key,
            value=__value,
            operator=">",
        )

    def __le__(self, __value: object) -> BinaryExpression:
        if isinstance(__value, str):
            raise TypeError("`__le__` does not support type `str`")
        self._validate(__value)
        return BinaryExpression(
            name=self.name,
            key=self.key,
            value=__value,
            operator="<=",
        )

    def __ge__(self, __value: object) -> BinaryExpression:
        if isinstance(__value, str):
            raise TypeError("`__ge__` does not support type `str`")
        self._validate(__value)
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


@dataclass
class Filter:
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
    labels: List[Dict[str, str]] = None
    label_ids: List[int] = None
    label_keys: List[str] = None

    @classmethod
    def create(cls, expressions: List[BinaryExpression]) -> "Filter":
        """
        Parses a list of `BinaryExpression` to create a `schemas.Filter` object.
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
            raise NotImplementedError

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
            raise NotImplementedError

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
            raise NotImplementedError

        # annotations
        if "task_types" in expression_dict:
            filter_request.task_types = [
                expr.value for expr in expression_dict["task_types"]
            ]
        if "annotation_types" in expression_dict:
            filter_request.annotation_types = [
                expr.value for expr in expression_dict["annotation_types"]
            ]
        if "annotation_geometic_area" in expression_dict:
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
            raise NotImplementedError

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
        if "labels" in expression_dict:
            filter_request.labels = [
                {expr.key: expr.value} for expr in expression_dict["labels"]
            ]
        if "label_keys" in expression_dict:
            filter_request.label_keys = [
                expr.value for expr in expression_dict["label_keys"]
            ]

        return filter_request
