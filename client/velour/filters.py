from dataclasses import dataclass
from enum import Enum
from typing import List, Union

from velour.enums import AnnotationType
from velour.schemas.filters import (
    AnnotationFilter,
    DatasetFilter,
    DatumFilter,
    Filter,
    GeometricAnnotationFilter,
    KeyValueFilter,
    LabelFilter,
    ModelFilter,
    NumericFilter,
    PredictionFilter,
    StringFilter,
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

    def __getitem__(self, key: str):
        return DeclarativeMapper(
            name=self.name,
            object_type=self.object_type,
            key=key,
        )

    def __eq__(self, __value: object) -> BinaryExpression:
        if not isinstance(__value, self.object_type):
            raise ValueError(
                f"`{self.name}` should be of type `{self.object_type}`"
            )
        return BinaryExpression(
            name=self.name,
            key=self.key,
            value=__value,
            operator="==",
        )

    def __ne__(self, __value: object) -> BinaryExpression:
        if not isinstance(__value, self.object_type):
            raise ValueError(
                f"`{self.name}` should be of type `{self.object_type}`"
            )
        return BinaryExpression(
            name=self.name,
            key=self.key,
            value=__value,
            operator="!=",
        )

    def __lt__(self, __value: object) -> BinaryExpression:
        if isinstance(__value, str):
            raise TypeError("`__lt__` does not support type `str`")
        if not isinstance(__value, self.object_type):
            raise ValueError(
                f"`{self.name}` should be of type `{self.object_type}`"
            )
        return BinaryExpression(
            name=self.name,
            key=self.key,
            value=__value,
            operator="<",
        )

    def __gt__(self, __value: object) -> BinaryExpression:
        if isinstance(__value, str):
            raise TypeError("`__gt__` does not support type `str`")
        if not isinstance(__value, self.object_type):
            raise ValueError(
                f"`{self.name}` should be of type `{self.object_type}`"
            )
        return BinaryExpression(
            name=self.name,
            key=self.key,
            value=__value,
            operator=">",
        )

    def __le__(self, __value: object) -> BinaryExpression:
        if isinstance(__value, str):
            raise TypeError("`__le__` does not support type `str`")
        if not isinstance(__value, self.object_type):
            raise ValueError(
                f"`{self.name}` should be of type `{self.object_type}`"
            )
        return BinaryExpression(
            name=self.name,
            key=self.key,
            value=__value,
            operator="<=",
        )

    def __ge__(self, __value: object) -> BinaryExpression:
        if isinstance(__value, str):
            raise TypeError("`__ge__` does not support type `str`")
        if not isinstance(__value, self.object_type):
            raise ValueError(
                f"`{self.name}` should be of type `{self.object_type}`"
            )
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


class Geometry:
    def __init__(self, annotation_type: AnnotationType):
        self.area = DeclarativeMapper(
            "annotation.area", Union[int, float], key=annotation_type
        )


def create_filter(expressions: List[BinaryExpression]) -> Filter:
    """
    Parses a list of `BinaryExpression` to create a `schemas.Filter` object.
    """

    # expand nested expressions
    expressions = [
        expr for expr in expressions if isinstance(expr, BinaryExpression)
    ] + [
        expr_
        for expr in expressions
        if isinstance(expr, list)
        for expr_ in expr
        if isinstance(expr_, BinaryExpression)
    ]

    # parse filters into highest level categories
    keys = {"dataset", "model", "datum", "annotation", "prediction", "label"}
    filters = {
        key: [expr for expr in expressions if key in expr.name] for key in keys
    }

    #
    filter_request = Filter()

    # parse dataset filters
    if filters["dataset"]:
        filter_request.datasets = DatasetFilter()
        for expr in filters["dataset"]:
            if "id" in expr.name:
                filter_request.datasets.ids.append(expr.value)
            elif "name" in expr.name:
                filter_request.datasets.names.append(expr.value)
            elif "metadata" in expr.name:
                if isinstance(expr.value, str):
                    filter_request.datasets.metadata.append(
                        KeyValueFilter(
                            key=expr.key,
                            comparison=StringFilter(
                                value=expr.value,
                                operator=expr.operator,
                            ),
                        )
                    )
                elif isinstance(expr.value, Union[int, float]):
                    filter_request.datasets.metadata.append(
                        KeyValueFilter(
                            key=expr.key,
                            comparison=NumericFilter(
                                value=expr.value,
                                operator=expr.operator,
                            ),
                        )
                    )
                else:
                    raise NotImplementedError(
                        f"Metadatum value with type `{type(expr.value)}` is not currently supported."
                    )
            elif "geo" in expr.names:
                raise NotImplementedError(
                    "Geospatial filters are not currently supported."
                )
            else:
                raise RuntimeError(f"Unknown input: `{expr}`")

    # parse model filters
    if filters["model"]:
        filter_request.models = ModelFilter()
        for expr in filters["model"]:
            if "id" in expr.name:
                filter_request.models.ids.append(expr.value)
            elif "name" in expr.name:
                filter_request.models.names.append(expr.value)
            elif "metadata" in expr.name:
                if isinstance(expr.value, str):
                    filter_request.models.metadata.append(
                        KeyValueFilter(
                            key=expr.key,
                            comparison=StringFilter(
                                value=expr.value,
                                operator=expr.operator,
                            ),
                        )
                    )
                elif isinstance(expr.value, Union[int, float]):
                    filter_request.models.metadata.append(
                        KeyValueFilter(
                            key=expr.key,
                            comparison=NumericFilter(
                                value=expr.value,
                                operator=expr.operator,
                            ),
                        )
                    )
                else:
                    raise NotImplementedError(
                        f"Metadatum value with type `{type(expr.value)}` is not currently supported."
                    )
            elif "geo" in expr.names:
                raise NotImplementedError(
                    "Geospatial filters are not currently supported."
                )
            else:
                raise RuntimeError(f"Unknown input: `{expr}`")

    # parse datum filters
    if filters["datum"]:
        filter_request.datums = DatumFilter()
        for expr in filters["datum"]:
            if ".id" in expr.name:
                filter_request.datums.ids.append(expr.value)
            elif ".uid" in expr.name:
                filter_request.datums.uids.append(expr.value)
            elif ".metadata" in expr.name:
                if isinstance(expr.value, str):
                    filter_request.datums.metadata.append(
                        KeyValueFilter(
                            key=expr.key,
                            comparison=StringFilter(
                                value=expr.value,
                                operator=expr.operator,
                            ),
                        )
                    )
                elif isinstance(expr.value, Union[int, float]):
                    filter_request.datums.metadata.append(
                        KeyValueFilter(
                            key=expr.key,
                            comparison=NumericFilter(
                                value=expr.value,
                                operator=expr.operator,
                            ),
                        )
                    )
                else:
                    raise NotImplementedError(
                        f"Metadatum value with type `{type(expr.value)}` is not currently supported."
                    )
            elif ".geo" in expr.names:
                raise NotImplementedError(
                    "Geospatial filters are not currently supported."
                )
            else:
                raise RuntimeError(f"Unknown input: `{expr}`")

    # parse annotation filters
    if filters["annotation"]:
        filter_request.annotations = AnnotationFilter()
        for expr in filters["annotation"]:
            if "id" in expr.name:
                filter_request.annotations.ids.append(expr.value)
            elif "task_type" in expr.name:
                filter_request.annotations.task_types.append(expr.value)
            elif "annotation_type" in expr.name:
                filter_request.annotations.annotation_types.append(expr.value)
            elif "area" in expr.name:
                filter_request.annotations.geometry.append(
                    GeometricAnnotationFilter(
                        annotation_type=expr.key,
                        area=[
                            NumericFilter(
                                value=expr.value, operator=expr.operator
                            )
                        ],
                    )
                )
            elif "json" in expr.name:
                raise NotImplementedError
            elif "metadata" in expr.name:
                if isinstance(expr.value, str):
                    filter_request.annotations.metadata.append(
                        KeyValueFilter(
                            key=expr.key,
                            comparison=StringFilter(
                                value=expr.value,
                                operator=expr.operator,
                            ),
                        )
                    )
                elif isinstance(expr.value, Union[int, float]):
                    filter_request.annotations.metadata.append(
                        KeyValueFilter(
                            key=expr.key,
                            comparison=NumericFilter(
                                value=expr.value,
                                operator=expr.operator,
                            ),
                        )
                    )
                else:
                    raise NotImplementedError(
                        f"Metadatum value with type `{type(expr.value)}` is not currently supported."
                    )
            elif "geo" in expr.names:
                raise NotImplementedError(
                    "Geospatial filters are not currently supported."
                )
            else:
                raise RuntimeError(f"Unknown input: `{expr}`")

    # parse prediction filters
    if filters["prediction"]:
        filter_request.predictions = PredictionFilter()
        for expr in filters["prediction"]:
            if "score" in expr.name:
                filter_request.predictions.scores.append(
                    NumericFilter(value=expr.value, operator=expr.operator)
                )

    # parse label filters
    if filters["label"]:
        filter_request.labels = LabelFilter()
        for expr in filters["label"]:
            if "id" in expr.name:
                filter_request.labels.ids.append(expr.value)
            elif "key" in expr.name:
                filter_request.labels.keys.append(expr.value)
            elif "label" in expr.name:
                filter_request.labels.labels.append(
                    {
                        "key": expr.key,
                        "value": expr.value,
                        "score": None,
                    }
                )

    return filter_request
