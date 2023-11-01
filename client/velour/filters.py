from dataclasses import dataclass
from enum import Enum
from typing import Union

from velour import schemas
from velour.enums import AnnotationType, TaskType
from velour.schemas import (
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

    def in_(self, __values: list[object]) -> list[BinaryExpression]:
        if not isinstance(__values, list):
            raise TypeError("`in_` takes a list as input.")
        for value in __values:
            if not isinstance(value, self.object_type):
                raise TypeError(
                    f"All elements must be of type `{self.object_type}`, got '{type(value)}'"
                )
        return [self == value for value in __values]


class Metadata:
    def __init__(self, name: str):
        self.name = name

    def __getitem__(self, key: str):
        if not isinstance(key, str):
            raise TypeError("Metadata key must be of type `str`")
        return DeclarativeMapper(name=self.name, object_type=object, key=key)


class Geometry:
    def __init__(self, annotation_type: AnnotationType):
        self.area = DeclarativeMapper(
            "annotation.area", Union[int, float], key=annotation_type
        )


class JSON:
    def __init__(self, name: str):
        self.name = name
        self.key = DeclarativeMapper(name + ".key", str)

    def __getitem__(self, key: str):
        if not isinstance(key, str):
            raise TypeError("JSON key must be of type `str`")
        return DeclarativeMapper(name=self.name, key=key, object_type=object)


class _BaseLabel:
    def __getitem__(self, key: str):
        if not isinstance(key, str):
            raise TypeError("Label key must be of type `str`")
        return DeclarativeMapper(
            name="label.value", key=key, object_type=object
        )


class Dataset:
    id = DeclarativeMapper("dataset.id", int)
    name = DeclarativeMapper("dataset.name", str)
    metadata = Metadata("dataset.metadata")


class Model:
    id = DeclarativeMapper("model.id", int)
    name = DeclarativeMapper("model.name", str)
    metadata = Metadata("model.metadata")


class Datum:
    id = DeclarativeMapper("datum.id", int)
    uid = DeclarativeMapper("datum.uid", str)
    metadata = Metadata("datum.metadata")


class Annotation:
    task_type = DeclarativeMapper("annotation.task_type", TaskType)
    annotation_type = DeclarativeMapper(
        "annotation.annotation_type", AnnotationType
    )
    box = Geometry("box")
    polygon = Geometry("polygon")
    multipolygon = Geometry("multipolygon")
    raster = Geometry("raster")
    json = JSON("annotation.json")
    metadata = Metadata("annotation.metadata")


class Prediction:
    score = DeclarativeMapper("prediction.score", Union[int, float])


class Label:
    key = DeclarativeMapper("label.key", str)
    label = _BaseLabel()


def create_filter(expressions: list[BinaryExpression]) -> Filter:
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

    for expr in expressions:
        print(expr)

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
                    schemas.Label(key=expr.key, value=expr.value)
                )

    return filter_request


if __name__ == "__main__":

    expressions = [
        Dataset.id.in_([1, 2]),
        Dataset.name.in_(["world", "foo", "bar"]),
        Dataset.metadata["angle"] > 0.5,
        Label.key == "class",
        Label.label["class"] == "dog",
        Annotation.box.area >= 220,
        Annotation.box.area <= 1000,
        Datum.uid == "uid1",
        Datum.metadata["type"] == "image",
        Annotation.raster.area <= 10,
    ]

    # expressions = {
    #     "dataset_ids": [1],
    #     "dataset_names": ["world", "foo", "bar", "hello"],
    #     "dataset_metadata": {
    #         "angle": ">0.5"
    #     },
    #     "label_keys": ["class"],
    #     "labels": [("class", "dog")],
    #     "annotation_box_area": [">=200"],
    # }

    for expr in expressions:
        print(expr)
    print()

    import json
    from dataclasses import asdict

    f = create_filter(expressions)
    print(json.dumps(asdict(f), indent=4))
