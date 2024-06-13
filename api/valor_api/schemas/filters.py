from enum import Enum

from pydantic import BaseModel, ConfigDict, model_validator

from valor_api.enums import TaskType
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


class SupportedType(str, Enum):
    BOOLEAN = "boolean"
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    TASK_TYPE = "tasktype"
    DATETIME = "datetime"
    DATE = "date"
    TIME = "time"
    DURATION = "duration"
    POINT = "point"
    MULTIPOINT = "multipoint"
    LINESTRING = "linestring"
    MULTILINESTRING = "multilinestring"
    POLYGON = "polygon"
    BOX = "box"
    MULTIPOLYGON = "multipolygon"
    RASTER = "raster"
    GEOJSON = "geojson"
    EMBEDDING = "embedding"
    LABEL = "label"


map_type_to_validator = {
    SupportedType.BOOLEAN: validate_type_bool,
    SupportedType.STRING: validate_type_string,
    SupportedType.INTEGER: validate_type_integer,
    SupportedType.FLOAT: validate_type_float,
    SupportedType.DATETIME: validate_type_datetime,
    SupportedType.DATE: validate_type_date,
    SupportedType.TIME: validate_type_time,
    SupportedType.DURATION: validate_type_duration,
    SupportedType.POINT: validate_type_point,
    SupportedType.MULTIPOINT: validate_type_multipoint,
    SupportedType.LINESTRING: validate_type_linestring,
    SupportedType.MULTILINESTRING: validate_type_multilinestring,
    SupportedType.POLYGON: validate_type_polygon,
    SupportedType.BOX: validate_type_box,
    SupportedType.MULTIPOLYGON: validate_type_multipolygon,
    SupportedType.TASK_TYPE: validate_type_string,
    SupportedType.LABEL: None,
    SupportedType.EMBEDDING: None,
    SupportedType.RASTER: None,
}


class Symbol(str, Enum):
    DATASET_NAME = "dataset.name"
    DATASET_META = "dataset.metadata"
    MODEL_NAME = "model.name"
    MODEL_META = "model.metadata"
    DATUM_UID = "datum.uid"
    DATUM_META = "datum.metadata"
    ANNOTATION_META = "annotation.metadata"
    TASK_TYPE = "annotation.task_type"
    BOX = "annotation.bounding_box"
    POLYGON = "annotation.polygon"
    RASTER = "annotation.raster"
    EMBEDDING = "annotation.embedding"
    LABELS = "annotation.labels"
    LABEL_KEY = "label.key"
    LABEL_VALUE = "label.value"
    SCORE = "label.score"

    # 'area' attribute
    DATASET_META_AREA = "dataset.metadata.area"
    MODEL_META_AREA = "dataset.metadata.area"
    DATUM_META_AREA = "dataset.metadata.area"
    ANNOTATION_META_AREA = "dataset.metadata.area"
    BOX_AREA = "annotation.bounding_box.area"
    POLYGON_AREA = "annotation.polygon.area"
    RASTER_AREA = "annotation.raster.area"

    # api-only attributes
    DATASET_ID = "dataset.id"
    MODEL_ID = "model.id"
    DATUM_ID = "datum.id"
    ANNOTATION_ID = "annotation.id"
    GROUNDTRUTH_ID = "groundtruth.id"
    PREDICTION_ID = "prediction.id"
    LABEL_ID = "label.id"
    EMBEDDING_ID = "embedding.id"

    @property
    def type(self) -> SupportedType | None:
        """
        Get the type associated with a symbol.

        Returns
        -------
        SupportedType
            The supported type.

        Raises
        ------
        NotImplementedError
            If the symbol does not have a type defined.
        """
        map_symbol_to_type = {
            Symbol.DATASET_NAME: SupportedType.STRING,
            Symbol.MODEL_NAME: SupportedType.STRING,
            Symbol.DATUM_UID: SupportedType.STRING,
            Symbol.TASK_TYPE: SupportedType.TASK_TYPE,
            Symbol.BOX: SupportedType.BOX,
            Symbol.POLYGON: SupportedType.POLYGON,
            Symbol.EMBEDDING: SupportedType.EMBEDDING,
            Symbol.LABEL_KEY: SupportedType.STRING,
            Symbol.LABEL_VALUE: SupportedType.STRING,
            Symbol.SCORE: SupportedType.FLOAT,
            # 'area' attribue
            Symbol.DATASET_META_AREA: SupportedType.FLOAT,
            Symbol.MODEL_META_AREA: SupportedType.FLOAT,
            Symbol.DATUM_META_AREA: SupportedType.FLOAT,
            Symbol.ANNOTATION_META_AREA: SupportedType.FLOAT,
            Symbol.BOX_AREA: SupportedType.FLOAT,
            Symbol.POLYGON_AREA: SupportedType.FLOAT,
            Symbol.RASTER_AREA: SupportedType.FLOAT,
            # api-only
            Symbol.DATASET_ID: SupportedType.INTEGER,
            Symbol.MODEL_ID: SupportedType.INTEGER,
            Symbol.DATUM_ID: SupportedType.INTEGER,
            Symbol.ANNOTATION_ID: SupportedType.INTEGER,
            Symbol.GROUNDTRUTH_ID: SupportedType.INTEGER,
            Symbol.PREDICTION_ID: SupportedType.INTEGER,
            Symbol.LABEL_ID: SupportedType.INTEGER,
            Symbol.EMBEDDING_ID: SupportedType.INTEGER,
            # unsupported
            Symbol.DATASET_META: None,
            Symbol.MODEL_META: None,
            Symbol.DATUM_META: None,
            Symbol.ANNOTATION_META: None,
            Symbol.RASTER: None,
            Symbol.LABELS: None,
        }
        if self not in map_symbol_to_type:
            raise NotImplementedError(f"{self} is does not have a type.")
        return map_symbol_to_type[self]


class FilterOperator(str, Enum):
    EQ = "eq"
    NE = "ne"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    INTERSECTS = "intersects"
    INSIDE = "inside"
    OUTSIDE = "outside"
    CONTAINS = "contains"
    ISNULL = "isnull"
    ISNOTNULL = "isnotnull"


class LogicalOperator(str, Enum):
    AND = "and"
    OR = "or"
    NOT = "not"


class Value(BaseModel):
    """
    A typed value.

    Attributes
    ----------
    type : SupportedType
        The type of the value.
    value : bool | int | float | str | list | dict
        The stored value.
    """

    type: SupportedType
    value: bool | int | float | str | list | dict
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _validate_value(self):
        if self.type not in map_type_to_validator:
            raise TypeError(f"'{self.type}' is not a valid type.")
        map_type_to_validator[self.type](self.value)
        return self

    @classmethod
    def infer(
        cls,
        value: bool | int | float | str | TaskType,
    ):
        type_ = type(value)
        if type_ is bool:
            return cls(type=SupportedType.BOOLEAN, value=value)
        elif type_ is int:
            return cls(type=SupportedType.INTEGER, value=value)
        elif type_ is float:
            return cls(type=SupportedType.FLOAT, value=value)
        elif type_ is str:
            return cls(type=SupportedType.STRING, value=value)
        elif type_ is TaskType:
            return cls(type=SupportedType.TASK_TYPE, value=value)
        else:
            raise TypeError(
                f"Type inference is not supported for type '{type_}'."
            )


class Condition(BaseModel):
    lhs: Symbol
    lhs_key: str | None = None
    rhs: Value | None = None
    rhs_key: str | None = None
    op: FilterOperator
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _validate_object(self):

        # validate operator
        match self.op:
            case (
                FilterOperator.EQ
                | FilterOperator.NE
                | FilterOperator.GT
                | FilterOperator.GTE
                | FilterOperator.LT
                | FilterOperator.LTE
                | FilterOperator.INTERSECTS
                | FilterOperator.INSIDE
                | FilterOperator.OUTSIDE
                | FilterOperator.CONTAINS
            ):
                if self.rhs is None:
                    raise ValueError("TODO")
            case (FilterOperator.ISNULL | FilterOperator.ISNOTNULL):
                if self.rhs is not None:
                    raise ValueError("TODO")
            case _:
                raise NotImplementedError(
                    f"Filter operator '{self.op}' is not implemented."
                )

        return self


class LogicalFunction(BaseModel):
    args: "Condition | LogicalFunction | list[Condition] | list[LogicalFunction] | list[Condition | LogicalFunction]"
    op: LogicalOperator


FunctionType = Condition | LogicalFunction


class Filter(BaseModel):
    """
    Filter schema that stores filters as logical trees under tables.

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
    model_config = ConfigDict(extra="forbid")


def soft_and(
    items: list[FunctionType] | list[FunctionType | None],
) -> FunctionType:
    conditions = [item for item in items if item is not None]
    if len(items) > 1:
        return LogicalFunction(
            args=conditions,
            op=LogicalOperator.AND,
        )
    elif len(items) == 1:
        return conditions[0]
    else:
        raise ValueError("Passed an empty list.")


def soft_or(
    items: list[FunctionType] | list[FunctionType | None],
) -> FunctionType:
    conditions = [item for item in items if item is not None]
    if len(items) > 1:
        return LogicalFunction(
            args=conditions,
            op=LogicalOperator.OR,
        )
    elif len(items) == 1:
        return conditions[0]
    else:
        raise ValueError("Passed an empty list.")
