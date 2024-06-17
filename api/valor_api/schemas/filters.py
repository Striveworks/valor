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


class SupportedSymbol(str, Enum):
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


map_type_to_operators = {
    SupportedType.BOOLEAN: {FilterOperator.EQ, FilterOperator.NE},
    SupportedType.STRING: {FilterOperator.EQ, FilterOperator.NE},
    SupportedType.INTEGER: {
        FilterOperator.EQ,
        FilterOperator.NE,
        FilterOperator.GT,
        FilterOperator.GTE,
        FilterOperator.LT,
        FilterOperator.LTE,
    },
    SupportedType.FLOAT: {
        FilterOperator.EQ,
        FilterOperator.NE,
        FilterOperator.GT,
        FilterOperator.GTE,
        FilterOperator.LT,
        FilterOperator.LTE,
    },
    SupportedType.DATETIME: {
        FilterOperator.EQ,
        FilterOperator.NE,
        FilterOperator.GT,
        FilterOperator.GTE,
        FilterOperator.LT,
        FilterOperator.LTE,
    },
    SupportedType.DATE: {
        FilterOperator.EQ,
        FilterOperator.NE,
        FilterOperator.GT,
        FilterOperator.GTE,
        FilterOperator.LT,
        FilterOperator.LTE,
    },
    SupportedType.TIME: {
        FilterOperator.EQ,
        FilterOperator.NE,
        FilterOperator.GT,
        FilterOperator.GTE,
        FilterOperator.LT,
        FilterOperator.LTE,
    },
    SupportedType.DURATION: {
        FilterOperator.EQ,
        FilterOperator.NE,
        FilterOperator.GT,
        FilterOperator.GTE,
        FilterOperator.LT,
        FilterOperator.LTE,
    },
    SupportedType.POINT: {
        FilterOperator.INTERSECTS,
        FilterOperator.INSIDE,
        FilterOperator.OUTSIDE,
    },
    SupportedType.MULTIPOINT: {
        FilterOperator.INTERSECTS,
        FilterOperator.INSIDE,
        FilterOperator.OUTSIDE,
    },
    SupportedType.LINESTRING: {
        FilterOperator.INTERSECTS,
        FilterOperator.INSIDE,
        FilterOperator.OUTSIDE,
    },
    SupportedType.MULTILINESTRING: {
        FilterOperator.INTERSECTS,
        FilterOperator.INSIDE,
        FilterOperator.OUTSIDE,
    },
    SupportedType.POLYGON: {
        FilterOperator.INTERSECTS,
        FilterOperator.INSIDE,
        FilterOperator.OUTSIDE,
    },
    SupportedType.BOX: {
        FilterOperator.INTERSECTS,
        FilterOperator.INSIDE,
        FilterOperator.OUTSIDE,
    },
    SupportedType.MULTIPOLYGON: {
        FilterOperator.INTERSECTS,
        FilterOperator.INSIDE,
        FilterOperator.OUTSIDE,
    },
    SupportedType.TASK_TYPE: {FilterOperator.EQ, FilterOperator.NE},
    SupportedType.LABEL: {FilterOperator.CONTAINS},
    SupportedType.EMBEDDING: {},
    SupportedType.RASTER: {},
}


class LogicalOperator(str, Enum):
    AND = "and"
    OR = "or"
    NOT = "not"


class Symbol(BaseModel):
    """
    A symbolic value.

    Attributes
    ----------
    name : str
        The name of the symbol.
    key : str, optional
        Optional dictionary key if the symbol is representing a dictionary value.
    """

    name: SupportedSymbol
    key: str | None = None

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
            SupportedSymbol.DATASET_NAME: SupportedType.STRING,
            SupportedSymbol.MODEL_NAME: SupportedType.STRING,
            SupportedSymbol.DATUM_UID: SupportedType.STRING,
            SupportedSymbol.TASK_TYPE: SupportedType.TASK_TYPE,
            SupportedSymbol.BOX: SupportedType.BOX,
            SupportedSymbol.POLYGON: SupportedType.POLYGON,
            SupportedSymbol.EMBEDDING: SupportedType.EMBEDDING,
            SupportedSymbol.LABEL_KEY: SupportedType.STRING,
            SupportedSymbol.LABEL_VALUE: SupportedType.STRING,
            SupportedSymbol.SCORE: SupportedType.FLOAT,
            # 'area' attribue
            SupportedSymbol.DATASET_META_AREA: SupportedType.FLOAT,
            SupportedSymbol.MODEL_META_AREA: SupportedType.FLOAT,
            SupportedSymbol.DATUM_META_AREA: SupportedType.FLOAT,
            SupportedSymbol.ANNOTATION_META_AREA: SupportedType.FLOAT,
            SupportedSymbol.BOX_AREA: SupportedType.FLOAT,
            SupportedSymbol.POLYGON_AREA: SupportedType.FLOAT,
            SupportedSymbol.RASTER_AREA: SupportedType.FLOAT,
            # api-only
            SupportedSymbol.DATASET_ID: SupportedType.INTEGER,
            SupportedSymbol.MODEL_ID: SupportedType.INTEGER,
            SupportedSymbol.DATUM_ID: SupportedType.INTEGER,
            SupportedSymbol.ANNOTATION_ID: SupportedType.INTEGER,
            SupportedSymbol.GROUNDTRUTH_ID: SupportedType.INTEGER,
            SupportedSymbol.PREDICTION_ID: SupportedType.INTEGER,
            SupportedSymbol.LABEL_ID: SupportedType.INTEGER,
            SupportedSymbol.EMBEDDING_ID: SupportedType.INTEGER,
            # unsupported
            SupportedSymbol.DATASET_META: None,
            SupportedSymbol.MODEL_META: None,
            SupportedSymbol.DATUM_META: None,
            SupportedSymbol.ANNOTATION_META: None,
            SupportedSymbol.RASTER: None,
            SupportedSymbol.LABELS: None,
        }
        if self.name not in map_symbol_to_type:
            raise NotImplementedError(f"{self.name} is does not have a type.")
        return map_symbol_to_type[self.name]


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

    def supports_operator(self, op: FilterOperator):
        """
        Validates whether value type supports operator.

        Parameters
        ----------
        op : FilterOperator
            The operator to validate.

        Raises
        ------
        TypeError
            If the type does not support this operation.
        """
        return

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
    rhs: Value | None = None
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
                    raise ValueError(
                        f"Operator '{self.op}' requires a rhs value."
                    )
                elif self.rhs.type not in map_type_to_operators:
                    raise ValueError(
                        f"Value type '{self.rhs.type}' does not support operator '{self.op}'."
                    )
            case (FilterOperator.ISNULL | FilterOperator.ISNOTNULL):
                if self.rhs is not None:
                    raise ValueError(
                        f"Operator '{self.op}' does not support a rhs value."
                    )
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
    conditions: list[FunctionType] | list[FunctionType | None],
) -> FunctionType:
    """
    Performs an AND operation if more than one element exists.

    Parameters
    ----------
    conditions : list[FunctionType | None]
        A list of conditions or functions.

    Returns
    -------
    FunctionType
    """
    items = [condition for condition in conditions if condition is not None]
    if len(items) > 1:
        return LogicalFunction(
            args=items,
            op=LogicalOperator.AND,
        )
    elif len(items) == 1:
        return items[0]
    else:
        raise ValueError("Passed an empty list.")


def soft_or(
    conditions: list[FunctionType] | list[FunctionType | None],
) -> FunctionType:
    """
    Performs an OR operation if more than one element exists.

    Parameters
    ----------
    conditions : list[FunctionType | None]
        A list of conditions or functions.

    Returns
    -------
    FunctionType
    """
    items = [condition for condition in conditions if condition is not None]
    if len(items) > 1:
        return LogicalFunction(
            args=items,
            op=LogicalOperator.OR,
        )
    elif len(items) == 1:
        return items[0]
    else:
        raise ValueError("Passed an empty list.")
