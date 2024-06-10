from pydantic import BaseModel, ConfigDict

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
    """
    A symbolic variable.

    Attributes
    ----------
    type : str
        The data type that this symbol represents.
    name : str
        The name of the symbol.
    key : str, optional
        Optional key to define dictionary access of a value.
    attribute : str, optional
        Optional attribute that modifies the underlying value.
    """

    type: str
    name: str
    key: str | None = None
    attribute: str | None = None


class Value(BaseModel):
    """
    A typed value.

    Attributes
    ----------
    type : str
        The type of the value.
    value : bool | int | float | str | list | dict
        The stored value.
    """

    type: str
    value: bool | int | float | str | list | dict
    model_config = ConfigDict(extra="forbid")


class Operands(BaseModel):
    """
    Function operands.

    Attributes
    ----------
    lhs : Symbol
        The symbol representing a table column this function should be applied to.
    rhs : Value
        A value to perform a comparison over.
    """

    lhs: Symbol
    rhs: Value
    model_config = ConfigDict(extra="forbid")


class And(BaseModel):
    """
    Logical function representing an AND operation.

    Attributes
    ----------
    logical_and : list[FunctionType]
        A list of functions to AND together.
    """

    logical_and: list["FunctionType"]
    model_config = ConfigDict(extra="forbid")

    @property
    def op(self) -> str:
        """Returns the operator name."""
        return type(self).__name__.lower()

    @property
    def args(self) -> list["FunctionType"]:
        """Returns the list of functional arguments."""
        return self.logical_and


class Or(BaseModel):
    """
    Logical function representing an OR operation.

    Attributes
    ----------
    logical_or : list[FunctionType]
        A list of functions to OR together.
    """

    logical_or: list["FunctionType"]
    model_config = ConfigDict(extra="forbid")

    @property
    def op(self) -> str:
        """Returns the operator name."""
        return type(self).__name__.lower()

    @property
    def args(self):
        """Returns the list of functional arguments."""
        return self.logical_or


class Not(BaseModel):
    """
    Logical function representing an OR operation.

    Attributes
    ----------
    logical_not : FunctionType
        A functions to logically negate.
    """

    logical_not: "FunctionType"
    model_config = ConfigDict(extra="forbid")

    @property
    def op(self) -> str:
        """Returns the operator name."""
        return type(self).__name__.lower()

    @property
    def arg(self):
        """Returns the functional argument."""
        return self.logical_not


class IsNull(BaseModel):
    """
    Checks if symbol represents a null value.

    Attributes
    ----------
    isnull : Symbol
        The symbolic argument.
    """

    isnull: Symbol
    model_config = ConfigDict(extra="forbid")

    @property
    def op(self) -> str:
        """Returns the operator name."""
        return type(self).__name__.lower()

    @property
    def arg(self):
        """Returns the symbolic argument."""
        return self.isnull


class IsNotNull(BaseModel):
    """
    Checks if symbol represents an existing value.

    Attributes
    ----------
    isnotnull : Symbol
        The symbolic argument.
    """

    isnotnull: Symbol
    model_config = ConfigDict(extra="forbid")

    @property
    def op(self) -> str:
        """Returns the operator name."""
        return type(self).__name__.lower()

    @property
    def arg(self):
        """Returns the symbolic argument."""
        return self.isnotnull


class Equal(BaseModel):
    """
    Checks if symbol is equal to a provided value.

    Attributes
    ----------
    eq : Operands
        The operands of the function.
    """

    eq: Operands
    model_config = ConfigDict(extra="forbid")

    @property
    def op(self) -> str:
        """Returns the operator name."""
        return type(self).__name__.lower()

    @property
    def lhs(self):
        """Returns the lhs operand."""
        return self.eq.lhs

    @property
    def rhs(self):
        """Returns the rhs operand."""
        return self.eq.rhs


class NotEqual(BaseModel):
    """
    Checks if symbol is not equal to a provided value.

    Attributes
    ----------
    ne : Operands
        The operands of the function.
    """

    ne: Operands
    model_config = ConfigDict(extra="forbid")

    @property
    def op(self) -> str:
        """Returns the operator name."""
        return type(self).__name__.lower()

    @property
    def lhs(self):
        """Returns the lhs operand."""
        return self.ne.lhs

    @property
    def rhs(self):
        """Returns the rhs operand."""
        return self.ne.rhs


class GreaterThan(BaseModel):
    """
    Checks if symbol is greater than a provided value.

    Attributes
    ----------
    gt : Operands
        The operands of the function.
    """

    gt: Operands
    model_config = ConfigDict(extra="forbid")

    @property
    def op(self) -> str:
        """Returns the operator name."""
        return type(self).__name__.lower()

    @property
    def lhs(self):
        """Returns the lhs operand."""
        return self.gt.lhs

    @property
    def rhs(self):
        """Returns the rhs operand."""
        return self.gt.rhs


class GreaterThanEqual(BaseModel):
    """
    Checks if symbol is greater than or equal to a provided value.

    Attributes
    ----------
    ge : Operands
        The operands of the function.
    """

    ge: Operands
    model_config = ConfigDict(extra="forbid")

    @property
    def op(self) -> str:
        """Returns the operator name."""
        return type(self).__name__.lower()

    @property
    def lhs(self):
        """Returns the lhs operand."""
        return self.ge.lhs

    @property
    def rhs(self):
        """Returns the rhs operand."""
        return self.ge.rhs


class LessThan(BaseModel):
    """
    Checks if symbol is less than a provided value.

    Attributes
    ----------
    lt : Operands
        The operands of the function.
    """

    lt: Operands
    model_config = ConfigDict(extra="forbid")

    @property
    def op(self) -> str:
        """Returns the operator name."""
        return type(self).__name__.lower()

    @property
    def lhs(self):
        """Returns the lhs operand."""
        return self.lt.lhs

    @property
    def rhs(self):
        """Returns the rhs operand."""
        return self.lt.rhs


class LessThanEqual(BaseModel):
    """
    Checks if symbol is less than or equal to a provided value.

    Attributes
    ----------
    le : Operands
        The operands of the function.
    """

    le: Operands
    model_config = ConfigDict(extra="forbid")

    @property
    def op(self) -> str:
        """Returns the operator name."""
        return type(self).__name__.lower()

    @property
    def lhs(self):
        """Returns the lhs operand."""
        return self.le.lhs

    @property
    def rhs(self):
        """Returns the rhs operand."""
        return self.le.rhs


class Intersects(BaseModel):
    """
    Checks if symbol intersects a provided value.

    Attributes
    ----------
    intersects : Operands
        The operands of the function.
    """

    intersects: Operands
    model_config = ConfigDict(extra="forbid")

    @property
    def op(self) -> str:
        """Returns the operator name."""
        return type(self).__name__.lower()

    @property
    def lhs(self):
        """Returns the lhs operand."""
        return self.intersects.lhs

    @property
    def rhs(self):
        """Returns the rhs operand."""
        return self.intersects.rhs


class Inside(BaseModel):
    """
    Checks if symbol is inside a provided value.

    Attributes
    ----------
    inside : Operands
        The operands of the function.
    """

    inside: Operands
    model_config = ConfigDict(extra="forbid")

    @property
    def op(self) -> str:
        """Returns the operator name."""
        return type(self).__name__.lower()

    @property
    def lhs(self):
        """Returns the lhs operand."""
        return self.inside.lhs

    @property
    def rhs(self):
        """Returns the rhs operand."""
        return self.inside.rhs


class Outside(BaseModel):
    """
    Checks if symbol is outside a provided value.

    Attributes
    ----------
    outside : Operands
        The operands of the function.
    """

    outside: Operands
    model_config = ConfigDict(extra="forbid")

    @property
    def op(self) -> str:
        """Returns the operator name."""
        return type(self).__name__.lower()

    @property
    def lhs(self):
        """Returns the lhs operand."""
        return self.outside.lhs

    @property
    def rhs(self):
        """Returns the rhs operand."""
        return self.outside.rhs


class Contains(BaseModel):
    """
    Checks if symbolic list contains a provided value.

    Attributes
    ----------
    contains : Operands
        The operands of the function.
    """

    contains: Operands
    model_config = ConfigDict(extra="forbid")

    @property
    def op(self) -> str:
        """Returns the operator name."""
        return type(self).__name__.lower()

    @property
    def lhs(self):
        """Returns the lhs operand."""
        return self.contains.lhs

    @property
    def rhs(self):
        """Returns the rhs operand."""
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
