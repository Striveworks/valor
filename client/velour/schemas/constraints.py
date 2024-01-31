import datetime
from dataclasses import dataclass
from typing import Any, List, Optional, Set, Union, Type

import numpy as np

from velour.enums import TaskType
from velour.schemas.geometry import (
    BoundingBox,
    MultiPolygon,
    Point,
    Polygon,
    Raster,
)
from velour.types import GeometryType, GeoJSONType, MetadataType, MetadataValueType


def getter_factory(name: str, type_: type):
    def _getter(self) -> type_:
        return getattr(self, name)
    return _getter


def setter_factory(name: str, type_: type):
    def _setter(self, __value: type_):
        setattr(self, name, __value)
    return _setter


@dataclass
class Constraint:
    """
    Represents a constraint with a value and an operator.

    Attributes:
        value : Any
            The value associated with the constraint.
        operator : str
            The operator used to define the constraint.
    """

    value: Any
    operator: str


@dataclass
class BinaryExpression:
    """
    Stores a conditional relationship.

    Attributes
    ----------
    name : str
        The name of the object the expression is operating over.
    constraint : velour.schemas.Constraint
        The operation that is performed.
    key : str, optional
        An optional key used for object retrieval.
    """

    name: str
    constraint: Constraint
    key: Union[str, None] = None


class _DeclarativeMapper(property):
    """
    Base class for constructing mapping objects.

    Parameters
    ----------
    property_name : str
        The name of the targeted attribute.
    filter_name : str, optional
        The name of the filter the expression is targeting. Only defined if different from `attribute_name`.
    key : str, optional
        An optional key used for object retrieval.
    """

    def __init__(
        self,
        property_name: str,
        filter_name: Optional[str] = None,
        key: Optional[str] = None,
        prefix: str = "_",
    ):
        self.property_name = property_name
        self.attribute_name = f"{prefix}{property_name}"
        self.filter_name = filter_name if filter_name else property_name
        self.key = key
        self.valid_operators = {}

        # set up `property`
        fget = lambda _ : self
        fset = None
        if not self.key:
            fset = lambda instance, value : self.__convert_to_value(instance, value)
        super().__init__(fget=fget, fset=fset)

    def __convert_to_value(self, instance, value: Any):
        fget = getter_factory(self.attribute_name, type_=self._type())
        fset = setter_factory(self.attribute_name, type_=self._type())
        super().__init__(fget=fget, fset=fset)
        setattr(instance, self.attribute_name, value)

    def _type(self) -> type:
        return Any

    def _valid_operators(self) -> Set[str]:
        """
        Set of valid operators.

        Overload in subclasses with the following:
        >>> valid_operators = {op1, op2}
        >>> return super()._valid_operators().union(valid_operators)
        """
        return set()

    def _create_expression(
        self, value: Any, operator: str
    ) -> BinaryExpression:
        """Called by operator overload functions."""
        if operator not in self._valid_operators():
            raise AttributeError(
                f"Mapper with type `{type(self)}` does not suppoert the `{operator}` operator."
            )

        self._validate(value=value, operator=operator)
        value = self._modify(value=value, operator=operator)
        return BinaryExpression(
            name=self.filter_name,
            key=self.key,
            constraint=Constraint(
                value=value,
                operator=operator,
            ),
        )

    def _validate(self, value: Any, operator: str) -> None:
        """Overload in subclasses to insert a validator."""
        pass

    def _modify(self, value: Any, operator: str) -> Any:
        """Overload in subclasses to insert a value modification."""
        return value

    def __eq__(self, value: Any) -> BinaryExpression:
        raise AttributeError(
            f"'{type(self)}' object has no attribute '=='"
        )

    def __ne__(self, value: Any) -> BinaryExpression:
        raise AttributeError(
            f"'{type(self)}' object has no attribute '!='"
        )

    def __lt__(self, value: Any) -> BinaryExpression:
        raise AttributeError(
            f"'{type(self)}' object has no attribute '<'"
        )

    def __gt__(self, value: Any) -> BinaryExpression:
        raise AttributeError(
            f"'{type(self)}' object has no attribute '>'"
        )

    def __le__(self, value: Any) -> BinaryExpression:
        raise AttributeError(
            f"'{type(self)}' object has no attribute '<='"
        )

    def __ge__(self, value: Any) -> BinaryExpression:
        raise AttributeError(
            f"'{type(self)}' object has no attribute '>='"
        )


class _NullableMapper(_DeclarativeMapper):
    """
    Defines a mapping object that handles nullable values.

    This object maps conditional expressions into `BinaryExpression` objects.

    Attributes
    ----------
    name : str
        The name of the object the expression is operating over.
    key : str, optional
        An optional key used for object retrieval.
    """

    def _valid_operators(self) -> Set[str]:
        valid_operators = {"is_none", "exists"}
        return super()._valid_operators().union(valid_operators)

    def is_none(self) -> BinaryExpression:
        return self._create_expression(None, "is_none")

    def exists(self) -> BinaryExpression:
        return self._create_expression(None, operator="exists")


class _EquatableMapper(_DeclarativeMapper):
    """
    Defines a mapping object that handles equatable values.

    This object maps conditional expressions into `BinaryExpression` objects.

    Attributes
    ----------
    name : str
        The name of the object the expression is operating over.
    key : str, optional
        An optional key used for object retrieval.
    """

    def _valid_operators(self) -> Set[str]:
        valid_operators = {"==", "!="}
        return super()._valid_operators().union(valid_operators)

    def __eq__(self, value: Any) -> BinaryExpression:
        return self._create_expression(value, "==")

    def __ne__(self, value: Any) -> BinaryExpression:
        return self._create_expression(value, "!=")

    def in_(self, values: List[Any]) -> List[BinaryExpression]:
        if not isinstance(values, list):
            raise TypeError("`in_` takes a list as input.")
        return [self == value for value in values]


class _QuantifiableMapper(_EquatableMapper):
    """
    Defines a mapping object that handles quantifiable values.

    This object maps conditional expressions into `BinaryExpression` objects.

    Attributes
    ----------
    name : str
        The name of the object the expression is operating over.
    key : str, optional
        An optional key used for object retrieval.
    """

    def _valid_operators(self) -> Set[str]:
        valid_operators = {">", "<", ">=", "<=", "==", "!="}
        return super()._valid_operators().union(valid_operators)

    def __lt__(self, value: Any) -> BinaryExpression:
        return self._create_expression(value, "<")

    def __gt__(self, value: Any) -> BinaryExpression:
        return self._create_expression(value, ">")

    def __le__(self, value: Any) -> BinaryExpression:
        return self._create_expression(value, "<=")

    def __ge__(self, value: Any) -> BinaryExpression:
        return self._create_expression(value, ">=")


class BoolMapper(_EquatableMapper):
    """
    Defines a mapping object that handles boolean values.

    This object maps conditional expressions into `BinaryExpression` objects.

    Attributes
    ----------
    name : str
        The name of the object the expression is operating over.
    key : str, optional
        An optional key used for object retrieval.
    """
    def _type(self) -> type:
        return bool

    def _validate(self, value: Any, operator: str) -> None:
        if not isinstance(value, bool):
            raise TypeError(
                f"BoolMapper does not support object type `{type(value)}`."
            )


class StringMapper(_EquatableMapper):
    """
    Defines a mapping object that handles string values.

    This object maps conditional expressions into `BinaryExpression` objects.

    Attributes
    ----------
    name : str
        The name of the object the expression is operating over.
    key : str, optional
        An optional key used for object retrieval.
    """
    def _type(self) -> type:
        return str
    
    def _validate(self, value: Any, operator: str) -> None:
        if not isinstance(value, str):
            raise TypeError(
                f"StringMapper does not support object type `{type(value)}`."
            )


class NumericMapper(_QuantifiableMapper):
    """
    Defines a mapping object that handles numeric values.

    This object maps conditional expressions into `BinaryExpression` objects.

    Attributes
    ----------
    name : str
        The name of the object the expression is operating over.
    key : str, optional
        An optional key used for object retrieval.
    """
    def _type(self) -> type:
        return Union[int, float, np.floating]
    
    def _validate(self, value: Any, operator: str) -> None:
        if type(value) not in [int, float, np.floating]:
            raise TypeError(
                f"NumericMapper does not support object type `{type(value)}`."
            )


class DatetimeMapper(_QuantifiableMapper):
    """
    Defines a mapping object that handles `datetime` objects.

    This object maps conditional expressions into `BinaryExpression` objects.

    Attributes
    ----------
    name : str
        The name of the object the expression is operating over.
    key : str, optional
        An optional key used for object retrieval.
    """
    def _type(self) -> type:
        return Union[datetime.datetime, datetime.date, datetime.time, datetime.timedelta]
    
    def _modify(self, value: Any, operator: str) -> Any:
        vtype = type(value)
        if vtype is datetime.datetime:
            return {"datetime": value.isoformat()}
        elif vtype is datetime.date:
            return {"date": value.isoformat()}
        elif vtype is datetime.time:
            return {"time": value.isoformat()}
        elif vtype is datetime.timedelta:
            return {"duration": str(value.total_seconds())}
        else:
            raise TypeError(
                f"DatetimeMapper does not support object type `{type(value)}`."
            )


class _SpatialMapper(_NullableMapper):
    """
    Defines a mapping object that handles spatial values.

    This object maps conditional expressions into `BinaryExpression` objects.

    Attributes
    ----------
    name : str
        The name of the object the expression is operating over.
    key : str, optional
        An optional key used for object retrieval.
    """

    def _valid_operators(self) -> Set[str]:
        valid_operators = {"contains", "inside", "outside", "intersect"}
        return super()._valid_operators().union(valid_operators)

    def contains(self, value: Any) -> BinaryExpression:
        return self._create_expression(value, "contains")

    def intersect(self, value: Any) -> BinaryExpression:
        return self._create_expression(value, "intersect")

    def inside(self, value: Any) -> BinaryExpression:
        return self._create_expression(value, "inside")

    def outside(self, value: Any) -> BinaryExpression:
        return self._create_expression(value, "outside")
    
    @property
    def area(self):
        return NumericMapper(self.property_name, f"{self.filter_name}_area")


class GeometryMapper(_SpatialMapper):
    """
    Defines a mapping object that handles `velour.schemas.geometry` objects.

    This object maps conditional expressions into `BinaryExpression` objects.

    Attributes
    ----------
    name : str
        The name of the object the expression is operating over.
    key : str, optional
        An optional key used for object retrieval.
    """
    def _type(self) -> type:
        return Union[BoundingBox, Polygon, MultiPolygon, Raster]
    
    def _validate(self, value: GeometryType, operator: str):

        if operator in {"is_none", "exists"}:
            return

        if type(value) not in [
            Point,
            BoundingBox,
            Polygon,
            MultiPolygon,
            Raster,
        ]:
            raise TypeError(
                f"GeometryMapper does not support objects of type `{type(value)}`."
            )

        raise NotImplementedError(
            "Geometric types only support 'is_none' and 'exists'. Support for other spatial operators is planned."
        )


class GeospatialMapper(_SpatialMapper):
    """
    Defines a mapping object that handles GeoJSON.

    This object maps conditional expressions into `BinaryExpression` objects.

    Attributes
    ----------
    name : str
        The name of the object the expression is operating over.
    key : str, optional
        An optional key used for object retrieval.
    """
    def _type(self) -> type:
        return GeoJSONType

    def _validate(self, value: GeometryType, operator: str):

        if operator not in {"inside", "outside", "intersect"}:
            raise NotImplementedError(
                "Geospatial types only support 'inside', 'outside' and 'intersect'. Support for other spatial operators is planned."
            )

        if not isinstance(value, dict):
            raise TypeError(
                "Geospatial values should be a GeoJSON-style dictionary containing the keys `type` and `coordinates`."
            )
        elif not value.get("type") or not value.get("coordinates"):
            raise KeyError(
                "Geospatial values should be a GeoJSON-style dictionary containing the keys `type` and `coordinates`."
            )


class _DictionaryValueMapper(_NullableMapper, _QuantifiableMapper):
    """
    Defines a mapping object that handles arbitrary objects.

    This object maps conditional expressions into `BinaryExpression` objects.

    Attributes
    ----------
    name : str
        The name of the object the expression is operating over.
    key : str, optional
        An optional key used for object retrieval.
    """
    def _type(self) -> type:
        return MetadataValueType

    def _create_expression(self, value: Any, operator: str) -> Any:
        if self.key is None:
            raise ValueError(
                "Attribute `key` is required for `_DictionaryValueMapper`."
            )

        if operator in {"is_none", "exists"} and value is None:
            return BinaryExpression(
                name=self.filter_name,
                key=self.key,
                constraint=Constraint(
                    value=None,
                    operator=operator,
                ),
            )

        # direct value to appropriate mapper (if it exists)
        vtype = type(value)
        if vtype is bool:
            return BoolMapper(self.property_name, filter_name=self.filter_name, key=self.key)._create_expression(
                value, operator
            )
        if vtype is str:
            return StringMapper(self.property_name, self.filter_name, self.key)._create_expression(
                value, operator
            )
        elif vtype in [int, float]:
            return NumericMapper(self.property_name, self.filter_name, self.key)._create_expression(
                value, operator
            )
        elif vtype in [
            datetime.datetime,
            datetime.date,
            datetime.time,
            datetime.timedelta,
        ]:
            return DatetimeMapper(self.property_name, self.filter_name, self.key)._create_expression(
                value, operator
            )
        else:
            raise NotImplementedError(
                f"Dictionary value with type `{type(value)}` is not suppoerted."
            )


class DictionaryMapper(_DeclarativeMapper):
    """
    Defines a dictionary mapping object that captures the key used to access the data..

    This object returns a `_DictionaryValueMapper` when given a key.

    Attributes
    ----------
    name : str
        The name of the object the expression is operating over.
    key : str, optional
        An optional key used for object retrieval.

    Examples
    --------
    >>> DictionaryMapper("name")["some_key"] == True
    BinaryExpression(name='name', constraint=Constraint(value=True, operator='=='), key='some_key')
    """
    def _type(self) -> type:
        return MetadataType

    def __getitem__(self, key: str):
        return _DictionaryValueMapper(self.property_name, self.filter_name, key)

    def _create_expression(self, value: Any, operator: str) -> None:
        raise NotImplementedError(
            "Dictionary mapper does not define any operations for iteself. Please use `dict[key]` to create an expression."
        )


class LabelMapper(_EquatableMapper):
    """
    Defines a mapping object that handles `velour.Label` objects.

    This object maps conditional expressions into `BinaryExpression` objects.

    Attributes
    ----------
    name : str
        The name of the object the expression is operating over.
    key : str, optional
        An optional key used for object retrieval.
    """
    
    def _modify(self, value: Any, operator: str) -> Any:
        # convert to dict
        if hasattr(value, "to_dict"):
            value = value.to_dict()

        # validate dict
        if not isinstance(value, dict):
            raise TypeError(
                "Label must be a `dict` or `velour.Label` that contains `key` and `value` attributes."
            )
        elif not set(value.keys()).issuperset({"key", "value"}):
            raise KeyError("Label must contain `key` and `value` keys.")
        elif type(value["key"]) is not str:
            raise ValueError("Label key must be of type `str`.")
        elif type(value["value"]) is not str:
            raise ValueError("Label value must be of type `str`.")

        return {value["key"]: value["value"]}

if __name__ == "__main__":
    class Test:
        score = DictionaryMapper("score", filter_name="prediction_scores")

        def __init__(self, score):
            self.score = score
            
    # expr = [
    #     Test.score > 1,
    #     Test.score > 1.0,
    #     Test.score > 1,
    # ]     
    # # print(expr)

    print(Test.score._type())

    print(Test.score["some_key"] == 123)
    print(Test.score["some_key"].is_none())
            
    t = Test({"key":"k1", "value":"k2"})

    print(t.score)

    # expr = [
    #     Test.score > 1,
    #     Test.score > 1.0,
    #     Test.score > 1,
    # ]     
    # print(expr)



