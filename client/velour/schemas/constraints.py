import datetime
from dataclasses import dataclass, is_dataclass, asdict
from enum import Enum
from typing import List, Set, Union, Any, Optional, Type

from velour.types import GeometryType, GeoJSONType
from velour.schemas.geometry import (
    Point, 
    Polygon, 
    BoundingBox, 
    MultiPolygon, 
    Raster
)

@dataclass
class Constraint:
    value: Any
    operator: str


@dataclass
class BinaryExpression:
    name: str
    constraint: Constraint
    key: Union[str, Enum, None] = None


@dataclass
class _DeclarativeMapper:
    name: str
    key: Optional[str] = None

    def __post_init__(self):
        self.valid_operators = {}

    def _valid_operators(self) -> Set[str]:
        return set()

    def _create_expression(self, value: Any, operator: str) -> BinaryExpression:
        if operator not in self._valid_operators():
            raise AttributeError(f"Mapper with type `{type(self)}` does not suppoert the `{operator}` operator.")

        self._validate(value=value, operator=operator)
        value = self._modify(value=value, operator=operator)
        return BinaryExpression(
            name=self.name,
            key=self.key,
            constraint=Constraint(
                value=value,
                operator=operator,  
            )
        )
    
    def _validate(self, value: Any, operator: str) -> None:
        pass
        
    def _modify(self, value: Any, operator: str) -> Any:
        return value
    
    def __eq__(self, value: Any) -> BinaryExpression:
        raise AttributeError(f"Objects with type `{type(value)}` do not support the `==` operator.")

    def __ne__(self, value: Any) -> BinaryExpression:
        raise AttributeError(f"Objects with type `{type(value)}` do not support the `!=` operator.")
    
    def __lt__(self, value: Any) -> BinaryExpression:
        raise AttributeError(f"Objects with type `{type(value)}` do not support the `<` operator.")

    def __gt__(self, value: Any) -> BinaryExpression:
        raise AttributeError(f"Objects with type `{type(value)}` do not support the `>` operator.")

    def __le__(self, value: Any) -> BinaryExpression:
        raise AttributeError(f"Objects with type `{type(value)}` do not support the `<=` operator.")

    def __ge__(self, value: Any) -> BinaryExpression:
        raise AttributeError(f"Objects with type `{type(value)}` do not support the `>=` operator.")


class _NullableMapper(_DeclarativeMapper):
    def _valid_operators(self) -> Set[str]:
        valid_operators = {"is_none", "exists"}
        return super()._valid_operators().union(valid_operators)
    
    def is_none(self) -> BinaryExpression:
        return self._create_expression(None, "is_none")
        
    def exists(self) -> BinaryExpression:
        return self._create_expression(None, operator="exists")


class _EquatableMapper(_DeclarativeMapper):

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
        return [
            self == value 
            for value in values
        ]


class _QuantifiableMapper(_EquatableMapper):

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


class _SpatialMapper(_NullableMapper):

    def __post_init__(self):
        self.area = NumericMapper(name=f"{self.name}_area")

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
    

class StringMapper(_EquatableMapper):
    """
    Declarative mapper for use with `str` type values.
    """
    def _validate(self, value: Any, operator: str) -> None:
        if not isinstance(value, str):
            raise TypeError(f"StringMapper does not support object type `{type(value)}`.")


class LabelMapper(_EquatableMapper):
    """
    Declarative mapper for use with `velour.Label` type values.
    """
    def _modify(self, value: Any, operator: str) -> Any:
        # convert to dict
        if is_dataclass(value):
            value = asdict(value)
        
        # validate dict
        if not isinstance(value, dict):
            raise TypeError("Label must be a `dict` or `dataclass` that contains `key` and `value` attributes.")
        elif not set(value.keys()).issuperset({"key", "value"}):
            raise KeyError("Label must contain `key` and `value` keys.")
        elif type(value["key"]) is not str:
            raise ValueError("Label key must be of type `str`.")
        elif type(value["value"]) is not str:
            raise ValueError("Label value must be of type `str`.")
        
        return {
            value["key"] : value["value"]
        }
    

class NumericMapper(_QuantifiableMapper):
    """
    Declarative mapper for use with `int` and `float` type values.
    """
    def _validate(self, value: Any, operator: str) -> None:
        if type(value) not in [int, float]:
            raise TypeError(f"NumericMapper does not support object type `{type(value)}`.")


class DatetimeMapper(_QuantifiableMapper):
    """
    Declarative mapper for use with `datetime` objects.
    """
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
            raise TypeError(f"DatetimeMapper does not support object type `{type(value)}`.")


class GeometryMapper(_SpatialMapper):

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
            raise TypeError(f"GeometryMapper does not support objects of type `{type(value)}`.")
        
        raise NotImplementedError(f"Geometric types only support 'is_none' and 'exists'. Support for other spatial operators is planned.")


class GeospatialMapper(_SpatialMapper):

    def _validate(self, value: GeometryType, operator: str):

        if operator not in {"inside", "outside", "intersect"}:
            raise NotImplementedError(f"Geospatial types only support 'inside', 'outside' and 'intersect'. Support for other spatial operators is planned.")

        if not isinstance(value, dict):
            raise TypeError(
                "Geospatial values should be a GeoJSON-style dictionary containing the keys `type` and `coordinates`."
            )
        elif not value.get("type") or not value.get("coordinates"):
            raise KeyError(
                "Geospatial values should be a GeoJSON-style dictionary containing the keys `type` and `coordinates`."
            )
        

class _DictionaryValueMapper(_NullableMapper, _QuantifiableMapper):

    def _create_expression(self, value: Any, operator: str) -> Any:
        if self.key is None:
            raise ValueError("Attribute `key` is required for `_DictionaryValueMapper`.")

        if (
            operator in {"is_none", "exists"} 
            and value is None
        ):
            return BinaryExpression(
                name=self.name,
                key=self.key,
                constraint=Constraint(
                    value=None,
                    operator=operator,
                )
            )
        
        # direct value to appropriate mapper (if it exists)
        vtype = type(value)
        if vtype is str:
            return StringMapper(self.name, self.key)._create_expression(value, operator)
        elif vtype in [int, float]:
            return NumericMapper(self.name, self.key)._create_expression(value, operator)
        elif vtype in [
            datetime.datetime,
            datetime.date,
            datetime.time,
            datetime.timedelta,
        ]:
            return DatetimeMapper(self.name, self.key)._create_expression(value, operator)
        else:
            raise NotImplementedError(f"Dictionary value with type `{type(value)}` is not suppoerted.")
        

class DictionaryMapper(_DeclarativeMapper):

    def __getitem__(self, key: str):
        return _DictionaryValueMapper(self.name, key)        
        
    def _create_expression(self, value: Any, operator: str) -> None:
        raise NotImplementedError("Dictionary mapper does not define any operations for iteself. Please use `dict[key]` to create and expression.")