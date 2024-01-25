import datetime
from dataclasses import dataclass, is_dataclass, asdict
from enum import Enum
from typing import List, Union, Any, Optional

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

    def _create_expression(self, value: Any, operator: str) -> BinaryExpression:
        self._validate(value=value, operator=operator)
        return BinaryExpression(
            name=self.name,
            key=self.key,
            constraint=Constraint(
                value=value,
                operator=operator,  
            )
        )
    
    def _validate(self, value: Any, operator: str):
        if operator not in {"==", "!="}:
            raise ValueError(f"When using undefined types only equality operators are allowed.")
    
    def is_none(self):
        return self._create_expression(None, "is_none")
        
    def exists(self) -> BinaryExpression:
        return self._create_expression(None, operator="exists")

    def __eq__(self, value: Any) -> BinaryExpression:
        return self._create_expression(value, "==")

    def __ne__(self, value: Any) -> BinaryExpression:
        return self._create_expression(value, "!=")

    def __lt__(self, value: Any) -> BinaryExpression:
        return self._create_expression(value, "<")

    def __gt__(self, value: Any) -> BinaryExpression:
        return self._create_expression(value, ">")

    def __le__(self, value: Any) -> BinaryExpression:
        return self._create_expression(value, "<=")

    def __ge__(self, value: Any) -> BinaryExpression:
        return self._create_expression(value, ">=")
    
    def contains(self, value: Any) -> BinaryExpression:
        return self._create_expression(value, "contains")
    
    def intersect(self, value: Any) -> BinaryExpression:
        return self._create_expression(value, "intersect")

    def inside(self, value: Any) -> BinaryExpression:
        return self._create_expression(value, "inside")

    def outside(self, value: Any) -> BinaryExpression:
        return self._create_expression(value, "outside")
    
    def in_(self, values: List[Any]) -> List[BinaryExpression]:
        if not isinstance(values, list):
            raise TypeError("`in_` takes a list as input.")
        return [
            self == value 
            for value in values
        ]
     

@dataclass(eq=False)
class NumericMapper(_DeclarativeMapper):

    def _validate(self, value: Any, operator: str):
        if operator not in {"==", "!=", ">=", "<=", ">", "<"}:
            raise ValueError(f"Numeric values do not support the operation given by `{operator}`.")
        
        vtype = type(value)
        if vtype not in [int, float]:
            raise TypeError(f"NumericMapper does not support object type `{type(value)}`.")
    

@dataclass(eq=False)
class StringMapper(_DeclarativeMapper):
    
    def _validate(self, value: Any, operator: str):
        if operator not in {"==", "!="}:
            raise ValueError(f"String values do not support the operation given by `{operator}`.")
        if not isinstance(value, str):
            raise TypeError(f"StringMapper does not support object type `{type(value)}`.")


@dataclass(eq=False)
class DatetimeMapper(_DeclarativeMapper):

    def _create_expression(self, value: Any, operator: str) -> BinaryExpression:
        if operator not in {">", "<", ">=", "<=", "==", "!="}:
            raise ValueError(f"String values do not support the operation given by `{operator}`.")
        
        vtype = type(value)
        if vtype is datetime.datetime:
            value = {"datetime": value.isoformat()}
        elif vtype is datetime.date:
            value = {"date": value.isoformat()}
        elif vtype is datetime.time:
            value = {"time": value.isoformat()}
        elif vtype is datetime.timedelta:
            value = {"duration": str(value.total_seconds())}
        else:
            raise TypeError(f"DatetimeMapper does not support object type `{type(value)}`.")
        
        return super()._create_expression(value, operator)
        
    def _validate(self, value: Any, operator: str):
        pass


@dataclass(eq=False)
class _SpatialMapper(_DeclarativeMapper):

    def __post_init__(self):
        self.area = NumericMapper(name=f"{self.name}_area")

    def _validate(self, value: Any, operator: str):
        """Validate the inputs to a spatial filter."""

        if operator not in {"is_none", "exists", "contains", "inside", "outside", "intersect"}:
            raise ValueError(f"Geometric values do not support the operation given by `{operator}`.")


@dataclass(eq=False)
class GeometryMapper(_SpatialMapper):

    def _validate(self, value: GeometryType, operator: str):
        super()._validate(value, operator)

        if operator in {"is_none", "exists"}:
            return

        vtype = type(value)
        if vtype not in [
            Point,
            BoundingBox,
            Polygon,
            MultiPolygon,
            Raster,
        ]:
            raise TypeError(f"GeometryMapper does not support objects of type `{type(value)}`.")
        
        raise NotImplementedError(f"Geometric types only support 'is_none' and 'exists'. Support for other spatial operators is planned.")            


@dataclass(eq=False)
class GeospatialMapper(_SpatialMapper):

    def _validate(self, value: GeometryType, operator: str):
        super()._validate(value, operator)

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
        

@dataclass(eq=False)
class _DictionaryValueMapper(_DeclarativeMapper):

    def _create_expression(self, value: Any, operator: str) -> BinaryExpression:
        if self.key is None:
            raise ValueError("Attribute `key` is required for `_DictionaryValueMapper`.")
        
        if operator in {"is_none", "exists"}:
            return BinaryExpression(
                name=self.name,
                key=self.key,
                constraint=Constraint(
                    value=None,
                    operator=operator,
                )
            )
        
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
            raise NotImplementedError(f"Value type `{type(value)}` is currently unsuppoerted.")


@dataclass(eq=False)
class DictionaryMapper(_DeclarativeMapper):
    name: str

    def __getitem__(self, key: str):
        return _DictionaryValueMapper(self.name, key)        
        
    def _create_expression(self, value: Any, operator: str) -> None:
        return
    
    def _validate(self, value: Any, operator: str):
        return
        

@dataclass(eq=False)
class LabelMapper(_DeclarativeMapper):

    def _create_expression(self, value: Any, operator: str) -> BinaryExpression:
        self._validate(value, operator)

        if is_dataclass(value):
            value = asdict(value)

        if not isinstance(value, dict):
            raise TypeError("Label must be a `dict` or `dataclass` that contains `key` and `value` attributes.")
            
        if not set(value.keys()).issuperset({"key", "value"}):
            raise KeyError("Label must contain `key` and `value` keys.")
        elif type(value["key"]) is not str:
            raise ValueError("Label key must be of type `str`.")
        elif type(value["value"]) is not str:
            raise ValueError("Label value must be of type `str`.")
        
        value = {
            value["key"] : value["value"]
        }

        return BinaryExpression(
            name=self.name,
            key=self.key,
            constraint=Constraint(
                value=value,
                operator=operator,  
            )
        )

    def _validate(self, value: Any, operator: str):
        if operator not in {"==", "!="}:
            raise ValueError("Labels only support equality comparisons.")