import json
import datetime
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Optional, Set, Dict, Union

import numpy as np

from valor.symbolic.modifiers import (
    Symbol,
    Value,
    Equatable,
    Quantifiable,
    Spatial,
)


class Bool(Equatable):
    """
    Bool wrapper.
    """
    
    @staticmethod
    def supports(value: Any) -> bool:
        return type(value) is bool


class Integer(Quantifiable):
    
    @staticmethod
    def supports(value: Any) -> bool:
        return type(value) in {int, np.integer}
    

class Float(Quantifiable):
    
    @staticmethod
    def supports(value: Any) -> bool:
        return (
            type(value) in {float, np.floating}
            or Integer.supports(value)
        )


class String(Equatable):
    
    @staticmethod
    def supports(value: Any) -> bool:
        return type(value) is str
        

class DateTime(Quantifiable):

    @staticmethod
    def supports(value: Any) -> bool:
        return type(value) is datetime.datetime

    @staticmethod
    def encode(value: datetime.datetime):
        return DateTime(value=value.isoformat())

    def decode(self) -> Any:
        if not isinstance(self._value, str):
            return self._value
        return datetime.datetime.fromisoformat(self._value)


class Date(Quantifiable):

    @staticmethod
    def supports(value: Any) -> bool:
        return type(value) is datetime.date

    @staticmethod
    def encode(value: datetime.date):
        return Date(value=value.isoformat())

    def decode(self) -> Any:
        if not isinstance(self._value, str):
            return self._value
        return datetime.date.fromisoformat(self._value)


class Time(Quantifiable):

    @staticmethod
    def supports(value: Any) -> bool:
        return type(value) is datetime.time

    @staticmethod
    def encode(value: datetime.time):
        return Time(value=value.isoformat())

    def decode(self) -> Any:
        if not isinstance(self._value, str):
            return self._value
        return datetime.time.fromisoformat(self._value)


class Duration(Quantifiable):

    @staticmethod
    def supports(value: Any) -> bool:
        return type(value) is datetime.timedelta

    @staticmethod
    def encode(value: Any) -> Any:
        return Duration(value=value.total_seconds())

    def decode(self) -> Any:
        if not isinstance(self._value, float):
            return self._value
        return datetime.timedelta(seconds=self._value)


class StaticCollection(Equatable):
        
    @staticmethod
    def search_for_values(obj):
        if issubclass(type(obj), StaticCollection):
            retval = dict()
            for name in vars(obj):
                if value := StaticCollection.search_for_values(obj.__getattribute__(name)):
                    retval[name] = value
            return retval
        elif issubclass(type(obj), Value) and obj.is_value():
            return obj._value
        elif type(obj) is list:
            retval = list()
            for element in obj:
                if value := StaticCollection.search_for_values(element):
                    retval.append(value)
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        if self.is_symbolic():
            return super().to_dict()
        ret = self.search_for_values(self)
        if not isinstance(ret, dict):
            raise ValueError  # This is just to avoid typing errors.
        return ret
    
    @staticmethod
    def supports(value: Any) -> bool:
        return type(value) is dict
    
    @staticmethod
    def encode(value: Value):
        return StaticCollection.search_for_values(value)
        
    def decode(self):
        return self
    
    def __repr__(self):
        if self.is_symbolic():
            return super().__repr__()
        return self.encode(self).__repr__()
    
    def __str__(self):
        if self.is_symbolic():
            return super().__str__()
        return str(self.encode(self))

