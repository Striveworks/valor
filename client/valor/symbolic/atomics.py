import datetime
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Optional, Set, Union

import numpy as np

from valor import coretypes
from valor.symbolic.modifiers import (
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
        return type(value) in {bool, Bool}


class Integer(Quantifiable):
    
    @staticmethod
    def supports(value: Any) -> bool:
        return type(value) in {int, np.integer, Integer}
    

class Float(Quantifiable):
    
    @staticmethod
    def supports(value: Any) -> bool:
        return type(value) in {float, np.floating, Float}


class String(Equatable):
    
    @staticmethod
    def supports(value: Any) -> bool:
        return type(value) in {str, String}


class Label(Equatable):
    
    @staticmethod
    def supports(value: Any) -> bool:
        return type(value) in {coretypes.Label, Label}

    @staticmethod
    def encode(value: coretypes.Label):
        return Label(value={value.key: value.value})

    def decode(self) -> Any:
        if type(self._value) is not dict:
            return self._value
        k, v = list(self._value.items())[0]
        return coretypes.Label(key=k, value=v)


class DateTime(Quantifiable):
    @staticmethod
    def supports(value: Any) -> bool:
        return type(value) in {datetime.datetime, str, DateTime}

    @staticmethod
    def encode(value: datetime.datetime):
        return DateTime(value=value.isoformat())

    def decode(self) -> Any:
        if type(self._value) is not str:
            return self._value
        return datetime.datetime.fromisoformat(self._value)


class Date(Quantifiable):
    @staticmethod
    def supports(value: Any) -> bool:
        return type(value) in {datetime.date, str, Date}

    @staticmethod
    def encode(value: datetime.date):
        return Date(value=value.isoformat())

    def decode(self) -> Any:
        if type(self._value) is not str:
            return self._value
        return datetime.date.fromisoformat(self._value)


class Time(Quantifiable):
    @staticmethod
    def supports(value: Any) -> bool:
        return type(value) in {datetime.time, str, Time}

    @staticmethod
    def encode(value: datetime.time):
        return Time(value=value.isoformat())

    def decode(self) -> Any:
        if type(self._value) is not str:
            return self._value
        return datetime.time.fromisoformat(self._value)


class Duration(Quantifiable):
    @staticmethod
    def supports(value: Any) -> bool:
        return type(value) in {datetime.timedelta, float, Duration}

    @staticmethod
    def encode(value: Any) -> Any:
        if type(value) is not datetime.timedelta:
            return value
        return Duration(value=value.total_seconds())

    def decode(self) -> Any:
        if type(self._value) is not float:
            return self._value
        return datetime.timedelta(seconds=self._value)
