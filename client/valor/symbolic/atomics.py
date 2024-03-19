import datetime
from enum import Enum
from typing import Any

import numpy as np

from valor.symbolic.modifiers import Equatable, Quantifiable


class Bool(Equatable):
    """
    Bool wrapper.
    """

    @classmethod
    def supports(cls, value: Any) -> bool:
        return isinstance(value, bool)


class Integer(Quantifiable):
    @classmethod
    def supports(cls, value: Any) -> bool:
        return isinstance(value, (int, np.integer))


class Float(Quantifiable):
    @classmethod
    def supports(cls, value: Any) -> bool:
        return isinstance(value, (float, np.floating)) or Integer.supports(
            value
        )


class String(Equatable):
    @classmethod
    def supports(cls, value: Any) -> bool:
        return isinstance(value, str)


class StringEnum(String):
    @classmethod
    def supports(cls, value: Any) -> bool:
        return issubclass(type(value), str) and issubclass(type(value), Enum)

    def encode(self):
        return self.get_value().value


class DateTime(Quantifiable):
    @classmethod
    def supports(cls, value: Any) -> bool:
        return type(value) is datetime.datetime

    @classmethod
    def decode(cls, value: str):
        return cls(value=datetime.datetime.fromisoformat(value))

    def encode(self):
        return self.get_value().isoformat()


class Date(Quantifiable):
    @classmethod
    def supports(cls, value: Any) -> bool:
        return type(value) is datetime.date

    @classmethod
    def decode(cls, value: str):
        return cls(value=datetime.date.fromisoformat(value))

    def encode(self):
        return self.get_value().isoformat()


class Time(Quantifiable):
    @classmethod
    def supports(cls, value: Any) -> bool:
        return type(value) is datetime.time

    @classmethod
    def decode(cls, value: str):
        return cls(value=datetime.time.fromisoformat(value))

    def encode(self):
        return self.get_value().isoformat()


class Duration(Quantifiable):
    @classmethod
    def supports(cls, value: Any) -> bool:
        return type(value) is datetime.timedelta

    @classmethod
    def decode(cls, value: int):
        return cls(value=datetime.timedelta(seconds=value))

    def encode(self):
        return self.get_value().total_seconds()
