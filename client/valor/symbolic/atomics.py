import datetime
from typing import Any, Dict, List

import numpy as np

from valor.symbolic.modifiers import Equatable, Quantifiable, Variable


class Bool(Equatable):
    """
    Bool wrapper.
    """

    @classmethod
    def supports(cls, value: Any) -> bool:
        return type(value) is bool


class Integer(Quantifiable):
    @classmethod
    def supports(cls, value: Any) -> bool:
        return type(value) in {int, np.integer}


class Float(Quantifiable):
    @classmethod
    def supports(cls, value: Any) -> bool:
        return type(value) in {float, np.floating} or Integer.supports(value)


class String(Equatable):
    @classmethod
    def supports(cls, value: Any) -> bool:
        return type(value) is str


class DateTime(Quantifiable):
    @classmethod
    def supports(cls, value: Any) -> bool:
        return type(value) is datetime.datetime

    @classmethod
    def encode(cls, value: datetime.datetime):
        return DateTime(value=value.isoformat())

    def decode(self) -> Any:
        if not isinstance(self._value, str):
            return self._value
        return datetime.datetime.fromisoformat(self._value)


class Date(Quantifiable):
    @classmethod
    def supports(cls, value: Any) -> bool:
        return type(value) is datetime.date

    @classmethod
    def encode(cls, value: datetime.date):
        return Date(value=value.isoformat())

    def decode(self) -> Any:
        if not isinstance(self._value, str):
            return self._value
        return datetime.date.fromisoformat(self._value)


class Time(Quantifiable):
    @classmethod
    def supports(cls, value: Any) -> bool:
        return type(value) is datetime.time

    @classmethod
    def encode(cls, value: datetime.time):
        return Time(value=value.isoformat())

    def decode(self) -> Any:
        if not isinstance(self._value, str):
            return self._value
        return datetime.time.fromisoformat(self._value)


class Duration(Quantifiable):
    @classmethod
    def supports(cls, value: Any) -> bool:
        return type(value) is datetime.timedelta

    @classmethod
    def encode(cls, value: Any) -> Any:
        return Duration(value=value.total_seconds())

    def decode(self) -> Any:
        if not isinstance(self._value, float):
            return self._value
        return datetime.timedelta(seconds=self._value)


class ValueList(Equatable):
    @staticmethod
    def supported_type():
        return Variable

    @classmethod
    def supports(cls, value: List[Any]) -> bool:
        return type(value) is list and all(
            [
                type(element) is cls.supported_type() and element.is_value()
                for element in value
            ]
        )

    @classmethod
    def encode(cls, value: List[Variable]):
        return [element.encode(element) for element in value]

    def decode(self):
        if self._value is None:
            return None
        if issubclass(self.supported_type(), Variable):
            return [
                self.supported_type().decode(element)
                for element in self._value
            ]

    def __getitem__(self, __key: int):
        if self.is_symbolic():
            raise ValueError("Variable is a symbol.")
        elif value := self.get_value():
            return value[__key]
        else:
            raise ValueError(f"Variable has value '{None}'")

    def __setitem__(self, __key: int, __value: Any):
        if self.is_symbolic():
            raise ValueError("Variable is a symbol.")
        elif self._value:
            if isinstance(__value, self.supported_type()):
                self._value[__key] = __value
            else:
                raise ValueError("values is not supported")
        else:
            raise ValueError(f"Variable has value '{None}'")


class StaticCollection(Equatable):
    def __init__(self, **kwargs):
        if len(kwargs) == 1:
            value = list(kwargs.values())[0]
        else:
            attribute_names = {
                name
                for name in vars(type(self))
                if issubclass(type(self.__getattribute__(name)), Variable)
            }
            kwargs_keys = set(kwargs.keys())
            if kwargs_keys.issubset(attribute_names):
                for name in attribute_names:
                    self.__setattr__(
                        name,
                        type(self.__getattribute__(name)).definite(
                            kwargs[name] if name in kwargs else None
                        ),
                    )
                value = self.encode(self)
            else:
                raise TypeError(
                    f"{type(self)}() does not take the following keyword arguments '{attribute_names - kwargs_keys}'"
                )

        super().__init__(value=value)

    @staticmethod
    def search_for_values(obj):
        if issubclass(type(obj), StaticCollection):
            retval = dict()
            for name in vars(obj):
                if value := StaticCollection.search_for_values(
                    obj.__getattribute__(name)
                ):
                    retval[name] = value
            return retval
        elif issubclass(type(obj), Variable) and obj.is_value():
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

    @classmethod
    def supports(cls, value: Any) -> bool:
        return type(value) is dict

    @classmethod
    def encode(cls, value: Variable):
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
