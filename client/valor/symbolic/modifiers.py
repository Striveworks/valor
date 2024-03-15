from typing import Any, Optional

import numpy as np

from valor.symbolic.functions import (
    Eq,
    Ge,
    Gt,
    Inside,
    Intersects,
    IsNotNull,
    IsNull,
    Le,
    Lt,
    Ne,
    Outside,
)


class Symbol:
    def __init__(
        self,
        name: str,
        key: Optional[str] = None,
        attribute: Optional[str] = None,
    ):
        self._name = name
        self._key = key
        self._attribute = attribute

    def __repr__(self):
        ret = type(self).__name__
        ret += f"(name={self._name}"
        if self._key:
            ret += f", key={self._key}"
        if self._attribute:
            ret += f", attribute={self._attribute}"
        ret += ")"
        return ret

    def __str__(self):
        ret = self._name
        if self._key is not None:
            ret += f"[{self._key}]"
        if self._attribute:
            ret += f".{self._attribute}"
        return ret

    def to_dict(self):
        ret = {}
        ret["symbol"] = self._name
        if self._key:
            ret["key"] = self._key
        if self._attribute:
            ret["attribute"] = self._attribute
        return ret


class Value:
    def __init__(
        self,
        value: Any,
    ):
        if value is not None:
            self.validate_type(value)
        self._value = value if value else None

    @classmethod
    def definite(
        cls,
        value: Any,
    ):
        if not cls.supports(value):
            raise TypeError
        return cls(value)

    @classmethod
    def symbolic(
        cls,
        name: Optional[str] = None,
        key: Optional[str] = None,
        attribute: Optional[str] = None,
    ):
        name = cls.__name__.lower() if not name else name
        return cls(
            value=Symbol(
                name=name,
                key=key,
                attribute=attribute,
            )
        )

    def __repr__(self) -> str:
        return self._value.__repr__()

    def __str__(self) -> str:
        return str(self._value)

    @staticmethod
    def supports(value: Any) -> bool:
        raise NotImplementedError

    @classmethod
    def validate_type(cls, value: Any):
        if type(value) not in {cls, Symbol} and not cls.supports(value):
            raise TypeError(
                f"Value with `{str(value)}` with type `{type(value)}` is not in supported types."
            )

    @classmethod
    def encode(cls, value: Any) -> Any:
        if type(value) in {cls, Symbol}:
            return value
        elif cls.supports(value):
            return cls(value)
        raise TypeError

    def decode(self) -> Any:
        return self._value

    def is_symbolic(self):
        return type(self._value) is Symbol

    def is_value(self):
        return not self.is_symbolic()

    def to_dict(self):
        if type(self._value) is Symbol:
            return self._value.to_dict()
        else:
            return {
                "type": type(self).__name__.lower(),
                "value": self._value,
            }

    def get_value(self):
        if type(self._value) is not Symbol:
            return self.decode()

    def get_symbol(self) -> Symbol:
        if type(self._value) is not Symbol:
            raise ValueError
        return self._value


class Equatable(Value):
    def __eq__(self, value: Any):
        other = self.encode(value)
        if self.is_value() and other.is_value():
            return self._value == other._value
        return Eq(self, other)

    def __ne__(self, value: Any):
        other = self.encode(value)
        if self.is_value() and other.is_value():
            return self._value != other._value
        return Ne(self, other)


class Quantifiable(Equatable):
    def __gt__(self, value: Any):
        other = self.encode(value)
        if self.is_value() and other.is_value():
            return self._value > other._value
        return Gt(self, self.encode(other))

    def __ge__(self, value: Any):
        other = self.encode(value)
        if self.is_value() and other.is_value():
            return self._value >= other._value
        return Ge(self, self.encode(other))

    def __lt__(self, value: Any):
        other = self.encode(value)
        if self.is_value() and other.is_value():
            return self._value < other._value
        return Lt(self, self.encode(other))

    def __le__(self, value: Any):
        other = self.encode(value)
        if self.is_value() and other.is_value():
            return self._value <= other._value
        return Le(self, self.encode(other))


class Nullable(Value):
    def is_none(self):
        return IsNull(self)

    def is_not_none(self):
        return IsNotNull(self)


class Spatial(Value):
    def intersects(self, other: Any):
        return Intersects(self, self.encode(other))

    def inside(self, other: Any):
        return Inside(self, self.encode(other))

    def outside(self, other: Any):
        return Outside(self, self.encode(other))
