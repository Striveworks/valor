import numpy as np

from typing import Any, Optional

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


class Variable:
    def __init__(
        self,
        value: Optional[Any] = None,
        name: Optional[str] = None,
        key: Optional[str] = None,
        attribute: Optional[str] = None,
    ):
        if value is not None:
            self.validate_type(value)
        self._value = value if value else None
        self._name = type(self).__name__.lower() if not name else name
        self._key = key
        self._attribute = attribute

    def __repr__(self) -> str:
        ret = type(self).__name__
        if self._value:
            ret += f"(value={self._value.__repr__()})"
        else:
            ret += f"(name={self._name}"
            if self._key:
                ret += f", key={self._key}"
            if self._attribute:
                ret += f", attribute={self._attribute}"
            ret += ")"
        return ret

    def __str__(self) -> str:
        if self._value:
            return str(self.value)
        else:
            ret = self._name
            if self._key:
                ret += f"[{self._key}]"
            if self._attribute:
                ret += f".{self._attribute}"
            return ret

    @staticmethod
    def supports(value: Any) -> bool:
        return False

    @classmethod
    def validate_type(cls, value: Any):
        if not cls.supports(value):
            raise TypeError(
                f"Value with `{str(value)}` with type `{type(value)}` is not in supported types."
            )

    @classmethod
    def encode(cls, value: Any) -> Any:
        cls.validate_type(value)
        return value

    def decode(self) -> Any:
        if self._value is None:
            raise ValueError("Value does not exist.")
        return self._value

    def is_symbolic(self):
        return self._value is None

    def is_value(self):
        return self._value is not None

    def to_dict(self):
        ret = {}
        if self.is_symbolic():
            ret["symbol"] = self._name
            if self._key:
                ret["key"] = self._key
            if self._attribute:
                ret["attribute"] = self._attribute
        else:
            ret[self._name] = self._value
        return ret

    @property
    def value(self):
        return self.decode()


class Equatable(Variable):
    def __eq__(self, other: Any):
        return Eq(self, self.encode(other))

    def __ne__(self, other: Any):
        return Ne(self, self.encode(other))


class Quantifiable(Equatable):
    def __gt__(self, other: Any):
        return Gt(self, self.encode(other))

    def __ge__(self, other: Any):
        return Ge(self, self.encode(other))

    def __lt__(self, other: Any):
        return Lt(self, self.encode(other))

    def __le__(self, other: Any):
        return Le(self, self.encode(other))


class Nullable(Variable):
    def is_none(self):
        return IsNull(self)

    def is_not_none(self):
        return IsNotNull(self)


class Spatial(Variable):

    def intersects(self, other: Any):
        return Intersects(self, self.encode(other))

    def inside(self, other: Any):
        return Inside(self, self.encode(other))

    def outside(self, other: Any):
        return Outside(self, self.encode(other))
