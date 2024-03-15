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


class Variable:
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

    @classmethod
    def supports(cls, value: Any) -> bool:
        raise NotImplementedError

    @classmethod
    def validate_type(cls, value: Any):
        if type(value) not in {cls, Symbol} and not cls.supports(value):
            raise TypeError(
                f"Variable with `{str(value)}` of type `{type(value)}` is not supported."
            )

    @classmethod
    def encode(cls, value: Any) -> Any:
        return value

    def decode(self) -> Any:
        return self._value

    def preprocess(self, value: Any):
        if type(value) in {type(self), Symbol}:
            return value
        elif self.supports(value):
            return self.encode(value)
        raise TypeError

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


class Equatable(Variable):
    def __eq__(self, value: Any):
        other = self.preprocess(value)
        if self.is_value() and other.is_value():
            return self._value == other._value
        return Eq(self, other)

    def __ne__(self, value: Any):
        other = self.preprocess(value)
        if self.is_value() and other.is_value():
            return self._value != other._value
        return Ne(self, other)

    def __hash__(self):
        if self.is_symbolic():
            raise ValueError("Variable is symbolic.")
        return hash(str(self))


class Quantifiable(Equatable):
    def __gt__(self, value: Any):
        other = self.preprocess(value)
        if self.is_value() and other.is_value():
            return self._value > other._value
        return Gt(self, self.encode(other))

    def __ge__(self, value: Any):
        other = self.preprocess(value)
        if self.is_value() and other.is_value():
            return self._value >= other._value
        return Ge(self, self.encode(other))

    def __lt__(self, value: Any):
        other = self.preprocess(value)
        if self.is_value() and other.is_value():
            return self._value < other._value
        return Lt(self, self.encode(other))

    def __le__(self, value: Any):
        other = self.preprocess(value)
        if self.is_value() and other.is_value():
            return self._value <= other._value
        return Le(self, self.encode(other))


class Nullable(Variable):
    def is_none(self):
        return IsNull(self)

    def is_not_none(self):
        return IsNotNull(self)


class Spatial(Variable):
    def intersects(self, other: Any):
        return Intersects(self, self.preprocess(other))

    def inside(self, other: Any):
        return Inside(self, self.preprocess(other))

    def outside(self, other: Any):
        return Outside(self, self.preprocess(other))
