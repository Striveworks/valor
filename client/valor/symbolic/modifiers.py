from typing import Any, List, Optional

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
    Or,
)


class Symbol:
    def __init__(
        self,
        name: str,
        key: Optional[str] = None,
        attribute: Optional[str] = None,
        owner: Optional[str] = None,
    ):
        self._owner = owner
        self._name = name
        self._key = key
        self._attribute = attribute

    def __repr__(self):
        ret = f"{type(self).__name__}("
        if self._owner:
            ret += f"owner='{self._owner}', "
        ret += f"name='{self._name}'"
        if self._key:
            ret += f", key='{self._key}'"
        if self._attribute:
            ret += f", attribute='{self._attribute}'"
        ret += ")"
        return ret

    def __str__(self):
        ret = ""
        if self._owner:
            ret += f"{self._owner}."
        ret += self._name
        if self._key is not None:
            ret += f"['{self._key}']"
        if self._attribute:
            ret += f".{self._attribute}"
        return ret

    def to_dict(self):
        return {
            "type": "symbol",
            "value": {
                "owner": self._owner,
                "name": self._name,
                "key": self._key,
                "attribute": self._attribute,
            },
        }


class Variable:
    def __init__(
        self,
        value: Optional[Any] = None,
        symbol: Optional[Symbol] = None,
    ):
        if (value is not None) and (symbol is not None):
            raise TypeError(
                f"{type(self).__name__} cannot be symbolic and contain a value at the same time."
            )
        elif symbol is not None and not isinstance(symbol, Symbol):
            raise TypeError(
                f"{type(self).__name__} symbol should have type 'Symbol' or be set to 'None'"
            )
        elif value is not None and not self.supports(value):
            raise TypeError(
                f"{type(self).__name__} value with type '{type(value).__name__}' is not supported."
            )
        self._value = symbol if symbol else value

    def __repr__(self) -> str:
        return self._value.__repr__()

    def __str__(self) -> str:
        return str(self._value)

    @classmethod
    def definite(
        cls,
        value: Any,
    ):
        """Initialize variable with a value."""
        if not cls.supports(value):
            raise TypeError(
                f"Value `{value}` with type `{type(value).__name__}` is not a supported type for `{cls.__name__}`"
            )
        return cls(value=value)

    @classmethod
    def symbolic(
        cls,
        name: Optional[str] = None,
        key: Optional[str] = None,
        attribute: Optional[str] = None,
        owner: Optional[str] = None,
    ):
        """Initialize variable as a symbol."""
        name = cls.__name__.lower() if not name else name
        return cls(
            symbol=Symbol(
                name=name,
                key=key,
                attribute=attribute,
                owner=owner,
            )
        )

    @classmethod
    def preprocess(cls, value: Any):
        """
        This method converts any type to an instance of the variable class.

        It will raise an error if a value is unsupported.

        Parameters
        ----------
        value : Any
            An instance of a variable, value, or symbol.

        Raises
        ------
        TypeError
            If a value or variable instance is of an incompatible type.
        """
        if isinstance(value, cls):
            return value
        elif isinstance(value, Symbol):
            return cls(symbol=value)
        elif cls.supports(value):
            return cls(value=value)
        raise TypeError("Incompatible type.")

    @classmethod
    def supports(cls, value: Any) -> bool:
        """Checks if value is a supported type."""
        raise NotImplementedError(
            f"Variable of type `{cls.__name__}` cannot be assigned a value."
        )

    @classmethod
    def decode_value(cls, value: Any):
        """Decode object value from JSON compatible dictionary."""
        return cls(value=value)

    def encode_value(self) -> Any:
        """Encode object value to JSON compatible dictionary."""
        return self.get_value()

    def to_dict(self) -> dict:
        if isinstance(self._value, Symbol):
            return self._value.to_dict()
        else:
            return {
                "type": type(self).__name__.lower(),
                "value": self.encode_value(),
            }

    @property
    def is_symbolic(self) -> bool:
        """Returns whether variable is symbolic."""
        return isinstance(self._value, Symbol)

    @property
    def is_value(self) -> bool:
        """Returns whether variable contains a value."""
        return not isinstance(self._value, Symbol)

    def get_value(self) -> Any:
        """Retrieve value, if it exists."""
        if isinstance(self._value, Symbol):
            raise TypeError(
                f"{type(self).__name__} is symbolic and does not contain a value."
            )
        return self._value

    def get_symbol(self) -> Symbol:
        """Retrieve symbol, if it exists."""
        if not isinstance(self._value, Symbol):
            raise TypeError(f"{type(self).__name__} is a valued object.")
        return self._value


class Equatable(Variable):
    def __eq__(self, value: Any):
        other = self.preprocess(value)
        if self.is_value and other.is_value:
            return self.get_value() == other.get_value()
        return Eq(self, other)

    def __ne__(self, value: Any):
        other = self.preprocess(value)
        if self.is_value and other.is_value:
            return self.get_value() != other.get_value()
        return Ne(self, other)

    def in_(self, vlist: List[Any]):
        return Or(
            *[
                Eq(self, self.preprocess(v))
                for v in vlist
            ]
        )

    def __hash__(self):
        if self.is_symbolic:
            raise ValueError("Variable is symbolic.")
        return hash(str(self))


class Quantifiable(Equatable):
    def __gt__(self, value: Any):
        other = self.preprocess(value)
        if self.is_value and other.is_value:
            return self.get_value() > other.get_value()
        return Gt(self, other)

    def __ge__(self, value: Any):
        other = self.preprocess(value)
        if self.is_value and other.is_value:
            return self.get_value() >= other.get_value()
        return Ge(self, other)

    def __lt__(self, value: Any):
        other = self.preprocess(value)
        if self.is_value and other.is_value:
            return self.get_value() < other.get_value()
        return Lt(self, other)

    def __le__(self, value: Any):
        other = self.preprocess(value)
        if self.is_value and other.is_value:
            return self.get_value() <= other.get_value()
        return Le(self, other)


class Nullable(Variable):
    def is_none(self):
        return IsNull(self)

    def is_not_none(self):
        return IsNotNull(self)

    def get_value(self) -> Optional[Any]:
        return super().get_value()


class Spatial(Variable):
    def intersects(self, other: Any):
        return Intersects(self, self.preprocess(other))

    def inside(self, other: Any):
        return Inside(self, self.preprocess(other))

    def outside(self, other: Any):
        return Outside(self, self.preprocess(other))
