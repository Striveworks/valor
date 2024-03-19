from typing import Any, Dict, List, Optional

from valor.symbolic.atomics import Float
from valor.symbolic.modifiers import Equatable, Symbol, Variable
from valor.symbolic.utils import get_type_by_name, get_type_by_value


class StaticCollection(Equatable):
    """
    A static collection is a Variable that stores its contents as static attributes.
    """

    @classmethod
    def definite(cls, **kwargs):
        instance = cls()
        _static_variables = {
            name
            for name in vars(cls)
            if issubclass(type(instance.__getattribute__(name)), Variable)
        }
        if set(kwargs.keys()).issubset(_static_variables):
            for name in _static_variables:
                objcls = type(instance.__getattribute__(name))
                objval = kwargs[name] if name in kwargs else None
                objval = (
                    objval
                    if (isinstance(objval, Variable) and objval.is_value)
                    else objcls.definite(objval)
                )
                try:
                    instance.__setattr__(name, objval)
                except TypeError as e:
                    raise TypeError(
                        f"Attribute `{cls.__name__}.{name}` raised: {str(e)}"
                    )
        else:
            raise TypeError(
                f"{cls.__name__}() does not take the following keyword arguments '{_static_variables - set(kwargs.keys())}'"
            )
        instance.__validate__()
        return instance

    def __validate__(self):
        pass

    def __repr__(self):
        if self.is_symbolic:
            return super().__repr__()
        return self.encode().__repr__()

    def __str__(self):
        if self.is_symbolic:
            return super().__str__()
        return str(self.encode())

    def _parse_object(self):
        retval = dict()
        for name in vars(type(self)):
            objval = self.__getattribute__(name)
            if issubclass(type(objval), Variable):
                retval[name] = objval
        return retval

    @classmethod
    def supports(cls, value: Any) -> bool:
        return value is None

    @classmethod
    def decode(cls, value: dict):
        return cls(**value)

    def encode(self):
        return {k: v.encode() for k, v in self._parse_object().items()}


class ValueList(Equatable):
    _supported_type = Variable

    @classmethod
    def definite(cls, value: Any):
        if value is None:
            value = list()
        return super().definite(value=value)

    @classmethod
    def supports(cls, value: List[Any]) -> bool:
        return type(value) is list and all(
            [
                (type(element) is cls._supported_type and element.is_value)
                or isinstance(element, dict)
                for element in value
            ]
        )

    @classmethod
    def decode(cls, value: List[dict]):
        if not value:
            return []
        if issubclass(type(value), Variable):
            return [cls._supported_type.decode(element) for element in value]

    def encode(self):
        return [element.encode() for element in self.get_value()]

    def __getitem__(self, __key: int):
        return self.get_value()[__key]

    def __setitem__(self, __key: int, __value: Any):
        value = self.get_value()
        if value is None:
            raise TypeError
        value[__key] = __value

    def __iter__(self):
        return iter([element for element in self.get_value()])


class DictionaryValue:
    def __init__(
        self,
        symbol: Symbol,
        key: str,
    ):
        self._key = key
        self._owner = symbol._owner
        self._name = symbol._name
        if symbol._attribute:
            raise ValueError("Symbol attribute should not be defined.")
        if symbol._key:
            raise ValueError("Symbol key should not be defined.")

    def __eq__(self, other: Any):
        return self.generate(fn="__eq__", other=other)

    def __ne__(self, other: Any):
        return self.generate(fn="__ne__", other=other)

    def __gt__(self, other: Any):
        return self.generate(fn="__gt__", other=other)

    def __ge__(self, other: Any):
        return self.generate(fn="__ge__", other=other)

    def __lt__(self, other: Any):
        return self.generate(fn="__lt__", other=other)

    def __le__(self, other: Any):
        return self.generate(fn="__le__", other=other)

    def intersects(self, other: Any):
        return self.generate(fn="intersects", other=other)

    def inside(self, other: Any):
        return self.generate(fn="inside", other=other)

    def outside(self, other: Any):
        return self.generate(fn="outside", other=other)

    def is_none(self, other: Any):
        return self.generate(fn="is_none", other=other)

    def is_not_none(self, other: Any):
        return self.generate(fn="is_not_none", other=other)

    @property
    def area(self):
        return Float.symbolic(
            owner=self._owner, name=self._name, key=self._key, attribute="area"
        )

    def generate(self, other: Any, fn: str):
        obj = get_type_by_value(other)
        sym = obj.symbolic(owner=self._owner, name=self._name, key=self._key)
        return sym.__getattribute__(fn)(other)


class Dictionary(Equatable):
    @classmethod
    def definite(
        cls,
        value: Optional[Dict[str, Any]] = None,
    ):
        value = value if value else dict()
        value = {k: get_type_by_value(v).definite(v) for k, v in value.items()}
        return super().definite(value)

    @classmethod
    def supports(cls, value: Any) -> bool:
        return type(value) in {dict, Dictionary}

    @classmethod
    def decode(cls, value: dict) -> Any:
        return {
            k: get_type_by_name(v["type"]).decode(v["value"])
            for k, v in value.items()
        }

    def encode(self) -> dict:
        return {k: v.to_dict() for k, v in self.items()}

    def __getitem__(self, key: str):
        if isinstance(self._value, Symbol):
            return DictionaryValue(symbol=self._value, key=key)
        return self.get_value()[key]

    def __setitem__(self, key: str, value: Any):
        if isinstance(self._value, Symbol):
            raise NotImplementedError(
                "Symbols do not support the setting of values."
            )
        self.get_value()[key] = value

    def items(self):
        if isinstance(self._value, Symbol):
            raise NotImplementedError("Variable is symbolic")
        return self._value.items() if self._value else dict.items({})
