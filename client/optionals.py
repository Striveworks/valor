import typing
from typing import Any

from valor.schemas import Annotation, Label
from valor.schemas.symbolic.operators import IsNotNull, IsNull
from valor.schemas.symbolic.types import Bool, Box, Float, Symbol, Variable

# T = typing.TypeVar("T", bound=Variable)

# def create_optional_type(variable_class: typing.Type[T]):

#     if not issubclass(variable_class, Variable):
#         raise TypeError("Provided type is not a subclass of Variable.")

#     class OptionalVariable(variable_class):
#         def __init__(
#             self,
#             value: typing.Optional[typing.Any] = None,
#             symbol: typing.Optional[Symbol] = None,
#         ):
#             if value is not None and isinstance(value, variable_class):
#                 value = value.get_value()
#             super().__init__(value, symbol)

#         @classmethod
#         def definite(cls, value: typing.Any):
#             """Initialize variable with a value."""
#             if isinstance(value, variable_class):
#                 value = value.get_value()
#             return cls(value=value, symbol=None)

#         @classmethod
#         def symbolic(
#             cls,
#             name: typing.Optional[str] = None,
#             key: typing.Optional[str] = None,
#             attribute: typing.Optional[str] = None,
#             owner: typing.Optional[str] = None,
#         ):
#             if name is None:
#                 name = f"optional[{variable_class.__name__.lower()}]"
#             return cls(
#                 value=None, symbol=Symbol(name, key, attribute, owner)
#             )

#         @classmethod
#         def __validate__(cls, value: typing.Any):
#             """
#             Validates typing.

#             Parameters
#             ----------
#             value : typing.Any
#                 The value to validate.

#             Raises
#             ------
#             TypeError
#                 If the value type is not supported.
#             """
#             if value is not None:
#                 variable_class.__validate__(value)

#         @classmethod
#         def decode_value(cls, value: typing.Any):
#             """Decode object from JSON compatible dictionary."""
#             if value is None:
#                 return cls(None)
#             return cls(variable_class.decode_value(value).get_value())

#         def encode_value(self):
#             """Encode object to JSON compatible dictionary."""
#             value = self.get_value()
#             if value is None:
#                 return None
#             return variable_class(value).encode_value()

#         def to_dict(self):
#             """Encode variable to a JSON-compatible dictionary."""
#             if isinstance(self._value, Symbol):
#                 return self._value.to_dict()
#             else:
#                 return {
#                     "type": f"optional[{variable_class.__name__.lower()}]",
#                     "value": self.encode_value(),
#                 }

#         def is_none(self) -> typing.Union[Bool, IsNull]:
#             """Conditional whether variable is 'None'"""
#             if self.is_value:
#                 return Bool(self.get_value() is None)
#             return IsNull(self)

#         def is_not_none(self) -> typing.Union["Bool", IsNotNull]:
#             """Conditional whether variable is not 'None'"""
#             if self.is_value:
#                 return Bool(self.get_value() is not None)
#             return IsNotNull(self)

#         def get_value(self) -> typing.Optional[typing.Any]:
#             """Re-typed to output 'Optional[Any]'"""
#             return super().get_value()

#         def unwrap(self) -> typing.Optional[variable_class]:
#             """Unwraps the optional into a subclass of Variable or 'None'."""
#             value = self.get_value()
#             if value is None:
#                 return None
#             return variable_class(value)

#     return OptionalVariable


# class OptionalBox(create_optional_type(Box), Box):
#     pass


# class OptionalFloat(create_optional_type(Float), Float):
#     pass


# T = typing.TypeVar("T", bound=Variable)

# class Optional(typing.Generic[T]):
#     def __class_getitem__(cls, item: typing.Type[T]):
#         if not issubclass(item, Variable):
#             raise TypeError("Provided type is not a subclass of Variable.")

#         items = {
#             Float: OptionalFloat,
#             Box: OptionalBox
#         }
#         return items[item]

# Optional[Box].


# if __name__ == "__main__":


#     import json

#     a = OptionalFloat.definite(None)
#     b = OptionalFloat.definite(0.5)

#     print(a.is_none())
#     print(b.is_none())

#     print(a.is_not_none())
#     print(b.is_not_none())

#     x = OptionalBox.symbolic().area > 1
#     y = OptionalBox.symbolic().is_none()
#     print(json.dumps(x.to_dict(),indent=2))
#     print(json.dumps(y.to_dict(),indent=2))
