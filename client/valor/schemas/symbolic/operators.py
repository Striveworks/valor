from typing import Any, Union


class Function:
    """Base class for defining a function."""

    _operator = None

    def __init__(self, *args) -> None:
        for arg in args:
            if not hasattr(arg, "to_dict"):
                raise ValueError(
                    "Functions can only take arguments that have a 'to_dict' method defined."
                )
        self._args = list(args)

    def __repr__(self):
        args = ", ".join([arg.__repr__() for arg in self._args])
        return f"{type(self).__name__}({args})"

    def __str__(self):
        values = [arg.__repr__() for arg in self._args]
        if self._operator is None:
            args = ", ".join(values)
            return f"{type(self).__name__}({args})"
        else:
            args = f" {self._operator} ".join(values)
            return f"({args})"

    def __and__(self, other: Any):
        return And(self, other)

    def __or__(self, other: Any):
        return Or(self, other)

    def __xor__(self, other: Any):
        return Xor(self, other)

    def __invert__(self):
        return Not(self)

    def to_dict(self):
        """Encode to a JSON-compatible dictionary."""
        return {
            "op": type(self).__name__.lower(),
            "args": [arg.to_dict() for arg in self._args],
        }


class OneArgumentFunction(Function):
    """Base class for defining single argument functions."""

    def __init__(self, arg) -> None:
        super().__init__(arg)

    @property
    def arg(self):
        """Returns the argument."""
        return self._args[0]

    def to_dict(self):
        """Encode to a JSON-compatible dictionary."""
        return {"op": type(self).__name__.lower(), "arg": self.arg.to_dict()}


class TwoArgumentFunction(Function):
    """Base class for defining two argument functions."""

    def __init__(self, lhs: Any, rhs: Any) -> None:
        self._lhs = lhs
        self._rhs = rhs
        super().__init__(lhs, rhs)

    @property
    def lhs(self):
        """Returns the lhs operand."""
        return self._lhs

    @property
    def rhs(self):
        """Returns the rhs operand."""
        return self._rhs

    def to_dict(self):
        """Encode to a JSON-compatible dictionary."""
        return {
            "op": type(self).__name__.lower(),
            "lhs": self.lhs.to_dict(),
            "rhs": self.rhs.to_dict(),
        }


class AppendableFunction(Function):
    """Base class for defining functions with an unlimited number of arguments."""

    _function = None

    def __init__(self, *args) -> None:
        """
        Appendable function.
        """
        if len(args) < 2:
            raise TypeError(
                f"missing {2 - len(args)} required positional argument"
            )
        flat_args = []
        for arg in args:
            if isinstance(arg, type(self)):
                flat_args += arg._args
            else:
                flat_args.append(arg)
        super().__init__(*flat_args)

    def append(self, value: Any):
        """Appends an argument to the function."""
        self._args.append(value)
        return self


class And(AppendableFunction):
    """Implementation of logical AND (&)."""

    _operator = "&"

    def __and__(self, other: Any):
        self.append(other)
        return self


class Or(AppendableFunction):
    """Implementation of logical OR (|)."""

    _operator = "|"

    def __or__(self, other: Any):
        self.append(other)
        return self


class Xor(AppendableFunction):
    """Implementation of logical XOR (^)."""

    _operator = "^"

    def __xor__(self, other: Any):
        self.append(other)
        return self


class Not(OneArgumentFunction):
    """Implementation of logical negation (~)."""

    _operator = "~"

    def __invert__(self):
        """Inverts negation so return contents."""
        return self.arg


class IsNull(OneArgumentFunction):
    """Implementation of is null value check."""

    pass


class IsNotNull(OneArgumentFunction):
    """Implementation of is not null value check."""

    pass


class Eq(TwoArgumentFunction):
    """Implementation of the equality operator '=='."""

    _operator = "=="


class Ne(TwoArgumentFunction):
    """Implementation of the inequality operator '!='."""

    _operator = "!="


class Gt(TwoArgumentFunction):
    """Implementation of the greater-than operator '>'."""

    _operator = ">"


class Ge(TwoArgumentFunction):
    """Implementation of the greater-than or equal operator '>='."""

    _operator = ">="


class Lt(TwoArgumentFunction):
    """Implementation of the less-than operator '<'."""

    _operator = "<"


class Le(TwoArgumentFunction):
    """Implementation of the less-than or equal operator '<='."""

    _operator = "<="


class Intersects(TwoArgumentFunction):
    """Implementation of the spatial 'intersects' operator."""

    pass


class Inside(TwoArgumentFunction):
    """Implementation of the spatial 'inside' operator."""

    pass


class Outside(TwoArgumentFunction):
    """Implementation of the spatial 'outside' operator."""

    pass


class Contains(AppendableFunction):
    """Implementation of the list 'contains' operator."""

    pass


class Filter(AppendableFunction):

    pass


FunctionType = Union[
    And,
    Or,
    Xor,
    Not,
    IsNull,
    IsNotNull,
    Eq,
    Ne,
    Gt,
    Ge,
    Lt,
    Le,
    Intersects,
    Inside,
    Outside,
    Contains,
]
