from typing import Any, Optional, Union


class Condition:
    """Base class for defined a filter condition."""

    def __init__(self, lhs: Any, rhs: Optional[Any] = None) -> None:
        """
        TODO

        """

        self.op = type(self).__name__.lower()

        # validate lhs
        lhd = lhs.to_dict()
        if lhd.keys() != {"name", "key"}:
            raise ValueError
        self.lhs = lhd["name"]
        self.lhs_key = lhd["key"]

        # validate rhs - symbols are not current supported
        self.rhs = None
        self.rhs_key = None
        if rhs is not None:
            rhd = rhs.to_dict()
            if rhd.keys() == {"name", "key"}:
                raise NotImplementedError(
                    "Symbols are not supported currently as rhs values."
                )
            elif rhd.keys() == {"type", "value"}:
                self.rhs = rhd
            else:
                raise ValueError

    def __and__(self, other: Any):
        return And(self, other)

    def __or__(self, other: Any):
        return Or(self, other)

    def __invert__(self):
        return Not(self)

    def to_dict(self):
        return {
            "lhs": self.lhs,
            "lhs_key": self.lhs_key,
            "rhs": self.rhs,
            "rhs_key": self.rhs_key,
            "op": self.op,
        }


class Function:
    """Base class for defining a logical function."""

    def __init__(self, *args) -> None:
        if len(args) == 0:
            raise ValueError("Expected at least one argument.")
        for arg in args:
            if not hasattr(arg, "to_dict"):
                raise ValueError(
                    "Functions can only take arguments that have a 'to_dict' method defined."
                )
        self._args = list(args) if len(args) > 1 else args[0]

    def __repr__(self):
        args = ", ".join([arg.__repr__() for arg in self._args])
        return f"{type(self).__name__}({args})"

    def __str__(self):
        values = [arg.__repr__() for arg in self._args]
        args = ", ".join(values)
        return f"{type(self).__name__}({args})"

    def __and__(self, other: Any):
        return And(self, other)

    def __or__(self, other: Any):
        return Or(self, other)

    def __invert__(self):
        return Not(self)

    def to_dict(self):
        """Encode to a JSON-compatible dictionary."""
        args = (
            [arg.to_dict() for arg in self._args]
            if isinstance(self._args, list)
            else self._args.to_dict()
        )
        return {"op": type(self).__name__.lower(), "args": args}


class And(Function):
    """Implementation of logical AND (&)."""

    pass

    def __and__(self, other: Any):
        if isinstance(other, And):
            self._args.extend(other._args)
        else:
            self._args.append(other)
        return self


class Or(Function):
    """Implementation of logical OR (|)."""

    pass

    def __or__(self, other: Any):
        if isinstance(other, Or):
            self._args.extend(other._args)
        else:
            self._args.append(other)
        return self


class Not(Function):
    """Implementation of logical negation (~)."""

    def __init__(self, *args):
        if len(args) != 1:
            raise ValueError("Negation only takes one argument.")
        super().__init__(*args)

    def __invert__(self):
        """Inverts negation so return contents."""
        if isinstance(self._args, list):
            raise ValueError("Negation only takes one argument.")
        return self._args


class IsNull(Condition):
    """Implementation of is null value check."""

    pass


class IsNotNull(Condition):
    """Implementation of is not null value check."""

    pass


class Eq(Condition):
    """Implementation of the equality operator '=='."""

    pass


class Ne(Condition):
    """Implementation of the inequality operator '!='."""

    pass


class Gt(Condition):
    """Implementation of the greater-than operator '>'."""

    pass


class Gte(Condition):
    """Implementation of the greater-than or equal operator '>='."""

    pass


class Lt(Condition):
    """Implementation of the less-than operator '<'."""

    pass


class Lte(Condition):
    """Implementation of the less-than or equal operator '<='."""

    pass


class Intersects(Condition):
    """Implementation of the spatial 'intersects' operator."""

    pass


class Inside(Condition):
    """Implementation of the spatial 'inside' operator."""

    pass


class Outside(Condition):
    """Implementation of the spatial 'outside' operator."""

    pass


class Contains(Condition):
    """Implementation of the list 'contains' operator."""

    pass


FunctionType = Union[
    And,
    Or,
    Not,
    IsNull,
    IsNotNull,
    Eq,
    Ne,
    Gt,
    Gte,
    Lt,
    Lte,
    Intersects,
    Inside,
    Outside,
    Contains,
]
