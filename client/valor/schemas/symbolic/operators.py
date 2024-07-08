import warnings
from typing import Any, Optional, Union


class Condition:
    """Base class for defining a conditional operation."""

    def __init__(self, lhs: Any, rhs: Optional[Any] = None) -> None:
        """
        Create a condition.

        Parameters
        ----------
        lhs : Variable
            A variable.
        rhs : Variable, optional
            An optional rhs variable.
        """
        # validate lhs
        if not lhs.is_symbolic:
            warnings.warn(
                "Values are currently not supported as the lhs operand in the API.",
                RuntimeWarning,
            )

        # validate rhs - symbols are not current supported
        if rhs is not None:
            if rhs.is_symbolic:
                warnings.warn(
                    "Symbols are currently not supported as the rhs operand in the API.",
                    RuntimeWarning,
                )

        self.lhs = lhs
        self.rhs = rhs
        self.op = type(self).__name__.lower()

    def __and__(self, other: Any):
        return And(self, other)

    def __or__(self, other: Any):
        return Or(self, other)

    def __invert__(self):
        return Not(self)

    def to_dict(self):
        return {
            "lhs": self.lhs.to_dict(),
            "rhs": self.rhs.to_dict() if self.rhs is not None else None,
            "op": self.op,
        }


class Function:
    """Base class for defining a logical function."""

    def __init__(self, *args) -> None:
        if len(args) == 0:
            raise ValueError("Expected at least one argument.")

        if any([not isinstance(arg, (Condition, Function)) for arg in args]):
            raise ValueError(
                f"Arguments should contain either Conditions or other Functions."
            )

        self.op = type(self).__name__.lower()
        self.args = list(args)

    def __repr__(self):
        args = ", ".join([arg.__repr__() for arg in self.args])
        return f"{type(self).__name__}({args})"

    def __str__(self):
        values = [arg.__repr__() for arg in self.args]
        args = ", ".join(values)
        return f"{type(self).__name__}({args})"

    def __and__(self, other: Any):
        return and_(self, other)

    def __or__(self, other: Any):
        return or_(self, other)

    def __invert__(self):
        return not_(self)

    def to_dict(self):
        """Encode to a JSON-compatible dictionary."""
        arg_dict = [arg.to_dict() for arg in self.args]
        return {"op": self.op, "args": arg_dict}


class And(Function):
    """Implementation of logical AND (&)."""

    def __init__(self, *args):
        if len(args) < 2:
            raise ValueError("Expected at least two arguments.")
        super().__init__(*args)

    def __and__(self, other: Any):
        if isinstance(other, And):
            self.args.extend(other.args)
        else:
            self.args.append(other)
        return self


class Or(Function):
    """Implementation of logical OR (|)."""

    def __init__(self, *args):
        if len(args) < 2:
            raise ValueError("Expected at least two arguments.")
        super().__init__(*args)

    def __or__(self, other: Any):
        if isinstance(other, Or):
            self.args.extend(other.args)
        else:
            self.args.append(other)
        return self


class Not(Function):
    """Implementation of logical negation (~)."""

    def __init__(self, arg: Union[Condition, Function]):
        super().__init__(arg)

    def __invert__(self):
        """Inverts negation so return contents."""
        return self.args[0]


def and_(*args) -> Any:

    if len(args) < 2:
        raise ValueError("Expected at least two arguments.")

    valid_args = list()
    for arg in args:
        if isinstance(arg, (Condition, Function)):
            valid_args.append(arg)
        elif isinstance(arg, bool):
            if arg is False:
                return False
        else:
            raise TypeError(
                f"Type `{type(arg)}` is not a supported function argument. Arguments should be a Condition or another Function."
            )

    # condition where and_(True, True)
    if len(valid_args) == 0:
        return True

    # condition where and_(Dataset.name == "x", True)
    elif len(valid_args) == 1:
        return valid_args[0]

    # otherwise, we have enough conditions to form an and-expression
    else:
        return And(*valid_args)


def or_(*args) -> Any:

    if len(args) < 2:
        raise ValueError("Expected at least two arguments.")

    valid_args = list()
    for arg in args:
        if isinstance(arg, (Condition, Function)):
            valid_args.append(arg)
        elif isinstance(arg, bool):
            if arg is True:
                return True
        else:
            raise TypeError(
                f"Type `{type(arg)}` is not a supported function argument. Arguments should be a Condition or another Function."
            )

    # condition where or_(False, False)
    if len(valid_args) == 0:
        return False

    # condition where or_(Dataset.name == "x", False)
    elif len(valid_args) == 1:
        return valid_args[0]

    # otherwise, we have enough conditions to form an or-expression
    else:
        return Or(*valid_args)


def not_(_arg: Union[bool, Condition, Function]) -> Any:
    if isinstance(_arg, bool):
        return not _arg
    elif isinstance(_arg, Function) and _arg.op == "not":
        return _arg.args
    elif isinstance(_arg, (Condition, Function)):
        return Not(_arg)
    else:
        raise TypeError(
            f"Type `{type(_arg)}` is not a supported function argument. Arguments should be a Condition or another Function."
        )


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

    def to_dict(self):
        return Not(Eq(lhs=self.lhs, rhs=self.rhs)).to_dict()


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
    Function,
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
