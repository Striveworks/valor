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

        self._args = []
        for arg in args:
            if not hasattr(arg, "to_dict"):
                raise ValueError(
                    f"Arguments should be symbolic or functional. Received '{arg}'."
                )
            if isinstance(arg, type(self)):
                self._args.extend(arg._args)
            else:
                self._args.append(arg)
        self._args = self._args if len(self._args) > 1 else self._args[0]

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
    """
    Implementation of logical AND (&).

    This class represents a logical AND operation that can be performed on
    two or more arguments. It supports chaining of multiple AND operations
    using the `&` operator.

    Parameters
    ----------
    *args : Any
        The arguments to be logically ANDed together. At least two arguments
        are required.

    Raises
    ------
    ValueError
        If fewer than two arguments are provided.

    Examples
    --------
    >>> a = And(Label.key == "k1", Label.value == "v1")
    >>> b = And(Label.key == "k1", Label.value == "v2")
    >>> c = a & b

    Methods
    -------
    __and__(other)
        Supports chaining of multiple `And` operations using the `&` operator.
    """

    def __init__(self, *args):
        if len(args) < 2:
            raise ValueError("Expected at least two arguments.")
        super().__init__(*args)

    def __and__(self, other: Any):
        if isinstance(other, And):
            self._args.extend(other._args)
        else:
            self._args.append(other)
        return self


class Or(Function):
    """Implementation of logical OR (|).

    This class represents a logical OR operation that can be performed on
    two or more arguments. It supports chaining of multiple OR operations
    using the `|` operator.

    Parameters
    ----------
    *args : Any
        The arguments to be logically ORed together. At least two arguments
        are required.

    Raises
    ------
    ValueError
        If fewer than two arguments are provided.

    Examples
    --------
    >>> a = Or(Label.key == "k1", Label.key == "k2")
    >>> b = Or(Label.value == "v1", Label.value == "v2")
    >>> c = a | b

    Methods
    -------
    __or__(other)
        Supports chaining of multiple `Or` operations using the `|` operator.
    """

    def __init__(self, *args):
        if len(args) < 2:
            raise ValueError("Expected at least two arguments.")
        super().__init__(*args)

    def __or__(self, other: Any):
        if isinstance(other, Or):
            self._args.extend(other._args)
        else:
            self._args.append(other)
        return self


class Not(Function):
    """Implementation of logical negation (~).

    This class represents a logical NOT operation that can be performed on
    on a single arguments. It supports chaining of multiple NOT operations
    using the `~` operator.

    Parameters
    ----------
    *args : Any
        The arguments to be logically ORed together. At least two arguments
        are required.

    Raises
    ------
    ValueError
        If the number of args is not equal to one.

    Examples
    --------
    >>> a = Not(Label.key == "k1")
    >>> b = ~a

    Methods
    -------
    __invert__()
        Supports chaining of multiple `Not` operations using the `~` operator.
    """

    def __init__(self, *args):
        if len(args) != 1:
            raise ValueError("Negation only takes one argument.")
        elif isinstance(args[0], Not):
            return args[0]._args
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
