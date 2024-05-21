from typing import Any, List, Optional, Union


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


class Contains(TwoArgumentFunction):
    """Implementation of the list 'contains' operator."""

    pass


FunctionType = Union[
    Function,
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


def get_operator_by_name(name: str) -> type:
    """Retrieves operator type by name."""
    operators = {
        "and": And,
        "or": Or,
        "xor": Xor,
        "not": Not,
        "isnull": IsNull,
        "isnotnull": IsNotNull,
        "eq": Eq,
        "ne": Ne,
        "gt": Gt,
        "ge": Ge,
        "lt": Lt,
        "le": Le,
        "intersects": Intersects,
        "inside": Inside,
        "outside": Outside,
        "contains": Contains,
    }
    if name not in operators:
        raise ValueError(
            f"A symbolic operator with name '{name}' does not exist."
        )
    return operators[name]


def _unpack_function_or_variable(
    expr: dict, additional_types: Optional[List[type]] = None
):
    """
    Parses a dictionary into a function or variable type.

    Parameters
    ----------
    value: dict
        A dictionary representation of a variable, symbol or a function.
    additional_types: List[type], optional
        Any additional variable types that are not defined in this file.

    Returns
    -------
    Union[FunctionType, Variable]
        A instance of a class that inherits from either Function or Variable.

    Raises
    ------
    TypeError
        If the provided value is not a dictionary.
    ValueError
        If the provided value does not conform to the dictionary representation of a variable or function.
    """

    if "op" in expr:
        op = get_operator_by_name(expr["op"])
        keys = set(expr.keys())
        if keys == {"op", "args"}:
            args = [
                _unpack_function_or_variable(
                    value, additional_types=additional_types
                )
                for value in expr["args"]
            ]
            return op(*args)
        elif keys == {"op", "arg"}:
            arg = _unpack_function_or_variable(
                expr["arg"], additional_types=additional_types
            )
            return op(arg)
        elif keys == {"op", "lhs", "rhs"}:
            lhs = _unpack_function_or_variable(
                expr["lhs"], additional_types=additional_types
            )
            rhs = _unpack_function_or_variable(
                expr["rhs"], additional_types=additional_types
            )
            return op(lhs, rhs)
        else:
            raise NotImplementedError(
                f"Unsupported operator arguments '{expr.keys}'."
            )
    elif "type" in expr:
        from valor.schemas.symbolic.types import unpack_variable

        return unpack_variable(expr, additional_types=additional_types)
    else:
        raise ValueError(
            f"Dictionary with keys '{expr.keys()}' does not conform a symbolic function."
        )


def unpack_function(
    expr: dict, additional_types: Optional[List[type]] = None
) -> FunctionType:
    """
    Parses a dictionary into a function type.

    Parameters
    ----------
    value: dict
        A dictionary representation of a function.
    additional_types: List[type], optional
        Any additional variable types that are not defined in this file.

    Returns
    -------
    FunctionType
        A instance of a class that subclasses Function.

    Raises
    ------
    TypeError
        If the provided value is not a dictionary.
    ValueError
        If the provided value does not conform to the dictionary representation of a function.
    """
    if isinstance(expr, Function):
        return expr
    elif not isinstance(expr, dict):
        raise TypeError("Expected input to be of type 'dict'.")

    retval = _unpack_function_or_variable(
        expr, additional_types=additional_types
    )
    if not isinstance(retval, Function):
        raise TypeError
    return retval
