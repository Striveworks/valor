from typing import Any


class Function:
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
        return Negate(self)

    def to_dict(self):
        return {
            "op": type(self).__name__.lower(),
            "args": [arg.to_dict() for arg in self._args],
        }


class OneArgumentFunction(Function):
    def __init__(self, arg) -> None:
        super().__init__(arg)

    @property
    def arg(self):
        return self._args[0]

    def to_dict(self):
        return {"op": type(self).__name__.lower(), "arg": self.arg.to_dict()}


class TwoArgumentFunction(Function):
    def __init__(self, lhs: Any, rhs: Any) -> None:
        self._lhs = lhs
        self._rhs = rhs
        super().__init__(lhs, rhs)

    @property
    def lhs(self):
        return self._lhs

    @property
    def rhs(self):
        return self._rhs

    def to_dict(self):
        return {
            "op": type(self).__name__.lower(),
            "lhs": self.lhs.to_dict(),
            "rhs": self.rhs.to_dict(),
        }


class AppendableFunction(Function):
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
        self._args.append(value)
        return self


class And(AppendableFunction):
    _operator = "&"

    def __and__(self, other: Any):
        self.append(other)
        return self


class Or(AppendableFunction):
    _operator = "|"

    def __or__(self, other: Any):
        self.append(other)
        return self


class Xor(AppendableFunction):
    _operator = "^"

    def __xor__(self, other: Any):
        self.append(other)
        return self


class Negate(OneArgumentFunction):
    _operator = "~"

    def __invert__(self):
        """Inverts negation so return contents."""
        return self.arg


class IsNull(OneArgumentFunction):
    pass


class IsNotNull(OneArgumentFunction):
    pass


class Eq(TwoArgumentFunction):
    _operator = "=="


class Ne(TwoArgumentFunction):
    _operator = "!="


class Gt(TwoArgumentFunction):
    _operator = ">"


class Ge(TwoArgumentFunction):
    _operator = ">="


class Lt(TwoArgumentFunction):
    _operator = "<"


class Le(TwoArgumentFunction):
    _operator = "<="


class Intersects(TwoArgumentFunction):
    pass


class Inside(TwoArgumentFunction):
    pass


class Outside(TwoArgumentFunction):
    pass


class Where(TwoArgumentFunction):
    pass
