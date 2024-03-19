from typing import Any


class Function:
    operator = None

    def __init__(self, *args) -> None:
        self._args = list(args)

    def __repr__(self):
        args = ", ".join([arg.__repr__() for arg in self._args])
        return f"{type(self).__name__}({args})"

    def __str__(self):
        values = [
            f"'{arg}'" if isinstance(arg, str) else str(arg)
            for arg in self._args
        ]
        if self.operator is None:
            args = ", ".join(values)
            return f"{type(self).__name__}({args})"
        else:
            args = f" {self.operator} ".join(values)
            return f"({args})"

    def __and__(self, other: Any):
        return And(self, other)

    def __or__(self, other: Any):
        return Or(self, other)

    def __xor__(self, other: Any):
        return Xor(self, other)

    def __invert__(self):
        return Negate(self)

    def reduce(self):
        return self

    def to_dict(self):
        return {
            "op": type(self).__name__.lower(), 
            "args": [arg.to_dict() for arg in self._args]
        }


class OneArgumentFunction(Function):
    def __init__(self, arg, **kwargs) -> None:
        super().__init__(arg, **kwargs)

    @property
    def arg(self):
        return self._args[0]
    
    def to_dict(self):
        return {
            "op": type(self).__name__.lower(),
            "arg": self.arg.to_dict()
        }


class TwoArgumentFunction(Function):
    def __init__(self, lhs: Any, rhs: Any, **kwargs) -> None:
        self._lhs = lhs
        self._rhs = rhs
        super().__init__(lhs, rhs, **kwargs)

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
    overload = None

    def __init__(self, *args, **kwargs) -> None:
        if len(args) < 2:
            raise ValueError
        if self.overload is None:
            raise NotImplementedError
        self.__setattr__(self.overload, self.append)
        super().__init__(*args, **kwargs)

    def append(self, value: Any):
        self._args.append(value)


class And(AppendableFunction):
    operator = "&"
    overload = "__and__"


class Or(AppendableFunction):
    operator = "|"
    overload = "__or__"


class Xor(AppendableFunction):
    operator = "^"
    overload = "__xor__"


class Negate(OneArgumentFunction):
    operator = "~"


class IsNull(OneArgumentFunction):
    pass


class IsNotNull(OneArgumentFunction):
    pass


class Eq(TwoArgumentFunction):
    operator = "=="


class Ne(TwoArgumentFunction):
    operator = "!="


class Gt(TwoArgumentFunction):
    operator = ">"


class Ge(TwoArgumentFunction):
    operator = ">="


class Lt(TwoArgumentFunction):
    operator = "<"


class Le(TwoArgumentFunction):
    operator = "<="


class Intersects(TwoArgumentFunction):
    pass


class Inside(TwoArgumentFunction):
    pass


class Outside(TwoArgumentFunction):
    pass


class Where(TwoArgumentFunction):
    pass
