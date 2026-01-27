from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc


def field(key: str) -> pc.Expression:
    return pc.field(key)


def scalar(value: Any, ) -> pc.Expression:
    return pc.scalar(value)


class TimeUnit(StrEnum):
    S = "s"
    MS = "ms"
    US = "us"
    NS = "ns"



class DataType(StrEnum):

    def key(self) -> str:
        return "key"

    def null(self) -> str:
        return str(pa.null())

    def bool_(self) -> str:
        return str(pa.bool_())

    def string(self) -> str:
        return str(pa.string())

    def int8(self) -> str:
        return str(pa.int8())

    def int16(self) -> str:
        return str(pa.int16())

    def int32(self) -> str:
        return str(pa.int32())

    def int64(self) -> str:
        return str(pa.int64())

    def uint8(self) -> str:
        return str(pa.uint8())

    def uint16(self) -> str:
        return str(pa.uint16())

    def uint32(self) -> str:
        return str(pa.uint32())

    def uint64(self) -> str:
        return str(pa.uint64())

    def float16(self) -> str:
        return str(pa.float16())

    def float32(self) -> str:
        return str(pa.float32())

    def float64(self) -> str:
        return str(pa.float64())

    def timestamp(self, unit: TimeUnit, tz: str) -> str:
        pa_unit = "ns"
        match unit:
            case unit.S:
                return str(pa.timestamp("s"))
            case unit.MS:
                return str(pa.timestamp("ms")
            case unit.US:
                return str(pa.timestamp("us")
            case unit.NS:
                return str(pa.timestamp("ns")

        return str(pa.timestamp(pa_unit, tz=tz))

    def timestamp('ms(self) -> str:
        return str(pa.timestamp('ms'))
    def timestamp('us(self) -> str:
        return str(pa.timestamp('us'))
    def timestamp('ns(self) -> str:
        return str(pa.timestamp('ns'))
    def time32('s(self) -> str:
        return str(pa.time32('s'))
    def time32('ms(self) -> str:
        return str(pa.time32('ms'))
    def time64('us(self) -> str:
        return str(pa.time64('us'))
    def time64('ns(self) -> str:
        return str(pa.time64('ns'))
    def date32(self) -> str:
        return str(pa.date32())
    def date64(self) -> str:
        return str(pa.date64())
    def duration('s(self) -> str:
        return str(pa.duration('s'))
    def duration('ms(self) -> str:
        return str(pa.duration('ms'))
    def duration('us(self) -> str:
        return str(pa.duration('us'))
    def duration('ns(self) -> str:
        return str(pa.duration('ns'))


@dataclass
class Field:
    value: str | int | float |
    dtype: DataType

    def resolve(self) -> pc.Expression:
        if self.dtype == DataType.KEY and isinstance(self.value, str):
            return pc.field(self.value)
        elif self.dtype != DataType.KEY and self.dtype in {DataType.STRING}:
            return pc.scalar(self.value)



class UnaryOperator(StrEnum):
    NEG = "-"
    INV = "~"

    def __call__(self, operand: Any) -> pc.Expression:
        match self:
            case self.NEG:
                return -operand
            case self.INV:
                return ~operand
            case unknown:
                raise NotImplementedError(f"operator '{unknown}' is not supported")


@dataclass
class UnaryExpression:
    operand: "UnaryExpression | BinaryExpression | Field | pc.Expression | int | float | str"
    operator: UnaryOperator

    def resolve(self) -> pc.Expression:
        operand = self.operand
        if isinstance(operand, (UnaryExpression, BinaryExpression)):
            operand = operand.resolve()
        return UnaryOperator(self.operator)(operand)


class BinaryOperator(StrEnum):
    EQ = "=="
    NE = "!="
    LT = "<"
    GT = ">"
    LE = "<="
    GE = ">="
    AND = "&"
    OR = "|"
    XOR = "^"
    CAST = "cast"

    def __call__(self, lhs: Any, rhs: Any) -> pc.Expression:
        match self:
            case self.EQ:
                return lhs == rhs
            case self.NE:
                return lhs != rhs
            case self.LT:
                return lhs < rhs
            case self.GT:
                return lhs > rhs
            case self.LE:
                return lhs <= rhs
            case self.GE:
                return lhs >= rhs
            case self.AND:
                return lhs & rhs
            case self.OR:
                return lhs | rhs
            case self.XOR:
                return lhs ^ rhs
            case self.CAST:
                return pc.cast(lhs, rhs)
            case unknown:
                raise NotImplementedError(f"operator '{unknown}' is not supported")


@dataclass
class BinaryExpression:
    lhs: "UnaryExpression | BinaryExpression | Field | int | float | str"
    rhs: "UnaryExpression | BinaryExpression | Field | int | float | str"
    operator: BinaryOperator

    def resolve(self) -> pc.Expression:
        lhs, rhs = self.lhs, self.rhs
        if isinstance(lhs, (Field, UnaryExpression, BinaryExpression)):
            lhs = lhs.resolve()
        if isinstance(rhs, (Field, UnaryExpression, BinaryExpression)):
            rhs = rhs.resolve
        return BinaryOperator(self.operator)(lhs, rhs)


BinaryExpression(
    lhs=Field("a"),
    rhs="xyz",
    operator="==",
).resolve()