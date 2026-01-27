from __future__ import annotations

from enum import StrEnum
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import pyarrow as pa
import pyarrow.compute as pc


class TimePrecision(StrEnum):
    NANOSECOND = "ns"
    MICROSECOND = "us"
    MILLISECOND = "ms"
    SECOND = "s"


class DataType:
    def __init__(self, dtype: pa.DataType):
        self._dtype = dtype

    @property
    def value(self) -> str:
        return str(self._dtype)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, DataType):
            return self.value == other.value
        return self.value == other

    def __str__(self) -> str:
        return self.value

    def to_arrow(self) -> pa.DataType:
        return self._dtype

    @classmethod
    def null(cls) -> DataType:
        return cls(pa.null())

    @classmethod
    def bool_(cls) -> DataType:
        return cls(pa.bool_())

    @classmethod
    def string(cls) -> DataType:
        return cls(pa.string())

    @classmethod
    def int8(cls) -> DataType:
        return cls(pa.int8())

    @classmethod
    def int16(cls) -> DataType:
        return cls(pa.int16())

    @classmethod
    def int32(cls) -> DataType:
        return cls(pa.int32())

    @classmethod
    def int64(cls) -> DataType:
        return cls(pa.int64())

    @classmethod
    def uint8(cls) -> DataType:
        return cls(pa.uint8())

    @classmethod
    def uint16(cls) -> DataType:
        return cls(pa.uint16())

    @classmethod
    def uint32(cls) -> DataType:
        return cls(pa.uint32())

    @classmethod
    def uint64(cls) -> DataType:
        return cls(pa.uint64())

    @classmethod
    def float16(cls) -> DataType:
        return cls(pa.float16())

    @classmethod
    def float32(cls) -> DataType:
        return cls(pa.float32())

    @classmethod
    def float64(cls) -> DataType:
        return cls(pa.float64())

    @classmethod
    def timestamp(cls, unit: TimePrecision, tz: str | None = None) -> DataType:
        if unit not in {
            TimePrecision.SECOND,
            TimePrecision.MILLISECOND,
            TimePrecision.MICROSECOND,
            TimePrecision.NANOSECOND,
        }:
            raise ValueError(f"timestamp does not support unit: '{unit}'")
        if tz:
            ZoneInfo(tz)  # check if valid timezone
            return cls(pa.timestamp(unit.value, tz))
        return cls(pa.timestamp(unit.value))

    @classmethod
    def time32(cls, unit: TimePrecision) -> DataType:
        if unit.value == "s" or unit.value == "ms":
            return cls(pa.time32(unit.value))
        raise ValueError(f"time32 does not support unit: '{unit}'")

    @classmethod
    def time64(cls, unit: TimePrecision) -> DataType:
        if unit.value == "us" or unit.value == "ns":
            return cls(pa.time64(unit.value))
        raise ValueError(f"time64 does not support unit: '{unit}'")

    @classmethod
    def date32(cls) -> DataType:
        return cls(pa.date32())

    @classmethod
    def date64(cls) -> DataType:
        return cls(pa.date64())

    @classmethod
    def duration(cls, unit: TimePrecision) -> DataType:
        if unit not in {
            TimePrecision.SECOND,
            TimePrecision.MILLISECOND,
            TimePrecision.MICROSECOND,
            TimePrecision.NANOSECOND,
        }:
            raise ValueError(f"timestamp does not support unit: '{unit}'")
        return cls(pa.duration(unit.value))


class _Symbol:
    def __init__(self, field_or_scalar: pc.Expression):
        self._symbol = field_or_scalar

    def __str__(self) -> str:
        return str(self._symbol)

    def __eq__(self, other: Any) -> Expression:  # type: ignore[reportIncompatibleMethodOverride]
        other = other._symbol if isinstance(other, _Symbol) else other
        return Expression(self._symbol == other)

    def __ne__(self, other: Any) -> Expression:  # type: ignore[reportIncompatibleMethodOverride]
        other = other._symbol if isinstance(other, _Symbol) else other
        return Expression(self._symbol != other)

    def __gt__(self, other: Any) -> Expression:
        other = other._symbol if isinstance(other, _Symbol) else other
        return Expression(self._symbol > other)

    def __lt__(self, other: Any) -> Expression:
        other = other._symbol if isinstance(other, _Symbol) else other
        return Expression(self._symbol < other)

    def __ge__(self, other: Any) -> Expression:
        other = other._symbol if isinstance(other, _Symbol) else other
        return Expression(self._symbol >= other)

    def __le__(self, other: Any) -> Expression:
        other = other._symbol if isinstance(other, _Symbol) else other
        return Expression(self._symbol <= other)

    def isin(self, values: Iterable[Any]) -> Expression:
        values = {v._symbol if isinstance(v, _Symbol) else v for v in values}
        return Expression(self._symbol.isin(values))

    def is_null(self, nan_is_null: bool = False) -> Expression:
        return Expression(self._symbol.is_null(nan_is_null=nan_is_null))

    def is_nan(self) -> Expression:
        return Expression(self._symbol.is_nan())

    def is_valid(self) -> Expression:
        return Expression(self._symbol.is_valid())  # type: ignore[reportGeneralTypeIssue] - pyarrow typing confused


class Field(_Symbol):
    def __init__(self, name: str):
        self._name = name
        super().__init__(pc.field(name))

    @property
    def name(self) -> str:
        return self._name

    def cast(
        self,
        dtype: DataType,
        *,
        allow_int_overflow: bool | None = None,
        allow_time_truncate: bool | None = None,
        allow_time_overflow: bool | None = None,
        allow_decimal_truncate: bool | None = None,
        allow_float_truncate: bool | None = None,
        allow_invalid_utf8: bool | None = None,
    ) -> Expression:
        """
        Apply a type cast onto the field.


        Parameters
        ----------
        expression : pyarrow.compute.Expression
            The expression to type cast.
        dtype : str
            The data type to cast to.
        allow_int_overflow : bool, default False
            Whether integer overflow is allowed when casting.
        allow_time_truncate : bool, default False
            Whether time precision truncation is allowed when casting.
        allow_time_overflow : bool, default False
            Whether date/time range overflow is allowed when casting.
        allow_decimal_truncate : bool, default False
            Whether decimal precision truncation is allowed when casting.
        allow_float_truncate : bool, default False
            Whether floating-point precision truncation is allowed when casting.
        allow_invalid_utf8 : bool, default False
            Whether producing invalid utf8 data is allowed when casting.

        Returns
        -------
        Expression
            An expression to be evaluated at runtime.
        """
        options = pc.CastOptions(
            target_type=dtype.to_arrow(),
            allow_int_overflow=allow_int_overflow,
            allow_time_truncate=allow_time_truncate,
            allow_time_overflow=allow_time_overflow,
            allow_decimal_truncate=allow_decimal_truncate,
            allow_float_truncate=allow_float_truncate,
            allow_invalid_utf8=allow_invalid_utf8,
        )
        expr = self._symbol.cast(options=options)  # type: ignore[reportGeneralTypeIssues] - pyarrow issue
        return Expression(expr)


class Value(_Symbol):
    def __init__(self, value: Any):
        # pyarrow.scalar creates an arrow-compatible value
        self._value = pa.scalar(value)
        # pyarrow.compute.scalar creates a symbolic expression
        super().__init__(pc.scalar(value))

    @property
    def value(self) -> Any:
        return self._value

    @property
    def dtype(self) -> DataType:
        return DataType(self._value.type)

    def cast(
        self,
        dtype: DataType,
        *,
        allow_int_overflow: bool | None = None,
        allow_time_truncate: bool | None = None,
        allow_time_overflow: bool | None = None,
        allow_decimal_truncate: bool | None = None,
        allow_float_truncate: bool | None = None,
        allow_invalid_utf8: bool | None = None,
    ) -> Value:
        """
        Apply a type cast onto the value.

        This creates a new value of the desired type.

        Parameters
        ----------
        expression : pyarrow.compute.Expression
            The expression to type cast.
        dtype : str
            The data type to cast to.
        allow_int_overflow : bool, default False
            Whether integer overflow is allowed when casting.
        allow_time_truncate : bool, default False
            Whether time precision truncation is allowed when casting.
        allow_time_overflow : bool, default False
            Whether date/time range overflow is allowed when casting.
        allow_decimal_truncate : bool, default False
            Whether decimal precision truncation is allowed when casting.
        allow_float_truncate : bool, default False
            Whether floating-point precision truncation is allowed when casting.
        allow_invalid_utf8 : bool, default False
            Whether producing invalid utf8 data is allowed when casting.

        Returns
        -------
        Value
            A copied value with the desired data type.
        """
        options = pc.CastOptions(
            target_type=dtype.to_arrow(),
            allow_int_overflow=allow_int_overflow,
            allow_time_truncate=allow_time_truncate,
            allow_time_overflow=allow_time_overflow,
            allow_decimal_truncate=allow_decimal_truncate,
            allow_float_truncate=allow_float_truncate,
            allow_invalid_utf8=allow_invalid_utf8,
        )
        return Value(self._value.cast(options=options))


class Expression:
    def __init__(self, expression: pc.Expression):
        self._expr = expression

    def __str__(self) -> str:
        return str(self._expr)

    def __and__(self, other: Expression) -> Expression:
        return Expression(self._expr & other._expr)

    def __or__(self, other: Expression) -> Expression:
        return Expression(self._expr | other._expr)

    def __xor__(self, other: Expression) -> Expression:
        return Expression(pc.xor(self._expr, other._expr))  # type: ignore[reportAttributeAccessIssue]

    def __iand__(self, other: Expression) -> Expression:
        self._expr &= other._expr
        return self

    def __ior__(self, other: Expression) -> Expression:
        self._expr |= other._expr
        return self

    def __ixor__(self, other: Expression) -> Expression:
        self._expr = pc.xor(self._expr, other._expr)  # type: ignore[reportAttributeAccessIssue]
        return self

    def __inv__(self) -> Expression:
        return Expression(pc.invert(self._expr))  # type: ignore[reportAttributeAccessIssue]

    def is_null(self, nan_is_null: bool = False) -> Expression:
        return Expression(self._expr.is_null(nan_is_null=nan_is_null))

    def is_nan(self, nan_is_null: bool = False) -> Expression:
        return Expression(self._expr.is_nan())

    def is_valid(self) -> Expression:
        return Expression(self._expr.is_valid())  # type: ignore[reportGeneralTypeIssue] - pyarrow typing confused

    def to_arrow(self):
        return self._expr
