import zoneinfo
from datetime import datetime, timedelta

import pytest

from valor_lite.filtering import DataType, Field, TimePrecision, Value


def test_filtering_value_casting():
    x = Value(True)
    assert x.dtype == DataType.bool_()
    y = x.cast(DataType.int8())
    assert str(y) == "1"
    assert y.dtype == DataType.int8()


def test_filtering_field_casting():
    x = Field("x")
    assert str(x) == "x"
    y = x.cast(DataType.int16())
    assert (
        str(y)
        == "cast(x, {to_type=int16, allow_int_overflow=false, allow_time_truncate=false, allow_time_overflow=false, allow_decimal_truncate=false, allow_float_truncate=false, allow_invalid_utf8=false})"
    )


def test_filtering_datetime():
    timestamp = datetime.now()

    a = Value(timestamp)
    assert a.dtype == DataType.timestamp(TimePrecision.MICROSECOND)

    b = Value(timestamp.date())
    assert b.dtype == DataType.date32()

    c = Value(timestamp.time())
    assert c.dtype == DataType.time64(TimePrecision.MICROSECOND)

    d = Value(timedelta(days=1))
    assert d.dtype == DataType.duration(TimePrecision.MICROSECOND)

    tz = "America/New_York"
    e = Value(timestamp.astimezone(tz=zoneinfo.ZoneInfo(tz)))
    assert e.dtype == DataType.timestamp(TimePrecision.MICROSECOND, tz=tz)
    assert str(e.dtype) == "timestamp[us, tz=America/New_York]"

    # unit validation
    for dtype_factory in [
        DataType.timestamp,
        DataType.time32,
        DataType.time64,
        DataType.date32,
        DataType.date64,
        DataType.duration,
    ]:
        with pytest.raises(ValueError) as e:
            dtype_factory(TimePrecision("abc"))  # type: ignore - testing

    # timezone validation
    with pytest.raises(zoneinfo.ZoneInfoNotFoundError):
        DataType.timestamp(TimePrecision.MILLISECOND, tz="does_not_exist")
