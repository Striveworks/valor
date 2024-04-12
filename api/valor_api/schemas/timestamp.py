import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from valor_api.schemas.validators import (
    deserialize,
    validate_type_date,
    validate_type_datetime,
    validate_type_duration,
    validate_type_time,
)


class DateTime(BaseModel):
    """
    An object describing a date and time.

    Attributes
    ----------
    value : str
        Datetime in ISO format.
    """

    value: str
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def deserialize_valor_type(cls, values: Any) -> Any:
        """Special deseraializer for Valor {type, value} formatting."""
        return deserialize(class_name=cls.__name__, values=values)

    @field_validator("value")
    @classmethod
    def validate_value(cls, v: str) -> str:
        """Type validator."""
        validate_type_datetime(v)
        return v

    @classmethod
    def from_datetime(cls, value: datetime.datetime):
        """Construct a class instance from a 'datetime.datetime' object."""
        cls(value=value.isoformat())

    def to_datetime(self):
        """Cast to a 'datetime.datetime' object."""
        return datetime.datetime.fromisoformat(self.value)


class Date(BaseModel):
    """
    An object describing a date.

    Attributes
    ----------
    value : str
        Date in ISO format.
    """

    value: str
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def deserialize_valor_type(cls, values: Any) -> Any:
        """Special deseraializer for Valor {type, value} formatting."""
        return deserialize(class_name=cls.__name__, values=values)

    @field_validator("value")
    @classmethod
    def validate_value(cls, v: str) -> str:
        """Type validator."""
        validate_type_date(v)
        return v

    @classmethod
    def from_date(cls, value: datetime.date):
        """Construct a class instance from a 'datetime.date' object."""
        cls(value=value.isoformat())

    def to_date(self):
        """Cast to a 'datetime.date' object."""
        return datetime.date.fromisoformat(self.value)


class Time(BaseModel):
    """
    An object describing a time.

    Attributes
    ----------
    value : str
        Time in ISO format.
    """

    value: str
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def deserialize_valor_type(cls, values: Any) -> Any:
        """Special deseraializer for Valor {type, value} formatting."""
        return deserialize(class_name=cls.__name__, values=values)

    @field_validator("value")
    @classmethod
    def validate_value(cls, v: str) -> str:
        """Type validator."""
        validate_type_time(v)
        return v

    @classmethod
    def from_time(cls, value: datetime.time):
        """Construct a class instance from a 'datetime.time' object."""
        cls(value=value.isoformat())

    def to_time(self):
        """Cast to a 'datetime.time' object."""
        return datetime.time.fromisoformat(self.value)


class Duration(BaseModel):
    """
    An object describing a time duration.

    Attributes
    ----------
    value : float
        Time duration in seconds.
    """

    value: float
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def deserialize_valor_type(cls, values: Any) -> Any:
        """Special deseraializer for Valor {type, value} formatting."""
        return deserialize(class_name=cls.__name__, values=values)

    @field_validator("value")
    @classmethod
    def validate_value(cls, v: str) -> str:
        """Type validator."""
        validate_type_duration(v)
        return v

    @classmethod
    def from_timedelta(cls, value: datetime.timedelta):
        """Construct a class instance from a 'datetime.timedelta' object."""
        cls(value=value.total_seconds())

    def to_timedelta(self):
        """Cast to a 'datetime.timedelta' object."""
        return datetime.timedelta(seconds=self.value)
