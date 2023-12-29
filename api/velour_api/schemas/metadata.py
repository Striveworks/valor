import datetime

from pydantic import BaseModel, ConfigDict


class DateTime(BaseModel):
    """
    An object describing a date and time.

    Attributes
    ----------
    datetime : str
        Datetime in ISO format.
    """
    datetime: str
    model_config = ConfigDict(extra="forbid")

    @property
    def key(self) -> str:
        return "datetime"
    
    @property
    def value(self) -> str:
        return self.datetime


class Date(BaseModel):
    """
    An object describing a date.

    Attributes
    ----------
    date : str
        Date in ISO format.
    """
    date: str
    model_config = ConfigDict(extra="forbid")

    @property
    def key(self) -> str:
        return "date"
    
    @property
    def value(self) -> str:
        return self.date


class Time(BaseModel):
    """
    An object describing a time.

    Attributes
    ----------
    time : str
        Time in ISO format.
    """
    time: str
    model_config = ConfigDict(extra="forbid")

    @property
    def key(self) -> str:
        return "time"
    
    @property
    def value(self) -> str:
        return self.time
    

class Duration(BaseModel):
    """
    An object describing a time duration.

    Attributes
    ----------
    duration : str
        Time duration in seconds.
    """
    duration: str
    model_config = ConfigDict(extra="forbid")

    @property
    def key(self) -> str:
        return "duration"
    
    @property
    def value(self) -> str:
        return self.duration


class Metadatum(BaseModel):
    """
    An object describing metadata that can be attached to other objects.

    Attributes
    ----------
    key : str
        The metadata key.
    value : float | str | DateTime
        The metadata value.
    """

    key: str
    value: float | str | DateTime

    model_config = ConfigDict(extra="forbid")
