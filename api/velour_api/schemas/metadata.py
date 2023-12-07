from pydantic import BaseModel


class DateTime(BaseModel):
    """
    An object describing a date and time.

    Attributes
    ----------
    timestamp : int
        The timestamp in UNIX format.
    timestamp_local : int
        The local timestamp in UNIX format.
    date : int
        The date in UNIX format.
    time : int
        The time in UNIX format.
    time_local : int
        The local time in UNIX format.
    interval : int
        The interval between moments in time.
    """

    timestamp: int
    timestamp_local: int
    date: int
    time: int
    time_local: int
    interval: int


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
