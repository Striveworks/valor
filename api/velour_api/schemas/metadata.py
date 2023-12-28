from pydantic import BaseModel, ConfigDict


class DateTime(BaseModel):
    """
    An object describing a date and time.

    See patterns at:
    https://www.postgresql.org/docs/current/functions-formatting.html

    Attributes
    ----------
    value : str
        Date and/or time value as a string.
    pattern : str
        Template pattern for formatting date and/or time value.
    """

    value: str
    pattern: str

    model_config = ConfigDict(extra="forbid")


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
