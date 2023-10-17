from pydantic import BaseModel


# @TODO placeholder
class DateTime(BaseModel):
    timestamp: int
    timestamp_local: int
    date: int
    time: int
    time_local: int
    interval: int


class Metadatum(BaseModel):
    key: str
    value: float | str | DateTime

    @property
    def string_value(self) -> str | None:
        if isinstance(self.value, str):
            return self.value
        return None

    @property
    def numeric_value(self) -> float | None:
        if isinstance(self.value, float):
            return self.value
        return None
