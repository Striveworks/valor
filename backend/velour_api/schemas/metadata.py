from pydantic import BaseModel, validator

from velour_api.schemas.geometry import GeoJSON


class MetaDatum(BaseModel):
    name: str
    value: float | str | GeoJSON

    @validator("name")
    def check_name(cls, v):
        if not isinstance(v, str):
            raise ValueError
        return v

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

    # @property
    # def geo(self) ->
