from datetime import date, datetime, time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from valor_api.schemas.geojson import GeoJSON
from valor_api.schemas.validators import validate_string_identifier


class Metadata(BaseModel):
    """
    Metadata schema.

    Attributes
    ----------
    string : dict[str, str]
        A dictionary containing string-type values.
    integer : dict[str, int], optional
        A dictionary containing integer values.
    floating : dict[str, float], optional
        A dictionary containing floating-point values.
    geospatial : GeoJSON, optional
        A geospatial point or region represented by GeoJSON.
    datetime : str, optional
        A datetime in ISO string format.
    date : str, optional
        A date in ISO string format.
    time : str, optional
        A time in ISO string format.
    """

    string: dict[str, str] | None = None
    integer: dict[str, int] | None = None
    floating: dict[str, float] | None = None
    geospatial: dict[str, GeoJSON] | None = None
    datetime: dict[str, "datetime"] | None = None
    date: dict[str, "date"] | None = None
    time: dict[str, "time"] | None = None
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _validate(cls, kwargs: Any) -> Any:
        for key in kwargs.keys():
            validate_string_identifier(key)
            for k, v in kwargs[key].items():
                if key == "datetime":
                    kwargs[key][k] = datetime.fromisoformat(v)
                elif key == "date":
                    kwargs[key][k] = date.fromisoformat(v)
                elif key == "time":
                    kwargs[key][k] = time.fromisoformat(v)
        return kwargs
