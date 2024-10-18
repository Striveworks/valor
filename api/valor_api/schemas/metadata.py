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

    string: dict[str, str]
    integer: dict[str, int]
    floating: dict[str, float]
    geospatial: dict[str, GeoJSON]
    datetime: dict[str, "datetime"]
    date: dict[str, "date"]
    time: dict[str, "time"]
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _validate(cls, kwargs: Any) -> Any:
        for key in kwargs.keys():
            validate_string_identifier(key)
            for k, v in kwargs[key].items():
                if key == "datetime" and isinstance(kwargs[key][k], str):
                        kwargs[key][k] = datetime.fromisoformat(v)
                elif key == "date" and isinstance(kwargs[key][k], str):
                    kwargs[key][k] = date.fromisoformat(v)
                elif key == "time" and isinstance(kwargs[key][k], str):
                    kwargs[key][k] = time.fromisoformat(v)
        return kwargs
