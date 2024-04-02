from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

from valor_api.schemas.validators import deserialize, validate_geojson


class Point(BaseModel):
    value: tuple[int | float, int | float]
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def deserialize_valor_type(cls, data: Any) -> Any:
        return deserialize(class_name=cls.__name__, data=data)

    @classmethod
    def from_geojson(cls, geojson: dict):
        validate_geojson(class_name=cls.__name__, geojson=geojson)
        value = geojson.get("coordinates")
        if not isinstance(value, list):
            raise TypeError("Coordinates should contain a list.")
        return cls(value=tuple(value))

    def to_geojson(self) -> dict:
        return {"type": "Point", "coordinates": list(self.value)}


class MultiPoint(BaseModel):
    value: list[tuple[int | float, int | float]]
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def deserialize_valor_type(cls, data: Any) -> Any:
        return deserialize(class_name=cls.__name__, data=data)

    @classmethod
    def from_geojson(cls, geojson: dict):
        validate_geojson(class_name=cls.__name__, geojson=geojson)
        value = geojson.get("coordinates")
        if not isinstance(value, list):
            raise TypeError("Coordinates should contain a list.")
        return cls(value=[tuple(point) for point in value])

    def to_geojson(self) -> dict:
        return {
            "type": "MultiPoint",
            "coordinates": [list(point) for point in self.value],
        }


class LineString(BaseModel):
    value: list[tuple[int | float, int | float]]
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def deserialize_valor_type(cls, data: Any) -> Any:
        return deserialize(class_name=cls.__name__, data=data)

    @classmethod
    def from_geojson(cls, geojson: dict):
        validate_geojson(class_name=cls.__name__, geojson=geojson)
        value = geojson.get("coordinates")
        if not isinstance(value, list):
            raise TypeError("Coordinates should contain a list.")
        return cls(value=[tuple(point) for point in value])

    def to_geojson(self) -> dict:
        return {
            "type": "LineString",
            "coordinates": [list(point) for point in self.value],
        }


class MultiLineString(BaseModel):
    value: list[list[tuple[int | float, int | float]]]
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def deserialize_valor_type(cls, data: Any) -> Any:
        return deserialize(class_name=cls.__name__, data=data)

    @classmethod
    def from_geojson(cls, geojson: dict):
        validate_geojson(class_name=cls.__name__, geojson=geojson)
        value = geojson.get("coordinates")
        if not isinstance(value, list):
            raise TypeError("Coordinates should contain a list.")
        return cls(value=[[tuple(point) for point in line] for line in value])

    def to_geojson(self) -> dict:
        return {
            "type": "MultiLineString",
            "coordinates": [
                [list(point) for point in line] for line in self.value
            ],
        }


class Polygon(BaseModel):
    value: list[list[tuple[int | float, int | float]]]
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def deserialize_valor_type(cls, data: Any) -> Any:
        return deserialize(class_name=cls.__name__, data=data)

    @classmethod
    def from_geojson(cls, geojson: dict):
        validate_geojson(class_name=cls.__name__, geojson=geojson)
        value = geojson.get("coordinates")
        if not isinstance(value, list):
            raise TypeError("Coordinates should contain a list.")
        return cls(
            value=[
                [tuple(point) for point in subpolygon] for subpolygon in value
            ]
        )

    def to_geojson(self) -> dict:
        return {
            "type": "Polygon",
            "coordinates": [
                [list(point) for point in subpolygon]
                for subpolygon in self.value
            ],
        }


class MultiPolygon(BaseModel):
    value: list[list[list[tuple[int | float, int | float]]]]
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def deserialize_valor_type(cls, data: Any) -> Any:
        return deserialize(class_name=cls.__name__, data=data)

    @classmethod
    def from_geojson(cls, geojson: dict):
        validate_geojson(class_name=cls.__name__, geojson=geojson)
        value = geojson.get("coordinates")
        if not isinstance(value, list):
            raise TypeError("Coordinates should contain a list.")
        return cls(
            value=[
                [
                    [tuple(point) for point in subpolygon]
                    for subpolygon in polygon
                ]
                for polygon in value
            ]
        )

    def to_geojson(self) -> dict:
        return {
            "type": "Polygon",
            "coordinates": [
                [
                    [list(point) for point in subpolygon]
                    for subpolygon in polygon
                ]
                for polygon in self.value
            ],
        }
