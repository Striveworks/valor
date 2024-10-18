import json
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from valor_api.schemas.validators import (
    deserialize,
    validate_geojson,
    validate_type_linestring,
    validate_type_multilinestring,
    validate_type_multipoint,
    validate_type_multipolygon,
    validate_type_point,
    validate_type_polygon,
)


class Point(BaseModel):
    """
    Describes a Point in (x,y) coordinates.

    Attributes
    ----------
    value : tuple[int | float, int | float]
        A list of coordinates describing the Point.

    Raises
    ------
    ValueError
        If the value doesn't conform to the type.
    """

    value: tuple[int | float, int | float]
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def deserialize_valor_type(cls, values: Any) -> Any:
        """Special deseraializer for Valor {type, value} formatting."""
        return deserialize(class_name=cls.__name__, values=values)

    @field_validator("value")
    @classmethod
    def validate_value(
        cls, v: tuple[int | float, int | float]
    ) -> tuple[int | float, int | float]:
        """Type validator."""
        validate_type_point(v)
        return v

    @classmethod
    def from_dict(cls, geojson: dict) -> "Point":
        """
        Create a Point from a GeoJSON in dictionary format.

        Parameters
        ----------
        geojson: dict[str, str | list[int | float]]
            A Point value in GeoJSON format.
        """
        geometry = GeoJSON(**geojson).geometry
        if not isinstance(geometry, Point):
            raise TypeError(f"GeoJSON is for a different type '{geojson}'.")
        return geometry

    def to_dict(self) -> dict[str, str | list[int | float]]:
        """
        Create a dictionary that represents the Point in GeoJSON format.

        Returns
        ----------
        dict[str, str | list[int | float]]
            A Point value in GeoJSON format.
        """
        return {"type": "Point", "coordinates": list(self.value)}

    @classmethod
    def from_json(cls, geojson: str) -> "Point":
        """
        Create a Point from a GeoJSON in json format.

        Parameters
        ----------
        geojson: str
            A Point value in GeoJSON format.
        """
        return cls.from_dict(json.loads(geojson))

    def to_json(self) -> str:
        """
        Create a json string that represents the Point in GeoJSON format.

        Returns
        ----------
        str
            A Point value in GeoJSON format.
        """
        return json.dumps(self.to_dict())

    def to_wkt(self) -> str:
        """
        Casts the geometric object into a string using Well-Known-Text (WKT) Format.

        Returns
        -------
        str
            The WKT formatted string.
        """
        return f"POINT ({self.value[0]} {self.value[1]})"


class MultiPoint(BaseModel):
    """
    Describes a MultiPoint in (x,y) coordinates.

    Attributes
    ----------
    value : list[tuple[int | float, int | float]]
        A list of coordinates describing the MultiPoint.

    Raises
    ------
    ValueError
        If the value doesn't conform to the type.
    """

    value: list[tuple[int | float, int | float]]
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def deserialize_valor_type(cls, values: Any) -> Any:
        """Special deseraializer for Valor {type, value} formatting."""
        return deserialize(class_name=cls.__name__, values=values)

    @field_validator("value")
    @classmethod
    def validate_value(
        cls, v: list[tuple[int | float, int | float]]
    ) -> list[tuple[int | float, int | float]]:
        """Type validator."""
        validate_type_multipoint(v)
        return v

    @classmethod
    def from_dict(cls, geojson: dict) -> "MultiPoint":
        """
        Create a MultiPoint from a GeoJSON in dictionary format.

        Parameters
        ----------
        geojson: dict[str, str | list[list[int | float]]]
            A MultiPoint value in GeoJSON format.
        """
        geometry = GeoJSON(**geojson).geometry
        if not isinstance(geometry, MultiPoint):
            raise TypeError(f"GeoJSON is for a different type '{geojson}'.")
        return geometry

    def to_dict(self) -> dict[str, str | list[list[int | float]]]:
        """
        Create a dictionary that represents the MultiPoint in GeoJSON format.

        Returns
        ----------
        dict[str, str | list[list[int | float]]]
            A MultiPoint value in GeoJSON format.
        """
        return {
            "type": "MultiPoint",
            "coordinates": [list(point) for point in self.value],
        }

    @classmethod
    def from_json(cls, geojson: str) -> "MultiPoint":
        """
        Create a MultiPoint from a GeoJSON in json format.

        Parameters
        ----------
        geojson: str
            A MultiPoint value in GeoJSON format.
        """
        return cls.from_dict(json.loads(geojson))

    def to_json(self) -> str:
        """
        Create a json string that represents the MultiPoint in GeoJSON format.

        Returns
        ----------
        str
            A MultiPoint value in GeoJSON format.
        """
        return json.dumps(self.to_dict())

    def to_wkt(self) -> str:
        """
        Casts the geometric object into a string using Well-Known-Text (WKT) Format.

        Returns
        -------
        str
            The WKT formatted string.
        """
        points = ", ".join(
            [f"({point[0]} {point[1]})" for point in self.value]
        )
        return f"MULTIPOINT ({points})"


class LineString(BaseModel):
    """
    Describes a LineString in (x,y) coordinates.

    Attributes
    ----------
    value : list[tuple[int | float, int | float]]
        A list of coordinates describing the LineString.

    Raises
    ------
    ValueError
        If the value doesn't conform to the type.
    """

    value: list[tuple[int | float, int | float]]
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def deserialize_valor_type(cls, values: Any) -> Any:
        """Special deseraializer for Valor {type, value} formatting."""
        return deserialize(class_name=cls.__name__, values=values)

    @field_validator("value")
    @classmethod
    def validate_value(
        cls, v: list[tuple[int | float, int | float]]
    ) -> list[tuple[int | float, int | float]]:
        """Type validator."""
        validate_type_linestring(v)
        return v

    @classmethod
    def from_dict(cls, geojson: dict) -> "LineString":
        """
        Create a LineString from a GeoJSON in dictionary format.

        Parameters
        ----------
        geojson: dict[str, str | list[list[int | float]]]
            A LineString value in GeoJSON format.
        """
        geometry = GeoJSON(**geojson).geometry
        if not isinstance(geometry, LineString):
            raise TypeError(f"GeoJSON is for a different type '{geojson}'.")
        return geometry

    def to_dict(self) -> dict[str, str | list[list[int | float]]]:
        """
        Create a dictionary that represents the LineString in GeoJSON format.

        Returns
        ----------
        dict[str, str | list[list[int | float]]]
            A LineString value in GeoJSON format.
        """
        return {
            "type": "LineString",
            "coordinates": [list(point) for point in self.value],
        }

    @classmethod
    def from_json(cls, geojson: str) -> "LineString":
        """
        Create a LineString from a GeoJSON in json format.

        Parameters
        ----------
        geojson: str
            A LineString value in GeoJSON format.
        """
        return cls.from_dict(json.loads(geojson))

    def to_json(self) -> str:
        """
        Create a json string that represents the LineString in GeoJSON format.

        Returns
        ----------
        str
            A LineString value in GeoJSON format.
        """
        return json.dumps(self.to_dict())

    def to_wkt(self) -> str:
        """
        Casts the geometric object into a string using Well-Known-Text (WKT) Format.

        Returns
        -------
        str
            The WKT formatted string.
        """
        points = ", ".join([f"{point[0]} {point[1]}" for point in self.value])
        return f"LINESTRING ({points})"


class MultiLineString(BaseModel):
    """
    Describes a MultiLineString in (x,y) coordinates.

    Attributes
    ----------
    value : list[list[tuple[int | float, int | float]]]
        A list of coordinates describing the MultiLineString.

    Raises
    ------
    ValueError
        If the value doesn't conform to the type.
    """

    value: list[list[tuple[int | float, int | float]]]
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def deserialize_valor_type(cls, values: Any) -> Any:
        """Special deseraializer for Valor {type, value} formatting."""
        return deserialize(class_name=cls.__name__, values=values)

    @field_validator("value")
    @classmethod
    def validate_value(
        cls, v: list[list[tuple[int | float, int | float]]]
    ) -> list[list[tuple[int | float, int | float]]]:
        """Type validator."""
        validate_type_multilinestring(v)
        return v

    @classmethod
    def from_dict(cls, geojson: dict) -> "MultiLineString":
        """
        Create a MultiLineString from a GeoJSON in dictionary format.

        Parameters
        ----------
        geojson: dict[str, str | list[list[list[int | float]]]]
            A MultiLineString value in GeoJSON format.
        """
        geometry = GeoJSON(**geojson).geometry
        if not isinstance(geometry, MultiLineString):
            raise TypeError(f"GeoJSON is for a different type '{geojson}'.")
        return geometry

    def to_dict(self) -> dict[str, str | list[list[list[int | float]]]]:
        """
        Create a dictionary that represents the MultiLineString in GeoJSON format.

        Returns
        ----------
        dict[str, str | list[list[list[int | float]]]]
            A MultiLineString value in GeoJSON format.
        """
        return {
            "type": "MultiLineString",
            "coordinates": [
                [list(point) for point in line] for line in self.value
            ],
        }

    @classmethod
    def from_json(cls, geojson: str) -> "MultiLineString":
        """
        Create a MultiLineString from a GeoJSON in json format.

        Parameters
        ----------
        geojson: str
            A MultiLineString value in GeoJSON format.
        """
        return cls.from_dict(json.loads(geojson))

    def to_json(self) -> str:
        """
        Create a json string that represents the MultiLineString in GeoJSON format.

        Returns
        ----------
        str
            A MultiLineString value in GeoJSON format.
        """
        return json.dumps(self.to_dict())

    def to_wkt(self) -> str:
        """
        Casts the geometric object into a string using Well-Known-Text (WKT) Format.

        Returns
        -------
        str
            The WKT formatted string.
        """
        points = "),(".join(
            [
                ", ".join([f"{point[0]} {point[1]}" for point in line])
                for line in self.value
            ]
        )
        return f"MULTILINESTRING (({points}))"


class Polygon(BaseModel):
    """
    Describes a Polygon in (x,y) coordinates.

    Attributes
    ----------
    value : list[list[tuple[int | float, int | float]]]
        A list of coordinates describing the Box.

    Raises
    ------
    ValueError
        If the value doesn't conform to the type.
    """

    value: list[list[tuple[int | float, int | float]]]
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def deserialize_valor_type(cls, values: Any) -> Any:
        """Special deseraializer for Valor {type, value} formatting."""
        return deserialize(class_name=cls.__name__, values=values)

    @field_validator("value")
    @classmethod
    def validate_value(
        cls, v: list[list[tuple[int | float, int | float]]]
    ) -> list[list[tuple[int | float, int | float]]]:
        """Type validator."""
        validate_type_polygon(v)
        return v

    @classmethod
    def from_dict(cls, geojson: dict) -> "Polygon":
        """
        Create a Polygon from a GeoJSON in dictionary format.

        Parameters
        ----------
        geojson: dict[str, str | list[list[list[int | float]]]]
            A Polygon value in GeoJSON format.
        """
        geometry = GeoJSON(**geojson).geometry
        if not isinstance(geometry, Polygon):
            raise TypeError(f"GeoJSON is for a different type '{geojson}'.")
        return geometry

    def to_dict(self) -> dict[str, str | list[list[list[int | float]]]]:
        """
        Create a dictionary that represents the Polygon in GeoJSON format.

        Returns
        ----------
        dict[str, str | list[list[list[int | float]]]]
            A Polygon value in GeoJSON format.
        """
        return {
            "type": "Polygon",
            "coordinates": [
                [list(point) for point in subpolygon]
                for subpolygon in self.value
            ],
        }

    @classmethod
    def from_json(cls, geojson: str) -> "Polygon":
        """
        Create a Polygon from a GeoJSON in json format.

        Parameters
        ----------
        geojson: str
            A Polygon value in GeoJSON format.
        """
        return cls.from_dict(json.loads(geojson))

    def to_json(self) -> str:
        """
        Create a json string that represents the Polygon in GeoJSON format.

        Returns
        ----------
        str
            A Polygon value in GeoJSON format.
        """
        return json.dumps(self.to_dict())

    def to_wkt(self) -> str:
        """
        Casts the geometric object into a string using Well-Known-Text (WKT) Format.

        Returns
        -------
        str
            The WKT formatted string.
        """
        coords = "),(".join(
            [
                ", ".join([f"{point[0]} {point[1]}" for point in subpolygon])
                for subpolygon in self.value
            ]
        )
        return f"POLYGON (({coords}))"


class MultiPolygon(BaseModel):
    """
    Describes a MultiPolygon in (x,y) coordinates.

    Attributes
    ----------
    value : list[list[list[list[int | float]]]]
        A list of coordinates describing the MultiPolygon.

    Raises
    ------
    ValueError
        If the value doesn't conform to the type.
    """

    value: list[list[list[tuple[int | float, int | float]]]]
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def deserialize_valor_type(cls, values: Any) -> Any:
        """Special deseraializer for Valor {type, value} formatting."""
        return deserialize(class_name=cls.__name__, values=values)

    @field_validator("value")
    @classmethod
    def validate_value(
        cls, v: list[list[list[tuple[int | float, int | float]]]]
    ) -> list[list[list[tuple[int | float, int | float]]]]:
        """Type validator."""
        validate_type_multipolygon(v)
        return v

    @classmethod
    def from_dict(cls, geojson: dict) -> "MultiPolygon":
        """
        Create a MultiPolygon from a GeoJSON in dictionary format.

        Parameters
        ----------
        geojson: dict[str, str | list[list[list[list[int | float]]]]]
            A MultiPolygon value in GeoJSON format.
        """
        geometry = GeoJSON(**geojson).geometry
        if not isinstance(geometry, MultiPolygon):
            raise TypeError(f"GeoJSON is for a different type '{geojson}'.")
        return geometry

    @classmethod
    def from_json(cls, geojson: str) -> "MultiPolygon":
        """
        Create a dictionary that represents the MultiPolygon in GeoJSON format.

        Returns
        ----------
        dict[str, str | list[list[list[list[int | float]]]]]
            A MultiPolygon value in GeoJSON format.
        """
        return cls.from_dict(json.loads(geojson))

    def to_dict(self) -> dict[str, str | list[list[list[list[int | float]]]]]:
        """
        Create a MultiPolygon from a GeoJSON in json format.

        Parameters
        ----------
        geojson: str
            A MultiPolygon value in GeoJSON format.
        """
        return {
            "type": "MultiPolygon",
            "coordinates": [
                [
                    [list(point) for point in subpolygon]
                    for subpolygon in polygon
                ]
                for polygon in self.value
            ],
        }

    def to_json(self) -> str:
        """
        Create a json string that represents the MultiPolygon in GeoJSON format.

        Returns
        ----------
        str
            A MultiPolygon value in GeoJSON format.
        """
        return json.dumps(self.to_dict())

    def to_wkt(self) -> str:
        """
        Casts the geometric object into a string using Well-Known-Text (WKT) Format.

        Returns
        -------
        str
            The WKT formatted string.
        """
        polygons = [
            "("
            + "),(".join(
                [
                    ",".join(
                        [f"{point[0]} {point[1]}" for point in subpolygon]
                    )
                    for subpolygon in polygon
                ]
            )
            + ")"
            for polygon in self.value
        ]
        coords = "),(".join(polygons)
        return f"MULTIPOLYGON (({coords}))"


class GeoJSON(BaseModel):
    type: str
    coordinates: list[float] | list[list[float]] | list[
        list[list[float]]
    ] | list[list[list[list[float]]]]

    @model_validator(mode="before")
    @classmethod
    def deserialize_valor_type(cls, values: Any) -> Any:
        """Special deseraializer for Valor {type, value} formatting."""
        values = deserialize(class_name=cls.__name__, values=values)
        validate_geojson(values)
        return values

    @property
    def geometry(
        self,
    ) -> Point | MultiPoint | LineString | MultiLineString | Polygon | MultiPolygon:
        map_str_to_type = {
            "Point": Point,
            "MultiPoint": MultiPoint,
            "LineString": LineString,
            "MultiLineString": MultiLineString,
            "Polygon": Polygon,
            "MultiPolygon": MultiPolygon,
        }
        return map_str_to_type[self.type](value=self.coordinates)

    def to_wkt(self) -> str:
        """
        Converts the GeoJSON to a string in Well-Known-Text (WKT) formatting.

        Returns
        -------
        str
            The geometry in WKT format.
        """
        return self.geometry.to_wkt()
