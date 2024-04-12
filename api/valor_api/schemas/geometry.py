import io
import json
from base64 import b64decode, b64encode
from typing import Any

import numpy as np
import PIL.Image
from geoalchemy2.functions import (
    ST_AddBand,
    ST_AsRaster,
    ST_GeomFromText,
    ST_MakeEmptyRaster,
    ST_MapAlgebra,
    ST_SnapToGrid,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from sqlalchemy import ScalarSelect, select

from valor_api.schemas.validators import (
    deserialize,
    validate_geojson,
    validate_type_box,
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
        """Deserialize if values are in the the valor {type, value} syntax."""
        return deserialize(class_name=cls.__name__, values=values)

    @field_validator("value")
    @classmethod
    def validate_value(cls, v):
        validate_type_point(v)
        return v

    @classmethod
    def from_dict(cls, geojson: dict):
        """
        Create a Point from a GeoJSON in dictionary format.

        Parameters
        ----------
        geojson: dict[str, str | list[int | float]]
            A Point value in GeoJSON format.
        """
        geometry = GeoJSON(**geojson).geometry
        if type(geometry) is not Point:
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
    def from_json(cls, geojson: str):
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
        """Deserialize if values are in the the valor schema {type, value} syntax."""
        return deserialize(class_name=cls.__name__, values=values)

    @field_validator("value")
    @classmethod
    def validate_value(cls, v):
        validate_type_multipoint(v)
        return v

    @classmethod
    def from_dict(cls, geojson: dict):
        """
        Create a MultiPoint from a GeoJSON in dictionary format.

        Parameters
        ----------
        geojson: dict[str, str | list[list[int | float]]]
            A MultiPoint value in GeoJSON format.
        """
        geometry = GeoJSON(**geojson).geometry
        if type(geometry) is not MultiPoint:
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
    def from_json(cls, geojson: str):
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
        """Deserialize if values are in the the valor {type, value} syntax."""
        return deserialize(class_name=cls.__name__, values=values)

    @field_validator("value")
    @classmethod
    def validate_value(cls, v):
        validate_type_linestring(v)
        return v

    @classmethod
    def from_dict(cls, geojson: dict):
        """
        Create a LineString from a GeoJSON in dictionary format.

        Parameters
        ----------
        geojson: dict[str, str | list[list[int | float]]]
            A LineString value in GeoJSON format.
        """
        geometry = GeoJSON(**geojson).geometry
        if type(geometry) is not LineString:
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
    def from_json(cls, geojson: str):
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
        """Deserialize if values are in the the valor {type, value} syntax."""
        return deserialize(class_name=cls.__name__, values=values)

    @field_validator("value")
    @classmethod
    def validate_value(cls, v):
        validate_type_multilinestring(v)
        return v

    @classmethod
    def from_dict(cls, geojson: dict):
        """
        Create a MultiLineString from a GeoJSON in dictionary format.

        Parameters
        ----------
        geojson: dict[str, str | list[list[list[int | float]]]]
            A MultiLineString value in GeoJSON format.
        """
        geometry = GeoJSON(**geojson).geometry
        if type(geometry) is not MultiLineString:
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
    def from_json(cls, geojson: str):
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
        """Deserialize if values are in the the valor {type, value} syntax."""
        return deserialize(class_name=cls.__name__, values=values)

    @field_validator("value")
    @classmethod
    def validate_value(cls, v):
        validate_type_polygon(v)
        return v

    @classmethod
    def from_dict(cls, geojson: dict):
        """
        Create a Polygon from a GeoJSON in dictionary format.

        Parameters
        ----------
        geojson: dict[str, str | list[list[list[int | float]]]]
            A Polygon value in GeoJSON format.
        """
        geometry = GeoJSON(**geojson).geometry
        if type(geometry) is not Polygon:
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
    def from_json(cls, geojson: str):
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
        coords = "),(".join(
            [
                ", ".join([f"{point[0]} {point[1]}" for point in subpolygon])
                for subpolygon in self.value
            ]
        )
        return f"POLYGON (({coords}))"


class Box(BaseModel):
    """
    Describes a Box in (x,y) coordinates.

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
        """Deserialize if values are in the the valor {type, value} syntax."""
        return deserialize(class_name=cls.__name__, values=values)

    @field_validator("value")
    @classmethod
    def validate_value(cls, v):
        validate_type_box(v)
        return v

    @classmethod
    def from_extrema(
        cls,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
    ):
        """
        Create a box from extrema.

        Parameters
        ----------
        xmin: float
            The minimum x-coordinate.
        xmax: float
            The maximum x-coordinate.
        ymin: float
            The minimum y-coordinate.
        ymax: float
            The maximum y-coordinate.
        """
        if xmin >= xmax or ymin >= ymax:
            raise ValueError(
                "Minimums cannot be greater-than or equal to maximums."
            )
        return cls(
            value=[
                [
                    (xmin, ymin),
                    (xmax, ymin),
                    (xmax, ymax),
                    (xmin, ymax),
                    (xmin, ymin),
                ]
            ]
        )

    @classmethod
    def from_dict(cls, geojson: dict):
        """
        Create a Box from a GeoJSON Polygon in dictionary format.

        Parameters
        ----------
        geojson: dict[str, str | list[list[list[int | float]]]]
            A Polygon value in GeoJSON format.
        """
        return cls(value=Polygon.from_dict(geojson).value)

    def to_dict(self) -> dict[str, str | list[list[list[int | float]]]]:
        """
        Create a dictionary that represents the Box using a GeoJSON Polygon.

        Returns
        ----------
        dict[str, str | list[list[list[int | float]]]]
            A Polygon value in GeoJSON format.
        """
        return Polygon(value=self.value).to_dict()

    @classmethod
    def from_json(cls, geojson: str):
        """
        Create a Box from a GeoJSON Polygon in json format.

        Parameters
        ----------
        geojson: str
            A Polygon value in GeoJSON format.
        """
        return cls.from_dict(json.loads(geojson))

    def to_json(self) -> str:
        """
        Create a json string that represents the Box using a GeoJSON Polygon.

        Returns
        ----------
        str
            A Polygon value in GeoJSON format.
        """
        return json.dumps(self.to_dict())

    def to_wkt(self) -> str:
        return Polygon(value=self.value).to_wkt()

    @property
    def xmin(self):
        return min([point[0] for point in self.value[0]])

    @property
    def xmax(self):
        return max([point[0] for point in self.value[0]])

    @property
    def ymin(self):
        return min([point[1] for point in self.value[0]])

    @property
    def ymax(self):
        return max([point[1] for point in self.value[0]])


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
        """Deserialize if values are in the the valor {type, value} syntax."""
        return deserialize(class_name=cls.__name__, values=values)

    @field_validator("value")
    @classmethod
    def validate_value(cls, v):
        validate_type_multipolygon(v)
        return v

    @classmethod
    def from_dict(cls, geojson: dict):
        """
        Create a MultiPolygon from a GeoJSON in dictionary format.

        Parameters
        ----------
        geojson: dict[str, str | list[list[list[list[int | float]]]]]
            A MultiPolygon value in GeoJSON format.
        """
        geometry = GeoJSON(**geojson).geometry
        if type(geometry) is not MultiPolygon:
            raise TypeError(f"GeoJSON is for a different type '{geojson}'.")
        return geometry

    @classmethod
    def from_json(cls, geojson: str):
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
        """Deserialize if values are in the the valor {type, value} syntax."""
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
        return self.geometry.to_wkt()


class Raster(BaseModel):
    """
    Describes a raster in geometric space.

    Attributes
    ----------
    mask : str
        The mask describing the raster.
    geometry : Box | Polygon | MultiPolygon, optional
        Option to define raster by a geometry. Overrides the bitmask.

    Raises
    ------
    ValueError
        If the image format is not PNG.
        If the image mode is not binary.
    """

    mask: str = Field(frozen=True)
    geometry: Box | Polygon | MultiPolygon | None = None
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def deserialize_valor_type(cls, values: Any) -> Any:
        """Deserialize if values are in the the valor {type, value} syntax."""
        return deserialize(class_name=cls.__name__, values=values)

    @field_validator("mask")
    @classmethod
    def _check_png_and_mode(cls, v):
        """Check that the bytes are for a png file and is binary"""
        f = io.BytesIO(b64decode(v))
        img = PIL.Image.open(f)
        f.close()
        if img.format != "PNG":
            raise ValueError(
                f"Expected image format PNG but got {img.format}."
            )
        if img.mode != "1":
            raise ValueError(
                f"Expected image mode to be binary but got mode {img.mode}."
            )
        return v

    @classmethod
    def from_numpy(cls, mask: np.ndarray):
        """
        Create a mask from a numpy array.

        Parameters
        ----------
        mask : np:ndarray
            A numpy array.

        Returns
        ----------
        Raster
            The raster object.

        Raises
        ----------
        ValueError
            If the array has more than two dimensions.
            If the array contains non-boolean elements.
        """
        if len(mask.shape) != 2:
            raise ValueError("raster currently only supports 2d arrays")
        if mask.dtype != bool:
            raise ValueError(
                f"Expecting a binary mask (i.e. of dtype bool) but got dtype {mask.dtype}"
            )
        f = io.BytesIO()
        PIL.Image.fromarray(mask).save(f, format="PNG")
        f.seek(0)
        mask_bytes = f.read()
        f.close()
        return cls(
            mask=b64encode(mask_bytes).decode(),
        )

    @classmethod
    def from_geometry(
        cls,
        geometry: Box | Polygon | MultiPolygon,
        height: int | float,
        width: int | float,
    ):
        """
        Create a Raster object from a geometry.

        Parameters
        ----------
        geometry : Box | Polygon | MultiPolygon
            Defines the bitmask as a geometry. Overrides any existing mask.
        height : int | float
            The intended height of the binary mask.
        width : int | float
            The intended width of the binary mask.

        Returns
        -------
        schemas.Raster
        """
        r = cls.from_numpy(np.full((int(height), int(width)), False))
        r.geometry = geometry
        return r

    def to_numpy(self) -> np.ndarray:
        """
        Convert the mask into an array.

        Returns
        ----------
        np.ndarray
            An array representing a mask.
        """
        mask_bytes = b64decode(self.mask)
        with io.BytesIO(mask_bytes) as f:
            img = PIL.Image.open(f)
            return np.array(img)

    @property
    def mask_bytes(self) -> bytes:
        """
        Serialize the mask into bytes.

        Returns
        ----------
        bytes
            A byte object.

        """
        if not hasattr(self, "_mask_bytes"):
            self._mask_bytes = b64decode(self.mask)
        return self._mask_bytes

    @property
    def array(self) -> np.ndarray:
        """
        Convert the mask into an array.

        Returns
        ----------
        np.ndarray
            An array representing a mask.

        """
        return self.to_numpy()

    @property
    def height(self) -> int:
        """
        Get the height of the raster.

        Returns
        -------
        int
            The height of the binary mask.
        """
        return self.array.shape[0]

    @property
    def width(self) -> int:
        """
        Get the width of the raster.

        Returns
        -------
        int
            The width of the binary mask.
        """
        return self.array.shape[1]

    def to_wkt(self) -> ScalarSelect | bytes:
        """
        Converts raster schema into a postgis-compatible type.

        Returns
        -------
        ScalarSelect | bytes
            A valid input to the models.Annotation.raster column.
        """
        if self.geometry:
            empty_raster = ST_AddBand(
                ST_MakeEmptyRaster(
                    self.width,
                    self.height,
                    0,  # upperleftx
                    0,  # upperlefty
                    1,  # scalex
                    1,  # scaley
                    0,  # skewx
                    0,  # skewy
                    0,  # srid
                ),
                "8BUI",
            )
            geom_raster = ST_AsRaster(
                ST_SnapToGrid(
                    ST_GeomFromText(self.geometry.to_wkt()),
                    1.0,
                ),
                1.0,  # scalex
                1.0,  # scaley
                "8BUI",  # pixeltype
                1,  # value
                0,  # nodataval
            )
            return select(
                ST_MapAlgebra(
                    empty_raster,
                    geom_raster,
                    "[rast2]",
                    "8BUI",
                    "UNION",
                )
            ).scalar_subquery()
        else:
            return self.mask_bytes
