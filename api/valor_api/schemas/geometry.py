import io
from base64 import b64decode, b64encode
from typing import Any

import numpy as np
import PIL.Image
from geoalchemy2.functions import (
    ST_AddBand,
    ST_AsRaster,
    ST_GeomFromGeoJSON,
    ST_MakeEmptyRaster,
    ST_MapAlgebra,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from sqlalchemy import ScalarSelect, select

from valor_api.schemas.validators import deserialize, validate_geojson


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
        return deserialize(class_name=cls.__name__, values=values)

    @classmethod
    def from_geojson(cls, geojson: dict):
        validate_geojson(class_name=cls.__name__, geojson=geojson)
        value = geojson.get("coordinates")
        if not isinstance(value, list):
            raise TypeError("Coordinates should contain a list.")
        return cls(value=tuple(value))

    def to_geojson(self) -> dict:
        return {"type": "Point", "coordinates": list(self.value)}

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
        return deserialize(class_name=cls.__name__, values=values)

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
        return deserialize(class_name=cls.__name__, values=values)

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
        return deserialize(class_name=cls.__name__, values=values)

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
        return deserialize(class_name=cls.__name__, values=values)

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
        return deserialize(class_name=cls.__name__, values=values)

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

        Returns
        -------
        BoundingBox
            The bounding box created from the extrema.
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
    def from_geojson(cls, geojson: dict):
        validate_geojson(class_name="Polygon", geojson=geojson)
        return cls(value=Polygon.from_geojson(geojson).value)

    def to_geojson(self) -> dict:
        return Polygon(value=self.value).to_geojson()

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
    value : list[list[list[tuple[int | float, int | float]]]]
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
        return deserialize(class_name=cls.__name__, values=values)

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
            "type": "MultiPolygon",
            "coordinates": [
                [
                    [list(point) for point in subpolygon]
                    for subpolygon in polygon
                ]
                for polygon in self.value
            ],
        }

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


class Raster(BaseModel):
    """
    Describes a raster in geometric space.

    Attributes
    ----------
    mask : str
        The mask describing the raster.
    geometry : BoundingBox | Polygon | MultiPolygon, optional
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
        geometry : BoundingBox | Polygon | MultiPolygon
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
                ST_GeomFromGeoJSON(self.geometry.to_geojson()),
                1.0,  # scalex
                1.0,  # scaley
                "8BUI",  # pixeltype
                1,  # value
                0,  # nodataval
            )
            return select(
                # geom_raster,
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
