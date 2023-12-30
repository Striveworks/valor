from pydantic import BaseModel, field_validator

from velour_api.schemas.geometry import (
    BasicPolygon,
    MultiPolygon,
    Point,
    Polygon,
)


class GeoJSONPoint(BaseModel):
    """
    Describes a point in geospatial coordinates.

    Attributes
    ----------
    type : str
        The type of GeoJSON. Should be "Point" for this class.
    coordinates : List[float | int]
        A list of coordinates describing where the `Point` lies.

    Raises
    ------
    ValueError
        If the type isn't correct.
        If passed an incorrect number of coordinates.
    """

    type: str
    coordinates: list[float | int]

    @field_validator("type")
    @classmethod
    def _check_type(cls, v):
        """Validate the GeoJSON type."""
        if v != "Point":
            raise ValueError("Incorrect geometry type.")
        return v

    @field_validator("coordinates")
    @classmethod
    def _check_coordinates(cls, v):
        """Validate the number of coordinates."""
        if len(v) != 2:
            raise ValueError("Incorrect number of points.")
        return v

    def geometry(self) -> Point:
        """
        Converts the GeoJSON into a Point object.

        Returns
        ----------
        Point
            A geometric point.
        """
        return Point(
            x=self.coordinates[0],
            y=self.coordinates[1],
        )
    
    def wkt(self) -> str:
        return self.geometry().wkt()


class GeoJSONPolygon(BaseModel):
    """
    Describes a polygon in geospatial coordinates.

    Attributes
    ----------
    type : str
        The type of GeoJSON. Should be "Polygon" for this class.
    coordinates : List[List[List[float | int]]]
        A list of coordinates describing where the `Polygon` lies.

    Raises
    ------
    ValueError
        If the type isn't correct.
    """

    type: str
    coordinates: list[list[list[float | int]]]

    @field_validator("type")
    @classmethod
    def _check_type(cls, v):
        """Validate the GeoJSON type."""
        if v != "Polygon":
            raise ValueError("Incorrect geometry type.")
        return v

    def geometry(self) -> Polygon:
        """
        Converts the GeoJSON into a Polygon object.

        Returns
        ----------
        Polygon
            A geometric polygon.

        Raises
        ----------
        ValueError
            If the coordinates are empty.
        """
        polygons = [
            BasicPolygon(
                points=[Point(x=coord[0], y=coord[1]) for coord in poly]
            )
            for poly in self.coordinates
        ]
        if not polygons:
            raise ValueError("Incorrect geometry type.")
        return Polygon(
            boundary=polygons[0],
            holes=polygons[1:] if len(polygons) > 1 else None,
        )
    
    def wkt(self) -> str:
        return self.geometry().wkt()


class GeoJSONMultiPolygon(BaseModel):
    """
    Describes a multipolygon in geospatial coordinates.

    Attributes
    ----------
    type : str
        The type of GeoJSON. Should be "MultiPolygon" for this class.
    coordinates : List[List[List[List[float | int]]]]
        A list of coordinates describing where the `MultiPolygon` lies.

    Raises
    ------
    ValueError
        If the type isn't correct.
    """

    type: str
    coordinates: list[list[list[list[float | int]]]]

    @field_validator("type")
    @classmethod
    def _check_type(cls, v):
        if v != "MultiPolygon":
            raise ValueError("Incorrect geometry type.")

        return v

    def geometry(self) -> MultiPolygon:
        """
        Converts the GeoJSON into a MultiPolygon object.

        Returns
        ----------
        MultiPolygon
            A geometric multipolygon.

        Raises
        ----------
        ValueError
            If coordinates are empty.
        """
        multipolygons = []
        for subpolygon in self.coordinates:
            polygons = [
                BasicPolygon(
                    points=[Point(x=coord[0], y=coord[1]) for coord in poly]
                )
                for poly in subpolygon
            ]
            multipolygons.append(
                Polygon(
                    boundary=polygons[0],
                    holes=polygons[1:] if len(polygons) > 1 else None,
                )
            )
        if not multipolygons:
            raise ValueError("Incorrect geometry type.")
        return MultiPolygon(polygons=multipolygons)
    
    def wkt(self) -> str:
        return self.geometry().wkt()


def from_dict(data: dict) -> GeoJSONPoint | GeoJSONPolygon | GeoJSONMultiPolygon:
    """
    Create a GeoJSON from a dictionary.

    Parameters
    ----------
    data : dict
        A dictionary of GeoJSON-like data.

    Returns
    ----------
    GeoJSON
        A GeoJSON object.

    Raises
    ----------
    ValueError
        If the dict doesn't contain a "type" or "coordinates" key.
        If the type of the GeoJSON isn't supported.
    """
    if "type" not in data:
        raise ValueError("missing geojson type")
    if "coordinates" not in data:
        raise ValueError("missing geojson coordinates")

    if data["type"] == "Point":
        return GeoJSONPoint(**data)
    elif data["type"] == "Polygon":
        return GeoJSONPolygon(**data)
    elif data["type"] == "MultiPolygon":
        return GeoJSONMultiPolygon(**data)
    else:
        raise ValueError("Unsupported type.")
