from geoalchemy2.shape import to_shape
from pydantic import BaseModel, field_validator

from velour_api.schemas.geometry import (
    BasicPolygon,
    MultiPolygon,
    Point,
    Polygon,
)


class GeoJSONPoint(BaseModel):
    type: str
    coordinates: list[float]

    @field_validator("type")
    @classmethod
    def check_type(cls, v):
        if v != "Point":
            raise ValueError("Incorrect geometry type.")

    @field_validator("coordinates")
    @classmethod
    def check_coordinates(cls, v):
        if len(v) != 2:
            raise ValueError("Incorrect number of points.")

    def point(self) -> Point:
        return Point(
            x=self.coordinates[0],
            y=self.coordinates[1],
        )

    def to_dict(self) -> dict[str | list[list[list[str]]]]:
        return {"type": "Point", "coordinates": self.coordinates}


class GeoJSONPolygon(BaseModel):
    type: str
    coordinates: list[list[list[float]]]

    @field_validator("type")
    @classmethod
    def check_type(cls, v):
        if v != "Polygon":
            raise ValueError("Incorrect geometry type.")

    def polygon(self) -> Polygon:
        polygons = [
            BasicPolygon(
                points=[Point(x=coord[0], y=coord[1]) for coord in poly]
            )
            for poly in self.coordinates
        ]
        if not polygons:
            raise ValueError("Invalid polygons.")
        return Polygon(
            boundary=polygons[0],
            holes=polygons[1:] if len(polygons) > 1 else None,
        )

    def to_dict(self) -> dict[str | list[list[list[str]]]]:
        return {"type": "Polygon", "coordinates": self.coordinates}


class GeoJSONMultiPolygon(BaseModel):
    type: str
    coordinates: list[list[list[list[float]]]]

    @field_validator("type")
    @classmethod
    def check_type(cls, v):
        if v != "MultiPolygon":
            raise ValueError("Incorrect geometry type.")

    def multipolygon(self) -> MultiPolygon:
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

    def to_dict(self) -> dict[str | list[list[list[str]]]]:
        return {"type": "MultiPolygon", "coordinates": self.coordinates}


# GeoJSON Standard
class GeoJSON(BaseModel):
    geometry: GeoJSONPoint | GeoJSONPolygon | GeoJSONMultiPolygon

    @classmethod
    def from_dict(cls, data: dict):
        if "type" not in data:
            raise ValueError("missing geojson type")
        if "coordinates" not in data:
            raise ValueError("missing geojson coordinates")

        if data["type"] == "Point":
            return cls(geometry=GeoJSONPoint(**data))
        elif data["type"] == "Polygon":
            return cls(geometry=GeoJSONPolygon(**data))
        elif data["type"] == "MultiPolygon":
            return cls(geometry=GeoJSONMultiPolygon(**data))
        else:
            raise ValueError("Unsupported type.")

    @classmethod
    def from_wkt(cls, wkt: str):
        wkt_str = to_shape(wkt).wkt

        if "point" in wkt_str.lower():
            # TODO might be incorrect
            start_index = 7
            geometry = "Point"
        elif "polygon" in wkt_str.lower():
            start_index = 10
            geometry = "Polygon"
        elif "multipolygon" in wkt_str.lower():
            start_index = 13
            geometry = "MultiPolygon"
        else:
            raise ValueError("Unsupported type.")

        wkt_split = wkt_str[start_index:-2].split(",")

        coordinates = [
            [
                tuple(map(float, coord_str.strip().split()))
                for coord_str in wkt_split
            ]
        ]

        return cls.from_dict({"type": geometry, "coordinates": coordinates})

    def shape(self):
        if isinstance(self.geometry, GeoJSONPoint):
            return self.geometry.point()
        elif isinstance(self.geometry, GeoJSONPolygon):
            return self.geometry.polygon()
        elif isinstance(self.geometry, GeoJSONMultiPolygon):
            return self.geometry.multipolygon()
        else:
            raise ValueError

    def to_dict(self):
        if isinstance(self.geometry, GeoJSONPoint):
            return self.geometry.to_dict()
        elif isinstance(self.geometry, GeoJSONPolygon):
            return self.geometry.to_dict()
        elif isinstance(self.geometry, GeoJSONMultiPolygon):
            return self.geometry.to_dict()
        else:
            raise ValueError
