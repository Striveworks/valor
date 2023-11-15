from pydantic import BaseModel, field_validator

from velour_api.schemas.geometry import (
    BasicPolygon,
    MultiPolygon,
    Point,
    Polygon,
)


class GeoJSONPoint(BaseModel):
    type: str
    coordinates: list[float | int]

    @field_validator("type")
    @classmethod
    def check_type(cls, v):
        if v != "Point":
            raise ValueError("Incorrect geometry type.")
        return v

    @field_validator("coordinates")
    @classmethod
    def check_coordinates(cls, v):
        if len(v) != 2:
            raise ValueError("Incorrect number of points.")
        return v

    def point(self) -> Point:
        return Point(
            x=self.coordinates[0],
            y=self.coordinates[1],
        )


class GeoJSONPolygon(BaseModel):
    type: str
    coordinates: list[list[list[float | int]]]

    @field_validator("type")
    @classmethod
    def check_type(cls, v):
        if v != "Polygon":
            raise ValueError("Incorrect geometry type.")
        return v

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


class GeoJSONMultiPolygon(BaseModel):
    type: str
    coordinates: list[list[list[list[float | int]]]]

    @field_validator("type")
    @classmethod
    def check_type(cls, v):
        if v != "MultiPolygon":
            raise ValueError("Incorrect geometry type.")

        return v

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

    def shape(self):
        if isinstance(self.geometry, GeoJSONPoint):
            return self.geometry.point()
        elif isinstance(self.geometry, GeoJSONPolygon):
            return self.geometry.polygon()
        elif isinstance(self.geometry, GeoJSONMultiPolygon):
            return self.geometry.multipolygon()
        else:
            raise ValueError
