from pydantic import BaseModel


class Point(BaseModel):
    value: tuple[float, float]

    @classmethod
    def from_wkt(cls, value: str):
        pass

    @classmethod
    def from_geojson(cls, value: dict):
        pass

    def to_wkt(self):
        pass

    def to_geojson(self):
        pass


class MultiPoint(BaseModel):
    value: list[tuple[float, float]]

    @classmethod
    def from_wkt(cls, value: str):
        pass

    @classmethod
    def from_geojson(cls, value: dict):
        pass

    def to_wkt(self):
        pass

    def to_geojson(self):
        pass


class LineString(BaseModel):
    value: list[tuple[float, float]]

    @classmethod
    def from_wkt(cls, value: str):
        pass

    @classmethod
    def from_geojson(cls, value: dict):
        pass

    def to_wkt(self):
        pass

    def to_geojson(self):
        pass


class MultiLineString(BaseModel):
    value: list[list[tuple[float, float]]]

    @classmethod
    def from_wkt(cls, value: str):
        pass

    @classmethod
    def from_geojson(cls, value: dict):
        pass

    def to_wkt(self):
        pass

    def to_geojson(self):
        pass


class Polygon(BaseModel):
    value: list[list[tuple[float, float]]]

    @classmethod
    def from_wkt(cls, value: str):
        pass

    @classmethod
    def from_geojson(cls, value: dict):
        pass

    def to_wkt(self):
        pass

    def to_geojson(self):
        pass


class MultiPolygon(BaseModel):
    value: list[list[list[tuple[float, float]]]]

    @classmethod
    def from_wkt(cls, value: str):
        pass

    @classmethod
    def from_geojson(cls, value: dict):
        pass

    def to_wkt(self):
        pass

    def to_geojson(self):
        pass
