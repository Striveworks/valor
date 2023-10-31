from velour import schemas


def test_metadata_geojson():
    # @TODO: Implement GeoJSON
    schemas.GeoJSON(type="this shouldnt work", coordinates=[])
