from velour_api.backend import state
from velour_api.schemas.metadata import MetaDatum


def _metadatum_mapping(metadatum: schemas.DatumMetadatum) -> dict:
    ret = {"name": metadatum.name}
    val = metadatum.value
    if isinstance(val, str):
        ret["string_value"] = val
    elif isinstance(val, float):
        ret["numeric_value"] = val
    elif isinstance(val, dict):
        ret["geo"] = ST_GeomFromGeoJSON(json.dumps(val))
    else:
        raise ValueError(
            f"Got unexpected value {metadatum.value} for metadatum"
        )
    return ret


def query_by_metadata(metadata: list[MetaDatum]):
    """Returns a subquery of ground truth / predictions that meet the criteria."""
    pass
