import datetime

from velour.types import MetadataType, ConvertibleMetadataType, DictMetadataType


def _validate_href(value: str):
    if not isinstance(value, str):
        raise TypeError("`href` key should have a `str` as its value.")
    if not (value.startswith("http://") or value.startswith("https://")):
        raise ValueError("`href` must start with http:// or https://")


def validate_metadata(metadata: MetadataType):
    """Validates metadata dictionary."""
    if not isinstance(metadata, dict):
        raise TypeError("`metadata` should be an object of type `dict`.")
    for key, value in metadata.items():
        if not isinstance(key, str):
            raise TypeError("`metadata` key should have type `str`.")
        if not (
            isinstance(value, int)
            or isinstance(value, float)
            or isinstance(value, str)
            or isinstance(value, datetime.datetime)
            or isinstance(value, datetime.date)
            or isinstance(value, datetime.time)
        ):
            raise TypeError(
                "`metadata` value should have type `str`, `int`, `float` or `datetime`."
            )

        # Handle special key-values
        if key == "href":
            if not isinstance(value, str):
                raise TypeError("The metadata key `href` is reserved for values of type `str`.")
            _validate_href(value)


def dump_metadata(metadata: MetadataType) -> ConvertibleMetadataType:
    """Ensures that all nested attributes are numerics or str types."""
    _metadata: ConvertibleMetadataType = {}
    for key, value in metadata.items():
        if isinstance(value, datetime.datetime):
            _metadata[key] = {"datetime": value.isoformat()}
        elif isinstance(value, datetime.date):
            _metadata[key] = {"date": value.isoformat()}
        elif isinstance(value, datetime.time):
            _metadata[key] = {"time": value.isoformat()}
        elif isinstance(value, datetime.timedelta):
            _metadata[key] = {"duration": str(value.total_seconds())}
        else:
            _metadata[key] = value
    return _metadata


def load_metadata(metadata: ConvertibleMetadataType) -> MetadataType:
    """Reconstructs nested objects from primitive types."""
    _metadata: DictMetadataType = {}
    for key, value in metadata.items():
        if isinstance(value, dict):
            if "datetime" in value:
                _metadata[key] = datetime.datetime.fromisoformat(
                    value["datetime"]
                )
            elif "date" in value:
                _metadata[key] = datetime.date.fromisoformat(value["date"])
            elif "time" in value:
                _metadata[key] = datetime.time.fromisoformat(value["time"])
            elif "duration" in value:
                _metadata[key] = datetime.timedelta(
                    seconds=float(value["duration"])
                )
        else:
            _metadata[key] = value
    return _metadata
