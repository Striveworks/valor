import datetime
from copy import deepcopy


def _validate_href(value: str):
    if not isinstance(value, str):
        raise TypeError("`href` key should have a `str` as its value.")
    if not (value.startswith("http://") or value.startswith("https://")):
        raise ValueError("`href` must start with http:// or https://")


def validate_metadata(metadata: dict):
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
            _validate_href(value)


def dump_metadata(metadata: dict) -> dict:
    """Ensures that all nested attributes are numerics or str types."""
    metadata = deepcopy(metadata)
    for key, value in metadata.items():
        if isinstance(value, datetime.datetime):
            metadata[key] = {"datetime": value.isoformat()}
        elif isinstance(value, datetime.date):
            metadata[key] = {"date": value.isoformat()}
        elif isinstance(value, datetime.time):
            metadata[key] = {"time": value.isoformat()}
        elif isinstance(value, datetime.timedelta):
            metadata[key] = {"duration": str(value.total_seconds())}
    return metadata


def load_metadata(metadata: dict) -> dict:
    """Reconstructs nested objects from primitive types."""
    metadata = deepcopy(metadata)
    for key, value in metadata.items():
        if isinstance(value, dict):
            if "datetime" in value:
                metadata[key] = datetime.datetime.fromisoformat(
                    value["datetime"]
                )
            elif "date" in value:
                metadata[key] = datetime.date.fromisoformat(value["date"])
            elif "time" in value:
                metadata[key] = datetime.time.fromisoformat(value["time"])
            elif "duration" in value:
                metadata[key] = datetime.timedelta(
                    seconds=float(value["duration"])
                )
    return metadata
