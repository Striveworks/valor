import pytest
from pydantic import ValidationError

from valor_api import schemas


def test_info_APIVersion():
    # valid
    schemas.APIVersion(api_version="1.1.1")

    # invalid
    with pytest.raises(ValidationError):
        schemas.APIVersion(api_version=1)

    with pytest.raises(ValidationError):
        schemas.APIVersion()
