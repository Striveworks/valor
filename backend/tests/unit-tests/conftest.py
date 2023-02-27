import pytest

from velour_api.schemas import Image


@pytest.fixture
def img() -> Image:
    return Image(uri="", height=1098, width=4591)
