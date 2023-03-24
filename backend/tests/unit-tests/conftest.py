import pytest

from velour_api.schemas import Image


@pytest.fixture
def img() -> Image:
    return Image(uid="", height=1098, width=4591)
