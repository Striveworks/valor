import pytest

from velour_api.schemas import Image


@pytest.fixture
def img() -> Image:
    return Image(uid="", dataset_name="test dataset", height=1098, width=4591)
