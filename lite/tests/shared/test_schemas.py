from dataclasses import dataclass

import pytest
from valor_lite.schemas import Metric, _BaseMetric


def test_metric():

    with pytest.raises(TypeError) as e:
        Metric(type="TestMetric", value=[1, 2, 3], parameters={})  # type: ignore - testing
    assert "Metric value must be of type `int`, `float` or `dict`." in str(e)


def test_base_metric():
    class Test1(_BaseMetric):
        x: int
        y: float
        label: str

    with pytest.raises(TypeError) as e:
        Test1().to_metric()
    assert "is not a dataclass." in str(e)

    with pytest.raises(TypeError) as e:
        Test1().to_dict()
    assert "is not a dataclass." in str(e)

    @dataclass
    class Test2(_BaseMetric):
        x: int
        y: float
        label: str
        score_threshold: float

    assert Test2(x=2, y=3, label="dog", score_threshold=0.5).to_dict() == {
        "type": "Test2",
        "value": {
            "x": 2,
            "y": 3,
        },
        "parameters": {"label": "dog", "score_threshold": 0.5},
    }

    @dataclass
    class Test3(_BaseMetric):
        value: int
        label: str
        score_threshold: float

    assert Test3(value=2, label="dog", score_threshold=0.5).to_dict() == {
        "type": "Test3",
        "value": 2,
        "parameters": {"label": "dog", "score_threshold": 0.5},
    }
