import pytest
from valor_lite.classification import Metric


def test_metric_type_validation():

    # test type attribute
    with pytest.raises(TypeError):
        Metric(
            type=1234,  # type: ignore - testing
            value=1234,
            parameters={},
        )

    # test value attribute
    with pytest.raises(TypeError):
        Metric(
            type="SomeMetric",
            value=[1, 2, 3],  # type: ignore - testing
            parameters={},
        )

    # test parameters attribute
    with pytest.raises(TypeError):
        Metric(
            type="SomeMetric",
            value=1,
            parameters=123,  # type: ignore - testing
        )

    # test parameter keys
    with pytest.raises(TypeError):
        Metric(
            type="SomeMetric",
            value=1,
            parameters={
                1: "hello",
            },  # type: ignore - testing
        )
