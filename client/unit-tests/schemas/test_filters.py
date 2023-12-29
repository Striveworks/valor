import pytest

import datetime

from velour import Label
from velour.schemas.filters import Filter, ValueFilter

def test_labels_filter():
    f = Filter.create(
        [
            Label.label.in_(
                [
                    Label(key="k1", value="v1"), 
                    Label(key="k2", value="v2")
                ]
            )
        ]
    )
    assert {'k1': 'v1'} in f.labels 
    assert {'k2': 'v2'} in f.labels


def test_value_filter():

    def _test_numeric(value):
        ValueFilter(value=value, operator="==")
        ValueFilter(value=value, operator="!=")
        ValueFilter(value=value, operator=">=")
        ValueFilter(value=value, operator="<=")
        ValueFilter(value=value, operator=">")
        ValueFilter(value=value, operator="<")
        # unsupported operator(s)
        with pytest.raises(ValueError):
            ValueFilter(value=value, operator="@")

    def _test_string(value):
        ValueFilter(value=value, operator="==")
        ValueFilter(value=value, operator="!=")
        # unsupported operator(s)
        with pytest.raises(ValueError):
            ValueFilter(value=value, operator=">=")
        with pytest.raises(ValueError):
            ValueFilter(value=value, operator="<=")
        with pytest.raises(ValueError):
            ValueFilter(value=value, operator=">")
        with pytest.raises(ValueError):
            ValueFilter(value=value, operator="<")
        with pytest.raises(ValueError):
            ValueFilter(value=value, operator="@")

    # int
    _test_numeric(int(123))
    
    # float
    _test_numeric(float(123))

    # string
    _test_string(str(123))

    # datetime.datetime
    _test_numeric(datetime.datetime.now())

    # datetime.date
    _test_numeric(datetime.date.today())

    # datetime.time
    _test_numeric(datetime.datetime.now().time())

    # datetime.timedelta
    _test_numeric(datetime.timedelta(days=1))
