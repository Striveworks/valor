from velour import Label
from velour.schemas.filters import Filter

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