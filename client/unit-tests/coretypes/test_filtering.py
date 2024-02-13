from velour import Annotation, Constraint, Dataset, Filter, Label, Model
from velour.coretypes import _format_filter


def test__format_filter():

    filter_object = Filter(
        dataset_names=["a", "b", "c"],
        model_names=["x", "y", "z"],
        label_scores=[Constraint(value=0.75, operator=">")],
        polygon_area=[
            Constraint(value=1000, operator=">"),
            Constraint(value=5000, operator="<"),
        ],
        dataset_metadata={
            "some_str": [Constraint(value="foobar", operator="==")],
            "some_float": [Constraint(value=0.123, operator=">=")],
        },
    )

    filter_from_constraints = _format_filter(
        [
            Dataset.name.in_(["a", "b", "c"]),
            Model.name.in_(["x", "y", "z"]),
            Label.score > 0.75,
            Annotation.polygon.area > 1000,
            Annotation.polygon.area < 5000,
            Dataset.metadata["some_str"] == "foobar",
            Dataset.metadata["some_float"] >= 0.123,
        ]
    )

    filter_from_dictionary = _format_filter(
        {
            "dataset_names": ["a", "b", "c"],
            "model_names": ["x", "y", "z"],
            "label_scores": [{"value": 0.75, "operator": ">"}],
            "polygon_area": [
                {"value": 1000, "operator": ">"},
                {"value": 5000, "operator": "<"},
            ],
            "dataset_metadata": {
                "some_str": [{"value": "foobar", "operator": "=="}],
                "some_float": [{"value": 0.123, "operator": ">="}],
            },
        }
    )

    assert filter_object == filter_from_constraints
    assert filter_object == filter_from_dictionary
