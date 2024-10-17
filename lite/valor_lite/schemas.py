from dataclasses import asdict, dataclass


@dataclass
class BaseMetric:
    type: str
    value: int | float | dict
    parameters: dict

    def __post_init__(self):
        if not isinstance(self.value, (int, float, dict)):
            raise TypeError(
                "Metric value must be of type `int`, `float` or `dict`."
            )

    def to_dict(self) -> dict:
        return asdict(self)
