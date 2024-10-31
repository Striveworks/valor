from dataclasses import asdict, dataclass


@dataclass
class BaseMetric:
    type: str
    value: int | float | dict
    parameters: dict

    def to_dict(self) -> dict:
        return asdict(self)
