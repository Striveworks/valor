from dataclasses import dataclass


@dataclass
class Metric:
    type: str
    value: float | dict | list
    parameters: dict

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "value": self.value,
            "parameters": self.parameters,
        }
