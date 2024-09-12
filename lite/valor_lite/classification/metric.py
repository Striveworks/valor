from dataclasses import dataclass
from enum import Enum


class MetricType(Enum):
    pass


@dataclass
class Metric:
    value: float
