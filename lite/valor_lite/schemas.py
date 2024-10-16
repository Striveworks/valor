from dataclasses import asdict, dataclass, is_dataclass


@dataclass
class Metric:
    type: str
    value: int | float | dict
    parameters: dict

    __match_args__ = ("type",)

    def __post_init__(self):
        if not isinstance(self.value, (int, float, dict)):
            raise TypeError(
                "Metric value must be of type `int`, `float` or `dict`."
            )

    def to_dict(self) -> dict:
        return asdict(self)


class _BaseMetric:
    """
    Base class subclassed by metric dataclasses.

    Automates conversion to a generic metric type.
    """

    def to_metric(self) -> Metric:
        """Converts the instance to a generic `Metric` object."""
        if not is_dataclass(self):
            raise TypeError(
                f"Type `{type(self)}` inherits `_BaseMeric` but is not a dataclass."
            )
        m_raw = asdict(self)

        m_type = type(self).__name__
        if "value" in m_raw:
            m_value = m_raw.pop("value")
            m_parameters = m_raw
        else:
            m_parameters = {
                key: m_raw.pop(key)
                for key in set(m_raw.keys())
                if key
                in {
                    "label",
                    "iou_threshold",
                    "iou_thresholds",
                    "score_threshold",
                    "score_thresholds",
                    "hardmax",
                    "maximum_number_of_examples",
                }
            }
            m_value = m_raw
        return Metric(
            type=m_type,
            value=m_value,
            parameters=m_parameters,
        )

    def to_dict(self) -> dict:
        """Converts the instance to a dictionary representation."""
        return self.to_metric().to_dict()
