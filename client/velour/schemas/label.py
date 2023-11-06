import math
from dataclasses import dataclass
from typing import Tuple, Union


@dataclass
class Label:
    key: str
    value: str
    score: Union[float, None] = None

    def __post_init__(self):
        if not isinstance(self.key, str):
            raise TypeError("key should be of type `str`")
        if not isinstance(self.value, str):
            raise TypeError("value should be of type `str`")
        if isinstance(self.score, int):
            self.score = float(self.score)
        if not isinstance(self.score, (float, type(None))):
            raise TypeError("score should be of type `float`")

    def tuple(self) -> Tuple[str, str, Union[float, None]]:
        return (self.key, self.value, self.score)

    def __eq__(self, other):
        if (
            not hasattr(other, "key")
            or not hasattr(other, "key")
            or not hasattr(other, "score")
        ):
            return False

        # if the scores aren't the same type return False
        if (other.score is None) != (self.score is None):
            return False

        scores_equal = (other.score is None and self.score is None) or (
            math.isclose(self.score, other.score)
        )

        return (
            scores_equal
            and self.key == other.key
            and self.value == other.value
        )

    def __hash__(self) -> int:
        return hash(f"key:{self.key},value:{self.value},score:{self.score}")
