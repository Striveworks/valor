import math

from pydantic import BaseModel


class Label(BaseModel):
    key: str
    value: str
    score: float | None = None

    @classmethod
    def from_key_value_tuple(cls, kv_tuple: tuple[str, str]):
        if not isinstance(kv_tuple, tuple):
            raise ValueError("Should be tuple of 2 elements.")
        if not len(kv_tuple) == 2:
            raise ValueError("Should be tuple of 2 elements.")
        return cls(key=str(kv_tuple[0]), value=str(kv_tuple[1]))

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
