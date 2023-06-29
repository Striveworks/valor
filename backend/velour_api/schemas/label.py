from pydantic import BaseModel


class Label(BaseModel):
    key: str
    value: str

    @classmethod
    def from_key_value_tuple(cls, kv_tuple: tuple[str, str]):
        return cls(key=kv_tuple[0], value=kv_tuple[1])

    def __eq__(self, other):
        if hasattr(other, "key") and hasattr(other, "value"):
            return self.key == other.key and self.value == other.value
        return False

    def __hash__(self) -> int:
        return hash(f"key:{self.key},value:{self.value}")


class ScoredLabel(BaseModel):
    label: Label
    score: float
