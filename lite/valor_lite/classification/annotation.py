from dataclasses import dataclass


@dataclass
class Classification:
    uid: str
    groundtruths: list[str]
    predictions: list[str]
    scores: list[float]

    def __post_init__(self):
        if len(self.predictions) != len(self.scores):
            raise ValueError("There must be a score per prediction label.")
