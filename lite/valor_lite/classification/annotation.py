from dataclasses import dataclass


@dataclass
class Classification:
    uid: str
    groundtruth: str
    predictions: list[str]
    scores: list[float]

    def __post_init__(self):
        if not isinstance(self.groundtruth, str):
            raise ValueError(
                "A classification must contain a single groundtruth."
            )
        if len(self.predictions) != len(self.scores):
            raise ValueError("There must be a score per prediction label.")
