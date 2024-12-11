from dataclasses import dataclass


@dataclass
class Classification:
    """
    Classification data structure containing a ground truth label and a list of predictions.

    Parameters
    ----------
    uid : str
        Unique identifier for the instance.
    groundtruth : str
        The true label for the instance.
    predictions : list of str
        List of predicted labels.
    scores : list of float
        Confidence scores corresponding to each predicted label.

    Examples
    --------
    >>> classification = Classification(
    ...     uid='123',
    ...     groundtruth='cat',
    ...     predictions=['cat', 'dog', 'bird'],
    ...     scores=[0.9, 0.05, 0.05]
    ... )
    """

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
