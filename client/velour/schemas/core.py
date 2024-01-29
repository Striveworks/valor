import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

import numpy as np

from velour.schemas.constraints import (
    NumericMapper,
    StringMapper,
    _DeclarativeMapper,
)


@dataclass
class Label:
    """
    An object for labeling datasets, models, and annotations.

    Attributes
    ----------
    key : str
        The class key of the label.
    value : str
        The class value of the label.
    score : float, optional
        The score associated with the label (if applicable).
    """

    value: str
    key: Union[str, StringMapper] = field(default=StringMapper("label_keys"))
    score: Union[float, np.floating, NumericMapper, None] = field(
        default=NumericMapper("prediction_scores")
    )

    def __post_init__(self):

        if isinstance(self.key, _DeclarativeMapper):
            raise TypeError(
                "`velour.Label` instances require the `key` attribute be of type `str`."
            )
        if isinstance(self.value, _DeclarativeMapper):
            raise TypeError(
                "`velour.Label` instances require the `value` attribute be of type `str`."
            )
        if isinstance(self.score, _DeclarativeMapper):
            self.score = None

        if not isinstance(self.key, str):
            raise TypeError("Attribute `key` should have type `str`.")
        if not isinstance(self.value, str):
            raise TypeError("Attribute `value` should have type `str`.")
        if self.score is not None:
            try:
                self.score = float(self.score)
            except ValueError:
                raise TypeError("score should be convertible to `float`")

    def __str__(self):
        return str(self.tuple())

    def __eq__(self, other):
        """
        Defines how `Labels` are compared to one another

        Parameters
        ----------
        other : Label
            The object to compare with the `Label`.

        Returns
        ----------
        boolean
            A boolean describing whether the two objects are equal.
        """
        if type(other) is not type(self):
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
        """
        Defines how a `Label` is hashed.

        Returns
        ----------
        int
            The hashed 'Label`.
        """
        return hash(f"key:{self.key},value:{self.value},score:{self.score}")

    def tuple(self) -> Tuple[str, str, Optional[float]]:
        """
        Defines how the `Label` is turned into a tuple.

        Returns
        ----------
        tuple
            A tuple of the `Label's` arguments.
        """
        return (self.key, self.value, self.score)
