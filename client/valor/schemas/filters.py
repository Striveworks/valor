from dataclasses import asdict, dataclass
from typing import Optional, Union

from valor.schemas.symbolic.operators import (
    And,
    Contains,
    Eq,
    FunctionType,
    Gt,
    Gte,
    Inside,
    Intersects,
    IsNotNull,
    IsNull,
    Lt,
    Lte,
    Ne,
    Not,
    Or,
    Outside,
)

FunctionTypeTuple = (
    And,
    Or,
    Not,
    IsNull,
    IsNotNull,
    Eq,
    Ne,
    Gt,
    Gte,
    Lt,
    Lte,
    Intersects,
    Inside,
    Outside,
    Contains,
)


@dataclass
class Filter:
    """
    A data class that encapsulates filter conditions for various Valor components.

    Attributes
    ----------
    datasets : dict | FunctionType, optional
        Filter conditions to apply to datasets.
    models : dict | FunctionType, optional
        Filter conditions to apply to models.
    datums : dict | FunctionType, optional
        Filter conditions to apply to datums.
    annotations : dict | FunctionType, optional
        Filter conditions to apply to annotations.
    groundtruths : dict | FunctionType, optional
        Filter conditions to apply to groundtruths.
    predictions : dict | FunctionType, optional
        Filter conditions to apply to predictions.
    labels : dict | FunctionType, optional
        Filter conditions to apply to labels.
    embeddings : dict | FunctionType, optional
        Filter conditions to apply to embeddings.

    Examples
    --------
    Filter annotations by area and label.
    >>> Filter(
    ...     annotations=And(
    ...         Label.key == "name",
    ...         Annotation.raster.area > upper_bound,
    ...     )
    ... )

    Filter datums by annotations and labels.
    >>> Filter(
    ...     datums=And(
    ...         Label.key == "name",
    ...         Annotation.raster.area > upper_bound,
    ...     )
    ... )
    """

    datasets: Optional[Union[dict, FunctionType]] = None
    models: Optional[Union[dict, FunctionType]] = None
    datums: Optional[Union[dict, FunctionType]] = None
    annotations: Optional[Union[dict, FunctionType]] = None
    groundtruths: Optional[Union[dict, FunctionType]] = None
    predictions: Optional[Union[dict, FunctionType]] = None
    labels: Optional[Union[dict, FunctionType]] = None
    embeddings: Optional[Union[dict, FunctionType]] = None

    def to_dict(self) -> dict:
        if isinstance(self.datasets, FunctionTypeTuple):
            self.datasets = self.datasets.to_dict()
        if isinstance(self.models, FunctionTypeTuple):
            self.models = self.models.to_dict()
        if isinstance(self.datums, FunctionTypeTuple):
            self.datums = self.datums.to_dict()
        if isinstance(self.annotations, FunctionTypeTuple):
            self.annotations = self.annotations.to_dict()
        if isinstance(self.groundtruths, FunctionTypeTuple):
            self.groundtruths = self.groundtruths.to_dict()
        if isinstance(self.predictions, FunctionTypeTuple):
            self.predictions = self.predictions.to_dict()
        if isinstance(self.labels, FunctionTypeTuple):
            self.labels = self.labels.to_dict()
        if isinstance(self.embeddings, FunctionTypeTuple):
            self.embeddings = self.embeddings.to_dict()
        return asdict(self)
