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
    datasets: Optional[Union[dict, FunctionType]] = None
    models: Optional[Union[dict, FunctionType]] = None
    datums: Optional[Union[dict, FunctionType]] = None
    annotations: Optional[Union[dict, FunctionType]] = None
    groundtruths: Optional[Union[dict, FunctionType]] = None
    predictions: Optional[Union[dict, FunctionType]] = None
    labels: Optional[Union[dict, FunctionType]] = None
    embeddings: Optional[Union[dict, FunctionType]] = None

    def __post_init__(self):
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

    def to_dict(self) -> dict:
        return asdict(self)
