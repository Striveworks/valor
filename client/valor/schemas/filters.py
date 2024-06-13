from dataclasses import asdict, dataclass
from typing import Optional, Union

from valor.schemas.symbolic.operators import FunctionType


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
        if isinstance(self.datasets, FunctionType):
            self.datasets = self.datasets.to_dict()
        if isinstance(self.models, FunctionType):
            self.models = self.models.to_dict()
        if isinstance(self.datums, FunctionType):
            self.datums = self.datums.to_dict()
        if isinstance(self.annotations, FunctionType):
            self.annotations = self.annotations.to_dict()
        if isinstance(self.groundtruths, FunctionType):
            self.groundtruths = self.groundtruths.to_dict()
        if isinstance(self.predictions, FunctionType):
            self.predictions = self.predictions.to_dict()
        if isinstance(self.labels, FunctionType):
            self.labels = self.labels.to_dict()
        if isinstance(self.embeddings, FunctionType):
            self.embeddings = self.embeddings.to_dict()

    def to_dict(self) -> dict:
        return asdict(self)
