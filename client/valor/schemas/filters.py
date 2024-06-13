from dataclasses import dataclass
from typing import Optional

from valor.schemas.symbolic.operators import FunctionType


@dataclass
class Filter:
    datasets: Optional[FunctionType] = None
    models: Optional[FunctionType] = None
    datums: Optional[FunctionType] = None
    annotations: Optional[FunctionType] = None
    groundtruths: Optional[FunctionType] = None
    predictions: Optional[FunctionType] = None
    labels: Optional[FunctionType] = None
    embeddings: Optional[FunctionType] = None

    def to_dict(self) -> dict:
        return {
            "datasets": self.datasets.to_dict() if self.datasets else None,
            "models": self.models.to_dict() if self.models else None,
            "datums": self.datums.to_dict() if self.datums else None,
            "annotations": self.annotations.to_dict()
            if self.annotations
            else None,
            "groundtruths": self.groundtruths.to_dict()
            if self.groundtruths
            else None,
            "predictions": self.predictions.to_dict()
            if self.predictions
            else None,
            "labels": self.labels.to_dict() if self.labels else None,
            "embeddings": self.embeddings.to_dict()
            if self.embeddings
            else None,
        }
