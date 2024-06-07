from typing import Type

from valor_api.backend.models import (
    Annotation,
    Dataset,
    Datum,
    Embedding,
    GroundTruth,
    Label,
    Model,
    Prediction,
)

TableTypeAlias = (
    Type[Dataset]
    | Type[Model]
    | Type[Datum]
    | Type[Annotation]
    | Type[GroundTruth]
    | Type[Prediction]
    | Type[Label]
    | Type[Embedding]
)
LabelSourceAlias = Type[GroundTruth] | Type[Prediction] | Type[Annotation]
