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
    Dataset
    | Model
    | Datum
    | Annotation
    | GroundTruth
    | Prediction
    | Label
    | Embedding
)
LabelSourceAlias = GroundTruth | Prediction | Annotation
