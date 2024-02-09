from velour_api.backend.models import (
    Annotation,
    Dataset,
    Datum,
    GroundTruth,
    Label,
    Model,
    Prediction,
)

TableTypeAlias = (
    Dataset | Model | Datum | Annotation | GroundTruth | Prediction | Label
)
