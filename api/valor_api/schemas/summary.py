from pydantic import BaseModel

from valor_api.schemas.types import Label, MetadataType


class DatasetSummary(BaseModel):
    name: str
    num_datums: int
    num_annotations: int
    num_bounding_boxes: int
    num_polygons: int
    num_rasters: int
    task_types: list[list[str]]
    labels: list[Label]
    datum_metadata: list[MetadataType]
    annotation_metadata: list[MetadataType]
