from pydantic import BaseModel, ConfigDict

from valor_api.enums import TaskType


class DeprecatedFilter(BaseModel):
    """
    Deprecated Filter Schema.

    Used to retrieve old evaluations.

    Attributes
    ----------
    dataset_names: List[str], default=None
        A list of `Dataset` names to filter on.
    dataset_metadata: Dict[dict], default=None
        A dictionary of `Dataset` metadata to filter on.
    model_names: List[str], default=None
        A list of `Model` names to filter on.
    model_metadata: Dict[dict], default=None
        A dictionary of `Model` metadata to filter on.
    datum_metadata: Dict[dict], default=None
        A dictionary of `Datum` metadata to filter on.
    task_types: List[TaskType], default=None
        A list of task types to filter on.
    annotation_metadata: Dict[dict], default=None
        A dictionary of `Annotation` metadata to filter on.
    require_bounding_box : bool, optional
        A toggle for filtering by bounding boxes.
    bounding_box_area : bool, optional
        An optional constraint to filter by bounding box area.
    require_polygon : bool, optional
        A toggle for filtering by polygons.
    polygon_area : bool, optional
        An optional constraint to filter by polygon area.
    require_raster : bool, optional
        A toggle for filtering by rasters.
    raster_area : bool, optional
        An optional constraint to filter by raster area.
    labels: List[Dict[str, str]], default=None
        A dictionary of `Labels' to filter on.
    label_ids: List[int], default=None
        A list of `Label` IDs to filter on.
    label_keys: List[str] = None, default=None
        A list of `Label` keys to filter on.
    label_scores: List[ValueFilter], default=None
        A list of `ValueFilters` which are used to filter `Evaluations` according to the `Model`'s prediction scores.
    """

    # datasets
    dataset_names: list[str] | None = None
    dataset_metadata: dict | None = None

    # models
    model_names: list[str] | None = None
    model_metadata: dict | None = None

    # datums
    datum_uids: list[str] | None = None
    datum_metadata: dict | None = None

    # annotations
    task_types: list[TaskType] | None = None
    annotation_metadata: dict | None = None

    require_bounding_box: bool | None = None
    bounding_box_area: list[dict] | None = None
    require_polygon: bool | None = None
    polygon_area: list[dict] | None = None
    require_raster: bool | None = None
    raster_area: list[dict] | None = None

    # labels
    labels: list[dict[str, str]] | None = None
    label_ids: list[int] | None = None
    label_keys: list[str] | None = None

    # predictions
    label_scores: list[dict] | None = None

    # pydantic settings
    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=("protected_",),
    )
