from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from velour_api import schemas
from velour_api.backend import core, models, ops


def _get_metadatum(
    db: Session,
    metadatum: models.MetaDatum = None,
) -> schemas.MetaDatum | None:

    # Parsing
    if metadatum.string_value is not None:
        value = metadatum.string_value
    elif metadatum.numeric_value is not None:
        value = metadatum.numeric_value
    elif metadatum.geo is not None:
        # @TODO: Add geographic type
        raise NotImplemented
    else:
        return None

    return schemas.MetaDatum(
        name=metadatum.name,
        value=value,
    )


def get_metadata(
    db: Session,
    dataset_name: str = None,
    model_name: str = None,
    datum_uid: str = None,
    name: str = None,
) -> list[schemas.MetaDatum]:

    dataset = core.get_model(db, dataset_name) if dataset_name else None
    model = core.get_model(db, model_name) if model_name else None
    datum = core.get_datum(db, datum_uid) if datum_uid else None

    # create query filter
    qf = ops.QueryFilter()
    qf.filter_by_id(
        target=models.MetaDatum,
        sources=[
            dataset,
            model,
            datum,
        ]
    )
    qf.filter(models.MetaDatum.name == name)

    return [
        _get_metadatum(db, metadatum=metadatum) 
        for metadatum in core.get_metadata(db, qf)
    ]
