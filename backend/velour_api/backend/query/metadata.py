from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from velour_api import schemas
from velour_api.backend import core, models


def get_metadatum(
    db: Session,
    metadatum: models.MetaDatum = None,
    dataset: models.Dataset = None,
    model: models.Model = None,
    datum: models.Datum = None,
    annotation: models.Annotation = None,
    name: str = None,
) -> schemas.MetaDatum | None:

    if not metadatum:
        if dataset and name:
            metadatum = (
                db.query(models.MetaDatum)
                .where(
                    and_(
                        models.MetaDatum.dataset_id == dataset.id,
                        models.MetaDatum.name == name,
                    )
                )
                .one_or_none()
            )
        elif model and name:
            metadatum = (
                db.query(models.MetaDatum)
                .where(
                    and_(
                        models.MetaDatum.model_id == model.id,
                        models.MetaDatum.name == name,
                    )
                )
                .one_or_none()
            )
        elif datum and name:
            metadatum = (
                db.query(models.MetaDatum)
                .where(
                    and_(
                        models.MetaDatum.datum_id == datum.id,
                        models.MetaDatum.name == name,
                    )
                )
                .one_or_none()
            )
        elif annotation and name:
            metadatum = (
                db.query(models.MetaDatum)
                .where(
                    and_(
                        models.MetaDatum.annotation_id == annotation.id,
                        models.MetaDatum.name == name,
                    )
                )
                .one_or_none()
            )
        else:
            return None

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
    dataset: models.Dataset = None,
    model: models.Model = None,
    datum: models.Datum = None,
    annotation: models.Annotation = None,
    name: str = None,
    value_type: type = None,
) -> list[schemas.MetaDatum]:

    metadata = core.get_metadata(
        db,
        dataset=dataset,
        model=model,
        datum=datum,
        annotation=annotation,
        name=name,
        value_type=value_type,
    )

    return [get_metadatum(db, metadatum=metadatum) for metadatum in metadata]


# @TODO
def compare_metadata(
    metadatum: schemas.MetaDatum
) -> list:
    comparison = [models.MetaDatum.name == metadatum.name]
    if isinstance(metadatum.value, str):
        comparison.append(models.MetaDatum.value == metadatum.value)
    if isinstance(metadatum.value, float):
        comparison.append(models.MetaDatum.value == metadatum.value)
    if isinstance(metadatum.value, schemas.GeoJSON):
        raise NotImplementedError("Havent implemented GeoJSON support.")
    return comparison