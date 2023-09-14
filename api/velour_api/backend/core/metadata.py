from velour_api import schemas


def deserialize_metadatums(
    metadata: list[schemas.Metadatum],
) -> dict[str, any]:
    def unpack(metadatum: schemas.Metadatum):
        if isinstance(metadatum.value, schemas.DateTime):
            raise NotImplementedError(
                "Have not implemented DateTime metadatums"
            )
        return metadatum.value

    return {
        metadatum.key: unpack(metadatum)
        for metadatum in metadata
        if metadatum.value is not None
    }


def serialize_metadatums(metadata: dict[str, any]) -> list[schemas.Metadatum]:
    return [
        schemas.Metadatum(key=key, value=metadata[key]) for key in metadata
    ]
