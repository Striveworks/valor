""" These integration tests should be run with a backend at http://localhost:8000
that is no auth
"""

# @TODO: Implement metadata querying + geojson
# def test_create_images_with_metadata(
#     client: Client, db: Session, metadata: list[Metadatum], rect1: BoundingBox
# ):
#     dataset = Dataset.create(client, dset_name)

#     md1, md2, md3 = metadata
#     img1 = ImageMetadata(uid="uid1", metadata=[md1], height=100, width=200).to_datum()
#     img2 = ImageMetadata(uid="uid2", metadata=[md2, md3], height=100, width=200).to_datum()

#     print(GroundTruth(
#             dataset=dset_name,
#             datum=img1.to_datum(),
#             annotations=[
#                 Annotation(
#                     task_type=TaskType.DETECTION,
#                     labels=[Label(key="k", value="v")],
#                     bounding_box=rect1,
#                 ),
#             ]
#         ))

#     dataset.add_groundtruth(
#         GroundTruth(
#             dataset=dset_name,
#             datum=img1.to_datum(),
#             annotations=[
#                 Annotation(
#                     task_type=TaskType.DETECTION,
#                     labels=[Label(key="k", value="v")],
#                     bounding_box=rect1,
#                 ),
#             ]
#         )
#     )
#     dataset.add_groundtruth(
#         GroundTruth(
#             dataset=dset_name,
#             datum=img2.to_datum(),
#             annotations=[
#                 Annotation(
#                     task_type=TaskType.CLASSIFICATION,
#                     labels=[Label(key="k", value="v")],
#                 )
#             ]
#         )
#     )

#     data = db.scalars(select(models.Datum)).all()
#     assert len(data) == 2
#     assert set(d.uid for d in data) == {"uid1", "uid2"}

#     metadata_links = data[0].datum_metadatum_links
#     assert len(metadata_links) == 1
#     metadatum = metadata_links[0].metadatum
#     assert metadata_links[0].metadatum.name == "metadatum name1"
#     assert json.loads(db.scalar(ST_AsGeoJSON(metadatum.geo))) == {
#         "type": "Point",
#         "coordinates": [-48.23456, 20.12345],
#     }
#     metadata_links = data[1].datum_metadatum_links
#     assert len(metadata_links) == 2
#     metadatum1 = metadata_links[0].metadatum
#     metadatum2 = metadata_links[1].metadatum
#     assert metadatum1.name == "metadatum name2"
#     assert metadatum1.string_value == "a string"
#     assert metadatum2.name == "metadatum name3"
#     assert metadatum2.numeric_value == 0.45
