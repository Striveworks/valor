# def _test_extreme_generate_query(target: graph.Node):
#     filters = {
#         graph.dataset: [models.Dataset.name == "dataset1"],
#         graph.model: [models.Model.name == "model1"],
#         graph.groundtruth_label: [models.Label.value == "dog"],
#         graph.prediction_label: [models.Label.value == "cat"],
#     }
#     generated_query = graph.generate_query(target, filters)

#     match target:
#         case graph.model:
#             expected_query = (
#                 select(models.Model.id)
#                 .join(models.Annotation, models.Annotation.model_id == models.Model.id)
#                 .join(models.Datum, models.Datum.id == models.Annotation.datum_id)
#                 .join(models.Prediction, models.Prediction.annotation_id == models.Annotation.id)
#                 .join(models.Label, models.Label.id == models.Prediction.label_id)
#                 .join(models.Dataset, models.Dataset.id == models.Datum.dataset_id)
#                 .where(
#                     models.Dataset.name == "dataset1",
#                     models.Label.value == "cat",
#                     models.Model.name == "model1",
#                     models.Datum.id.in_(
#                         select(models.Datum.id)
#                         .join(models.Annotation, models.Annotation.datum_id == models.Datum.id)
#                         .join(models.GroundTruth, models.GroundTruth.annotation_id == models.Annotation.id)
#                         .join(models.Label, models.Label.id == models.GroundTruth.label_id)
#                         .where(models.Label.value == "dog")
#                     )
#                 )
#             )
#         case graph.dataset:
#             expected_query = (
#                 select(models.Dataset.id)
#                 .join(models.Datum, models.Datum.dataset_id == models.Dataset.id)
#                 .join(models.Annotation, models.Annotation.datum_id == models.Datum.id)
#                 .join(models.Model, models.Model.id == models.Annotation.model_id)
#                 .join(models.Prediction, models.Prediction.annotation_id == models.Annotation.id)
#                 .join(models.Label, models.Label.id == models.Prediction.label_id)
#                 .where(
#                     models.Model.name == "model1",
#                     models.Label.value == "cat",
#                     models.Dataset.name == "dataset1",
#                     models.Datum.id.in_(
#                         select(models.Datum.id)
#                         .join(models.Annotation, models.Annotation.datum_id == models.Datum.id)
#                         .join(models.GroundTruth, models.GroundTruth.annotation_id == models.Annotation.id)
#                         .join(models.Label, models.Label.id == models.GroundTruth.label_id)
#                         .where(models.Label.value == "dog")
#                     )
#                 )
#             )

#     return generated_query, expected_query


# # WHERE dataset.name = :name_1 AND label.value = :value_1 AND model.name = :name_2 AND datum.id IN (
# #     SELECT anon_1.id
# #     FROM (
# #         SELECT datum.id AS id
# #         FROM datum
# #         JOIN annotation ON annotation.datum_id = datum.id
# #         JOIN groundtruth ON groundtruth.annotation_id = annotation.id
# #         JOIN label ON label.id = groundtruth.label_id
# #         WHERE label.value = :value_2
# #     )
# # AS anon_1)


# def test_generate_query():
#     # def generate_query(target: Node, filters: dict[Node, list[BinaryExpression]]):


#     # Q: Get model ids for all models that operate over a dataset with name "dataset1"

#     target = graph.model
#     filters = {graph.dataset: [models.Dataset.name == "dataset1"]}
#     generated_query = graph.generate_query(target, filters)

#     expected_query = (
#         select(models.Model.id)
#         .join(models.Annotation, models.Annotation.model_id == models.Model.id)
#         .join(models.Datum, models.Datum.id == models.Annotation.datum_id)
#         .join(models.Dataset, models.Dataset.id == models.Datum.dataset_id)
#         .where(models.Dataset.name == "dataset1")
#     )

#     assert str(generated_query) == str(expected_query)

#     # extreme request (models)
#     generated_query, expected_query = _test_extreme_generate_query(graph.model)
#     assert str(generated_query.compile()) == str(expected_query.compile())

#     # extreme request (datasets)
#     generated_query, expected_query = _test_extreme_generate_query(graph.dataset)
#     assert str(generated_query) == str(expected_query)

#     print(generated_query)
