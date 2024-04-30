# """ These integration tests should be run with a back end at http://localhost:8000
# that is no auth
# """

# import pytest

# from valor import (
#     Annotation,
#     Client,
#     Dataset,
#     Datum,
#     GroundTruth,
#     Label,
#     Model,
#     Prediction,
# )
# from valor.enums import TaskType


# @pytest.fixture
# def gt_correct_class_ranking():
#     # input option #1: a list of strings denoting all of the potential options. when users pass in a list of strings, they won't be able to get back NDCG
#     return [
#         GroundTruth(
#             datum=Datum(uid="uid1", metadata={}),
#             annotations=[
#                 Annotation(
#                     task_type=TaskType.RANKING,
#                     labels=[
#                         Label(key="k1", value="gt"),
#                     ],
#                     ranking=[
#                         "relevant_doc1",
#                         "relevant_doc2",
#                         "relevant_doc3",
#                         "relevant_doc4",
#                     ],
#                 )
#             ],
#         ),
#         GroundTruth(
#             datum=Datum(uid="uid2", metadata={}),
#             annotations=[
#                 Annotation(
#                     task_type=TaskType.RANKING,
#                     labels=[
#                         Label(key="k2", value="gt"),
#                     ],
#                     ranking=[
#                         "1",
#                         "2",
#                         "3",
#                         "4",
#                     ],
#                 )
#             ],
#         ),
#     ]


# @pytest.fixture
# def pd_correct_class_ranking():
#     return [
#         Prediction(
#             datum=Datum(uid="uid1", metadata={}),
#             annotations=[
#                 Annotation(
#                     task_type=TaskType.RANKING,
#                     labels=[
#                         Label(key="k1", value="gt"),
#                     ],
#                     ranking=[
#                         "foo",
#                         "bar",
#                         "relevant_doc2",
#                     ],
#                 )
#             ],
#         ),
#         Prediction(
#             datum=Datum(uid="uid1", metadata={}),
#             annotations=[
#                 Annotation(
#                     task_type=TaskType.RANKING,
#                     labels=[
#                         Label(key="k1", value="gt"),
#                     ],
#                     ranking=[
#                         "bbq",
#                         "iguana",
#                     ],
#                 )
#             ],
#         ),
#         Prediction(
#             datum=Datum(uid="uid1", metadata={}),
#             annotations=[
#                 Annotation(
#                     task_type=TaskType.RANKING,
#                     labels=[
#                         Label(key="k1", value="gt"),
#                     ],
#                     ranking=[
#                         "foo",
#                         "relevant_doc4",
#                         "relevant_doc1",
#                     ],
#                 )
#             ],
#         ),
#         # Prediction(
#         #     datum=Datum(uid="uid1", metadata={}),
#         #     annotations=[
#         #         Annotation(
#         #             task_type=TaskType.RANKING,
#         #             labels=[
#         #                 Label(key="k1", value="gt"),
#         #             ],
#         #             ranking=[
#         #                 0.4,
#         #                 0.3,
#         #                 0.2,
#         #             ],  # error case: length of this prediction doesn't match ground truth we're comparing against
#         #         )
#         #     ],
#         # ),
#         # Prediction(
#         #     datum=Datum(uid="uid1", metadata={}),
#         #     annotations=[
#         #         Annotation(
#         #             task_type=TaskType.RANKING,
#         #             labels=[
#         #                 Label(key="k1", value="gt"),
#         #             ],
#         #             ranking=[
#         #                 0.4,
#         #                 0.3,
#         #                 0.2,
#         #                 0.2,
#         #             ],  # error case: weights sum to greater than one
#         #         )
#         #     ],
#         # ),
#         # Prediction(
#         #     datum=Datum(uid="uid1", metadata={}),
#         #     annotations=[
#         #         Annotation(
#         #             task_type=TaskType.RANKING,
#         #             labels=[
#         #                 Label(key="k1", value="gt"),
#         #             ],
#         #             ranking=[
#         #                 0.4,
#         #                 0.3,
#         #                 0.2,
#         #                 0.1,
#         #             ],  # ranking by relevance scores
#         #         )
#         #     ],
#         # ),
#         # Prediction(
#         #     datum=Datum(uid="uid1", metadata={}),
#         #     annotations=[
#         #         Annotation(
#         #             task_type=TaskType.RANKING,
#         #             labels=[
#         #                 Label(key="k2", value="gt"),
#         #             ],
#         #             ranking=[
#         #                 "a",
#         #                 "b",
#         #                 "c",
#         #                 "d",
#         #             ],  # perfect ranking
#         #         )
#         #     ],
#         # ),
#         # Prediction(
#         #     datum=Datum(uid="uid2", metadata={}),
#         #     annotations=[
#         #         Annotation(
#         #             task_type=TaskType.RANKING,
#         #             labels=[
#         #                 Label(key="k1", value="gt"),
#         #             ],
#         #             ranking=[
#         #                 "3",
#         #                 "2",
#         #                 "1",
#         #                 "4",
#         #             ],
#         #         )
#         #     ],
#         # ),
#     ]


# def test_evaluate_correct_class_ranking(
#     client: Client,
#     dataset_name: str,
#     model_name: str,
#     gt_correct_class_ranking: list,
#     pd_correct_class_ranking: list,
# ):
#     dataset = Dataset.create(dataset_name)
#     model = Model.create(model_name)

#     for gt in gt_correct_class_ranking:
#         dataset.add_groundtruth(gt)

#     dataset.finalize()

#     for pred in pd_correct_class_ranking:
#         model.add_prediction(dataset, pred)

#     model.finalize_inferences(dataset)

#     eval_job = model.evaluate_ranking(dataset)

#     assert eval_job.id

#     # assert eval_job.wait_for_completion(timeout=30) == EvaluationStatus.DONE

#     # metrics = eval_job.metrics

#     # assert metrics


# def test_evaluate_relevancy_score_ranking():
#     pass


# def test_evaluate_embedding_ranking():
#     pass


# def test_evaluate_mixed_rankings():
#     pass


# # TODO test that each key only has one groundtruth
