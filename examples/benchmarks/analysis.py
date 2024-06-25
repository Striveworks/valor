# %%
import json
from time import time

from valor import (
    Annotation,
    Client,
    Dataset,
    Datum,
    GroundTruth,
    Label,
    Model,
    Prediction,
    connect,
    enums,
)

path = "./pr-curve-oom-data.json"
with open(path) as f:
    raw = json.load(f)

# %%
connect("http://0.0.0.0:8000")
client = Client()
dset = Dataset.create(name="bird-identification")
model = Model.create(name="some_model")


# %%
def time_func(func):
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f"Function {func.__name__!r} executed in {(t2-t1):.4f}s")
        return result

    return wrap_func


@time_func
def ingest_groundtruths_and_predictions(raw: dict):
    number_of_pairs = len(raw["groundtruth_prediction_pairs"])
    groundtruths = []
    predictions = []
    for groundtruth, prediction in raw["groundtruth_prediction_pairs"]:
        groundtruths.append(
            GroundTruth(
                datum=Datum(
                    uid=groundtruth["value"]["datum"]["uid"],
                    metadata={"width": 224, "height": 224},
                ),
                annotations=[
                    Annotation(
                        labels=[
                            Label(
                                key=label["key"],
                                value=label["value"],
                                score=label["score"],
                            )
                            for label in annotation["labels"]
                        ]
                    )
                    for annotation in groundtruth["value"]["annotations"]
                ],
            )
        )

        predictions.append(
            Prediction(
                datum=Datum(
                    uid=prediction["value"]["datum"]["uid"],
                    metadata={"width": 224, "height": 224},
                ),
                annotations=[
                    Annotation(
                        labels=[
                            Label(
                                key=label["key"],
                                value=label["value"],
                                score=label["score"],
                            )
                            for label in annotation["labels"]
                        ]
                    )
                    for annotation in prediction["value"]["annotations"]
                ],
            )
        )

    assert (
        len(predictions) == number_of_pairs
        and len(groundtruths) == number_of_pairs
    )

    for gt in groundtruths:
        dset.add_groundtruth(gt)

    for pred in predictions:
        model.add_prediction(dset, pred)

    dset.finalize()
    model.finalize_inferences(dataset=dset)


@time_func
def run_base_evaluation(dset, model):
    evaluation = model.evaluate_classification(dset)
    evaluation.wait_for_completion()
    return evaluation


@time_func
def run_pr_curve_evaluation(dset, model):
    evaluation = model.evaluate_classification(
        dset,
        metrics_to_return=[
            enums.MetricType.Precision,
            enums.MetricType.Recall,
            enums.MetricType.F1,
            enums.MetricType.Accuracy,
            enums.MetricType.ROCAUC,
            enums.MetricType.PrecisionRecallCurve,
            enums.MetricType.DetailedPrecisionRecallCurve,
        ],
    )
    evaluation.wait_for_completion()
    return evaluation


# %%

ingest_groundtruths_and_predictions(raw=raw)

# %%
run_base_evaluation(dset=dset, model=model)


# %%
run_pr_curve_evaluation(dset=dset, model=model)

# %%
raw["class_labels"]
