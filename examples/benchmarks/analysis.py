# %%
import json
from datetime import datetime
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

write_path = "./results.json"
path = "./pr-curve-oom-data.json"
with open(path) as f:
    raw = json.load(f)

# %%
connect("http://0.0.0.0:8000")
client = Client()
dset = Dataset.create(name="bird-identification")
model = Model.create(name="some_model")
PAIR_LIMIT = 10


# %%
def time_func(func):
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f"Function {func.__name__!r} executed in {(t2-t1):.4f}s")
        return result

    return wrap_func


def write_results_to_file(write_path: str, result_dict: dict):
    current_datetime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    with open(write_path, "a+") as f:
        try:
            result = json.load(f)
        except json.JSONDecodeError:
            result = {}
        result[current_datetime] = result_dict
        f.write(json.dumps(result))


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


def run_base_evaluation(dset, model):
    evaluation = model.evaluate_classification(dset)
    evaluation.wait_for_completion()
    return evaluation


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


def time_functions():

    start_time = time()

    ingest_groundtruths_and_predictions(raw=raw)
    ingest_time = f"{(time() - start_time):.4f}"

    run_base_evaluation(dset=dset, model=model)
    base_time = f"{(time() - start_time):.4f}"

    run_pr_curve_evaluation(dset=dset, model=model)
    pr_time = f"{(time() - start_time):.4f}"

    results = {
        "limit": PAIR_LIMIT,
        "ingest": ingest_time,
        "base": base_time,
        "pr": pr_time,
    }
    write_results_to_file(write_path=write_path, result_dict=results)
    return results


# %%
time_functions()

# %%
