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
)

write_path = "./results.json"
path = "./pr-curve-oom-data.json"
with open(path) as f:
    raw = json.load(f)

connect("http://0.0.0.0:8000")
client = Client()


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

    with open(write_path, "a+") as file:
        file.seek(0)
        data = json.load(file)

    data[current_datetime] = result_dict

    with open(write_path, "w+") as file:
        json.dump(data, file, indent=4)


def ingest_groundtruths_and_predictions(
    dset, model, raw: dict, pair_limit: int
):
    groundtruths = []
    predictions = []
    for groundtruth, prediction in raw["groundtruth_prediction_pairs"][
        :pair_limit
    ]:
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
                        ],
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
                        ],
                    )
                    for annotation in prediction["value"]["annotations"]
                ],
            )
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
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROCAUC",
            "PrecisionRecallCurve",
        ],
    )
    evaluation.wait_for_completion()
    return evaluation


def run_detailed_pr_curve_evaluation(dset, model):
    evaluation = model.evaluate_classification(
        dset,
        metrics_to_return=[
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROCAUC",
            "PrecisionRecallCurve",
            "DetailedPrecisionRecallCurve",
        ],
    )
    evaluation.wait_for_completion()
    return evaluation


def time_functions():

    for i, limit in enumerate([10, 100, 100, 500, 500]):

        dset = Dataset.create(name=f"bird-identification{i}")
        model = Model.create(name=f"some_model{i}")

        start_time = time()

        ingest_groundtruths_and_predictions(
            dset=dset, model=model, raw=raw, pair_limit=limit
        )
        ingest_time = f"{(time() - start_time):.4f}"

        run_base_evaluation(dset=dset, model=model)
        base_time = f"{(time() - start_time):.4f}"

        run_pr_curve_evaluation(dset=dset, model=model)
        pr_time = f"{(time() - start_time):.4f}"

        run_detailed_pr_curve_evaluation(dset=dset, model=model)
        detailed_pr_time = f"{(time() - start_time):.4f}"

        results = {
            "limit": limit,
            "ingest": ingest_time,
            "base": base_time,
            "pr": pr_time,
            "detailed pr": detailed_pr_time,
        }
        write_results_to_file(write_path=write_path, result_dict=results)
        print(results)


# %%
time_functions()

# %%
