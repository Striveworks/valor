# A script to generate profiling reports for Valor
# See README.md for setup instructions

import cProfile
import functools
import io
import os
import pickle
import pstats
import random
import timeit
import tracemalloc
import warnings
from pstats import SortKey
from typing import List, Tuple, cast

import memory_profiler
import numpy as np
import pandas as pd
import seaborn as sns
import yappi
from fastapi import HTTPException
from numpy.typing import NDArray
from sqlalchemy.orm import Session
from tqdm import tqdm

from valor import (
    Annotation,
    Client,
    Dataset,
    Datum,
    Evaluation,
    GroundTruth,
    Label,
    Model,
    Prediction,
)
from valor.enums import EvaluationStatus, TaskType
from valor.metatypes import ImageMetadata
from valor.schemas import BoundingBox, Raster


def _sample_without_replacement(array: list, n: int) -> list:
    """Sample from a list without replacement. Used to draw unique IDs from a pre-populated list"""
    random.shuffle(array)
    output = array[:n]
    del array[:n]
    return output


def _generate_mask(
    height: int,
    width: int,
    minimum_mask_percent: float = 0.05,
    maximum_mask_percent: float = 0.4,
) -> NDArray:
    """Generate a random mask for an image with a given height and width"""
    mask_cutoff = random.uniform(minimum_mask_percent, maximum_mask_percent)
    mask = (np.random.random((height, width))) < mask_cutoff

    return mask


def _generate_gt_annotation(
    height: int,
    width: int,
    unique_label_ids: list,
    n_labels: int,
) -> Annotation:
    """Generate an annotation for a given image with a given number of labels"""
    task_types = [
        TaskType.OBJECT_DETECTION,
        TaskType.SEMANTIC_SEGMENTATION,
    ]
    mask = _generate_mask(height=height, width=width)
    raster = Raster.from_numpy(mask)
    bounding_box = _generate_bounding_box(
        max_height=height, max_width=width, is_random=True
    )
    task_type = random.choice(task_types)

    labels = []
    for i in range(n_labels):
        unique_id = _sample_without_replacement(unique_label_ids, 1)[0]
        label = _generate_label(str(unique_id))
        labels.append(label)

    return Annotation(
        task_type=task_type,
        labels=labels,
        raster=raster,
        bounding_box=bounding_box
        if task_type == TaskType.OBJECT_DETECTION
        else None,
    )


def _generate_label(unique_id: str, add_score: bool = False) -> Label:
    """Generate a label given some unique ID"""
    if not add_score:
        return Label(key="k" + unique_id, value="v" + unique_id)
    else:
        return Label(
            key="k" + unique_id,
            value="v" + unique_id,
            score=random.uniform(0, 1),
        )


def _generate_image_metadata(
    unique_id: str,
    min_height: int = 360,
    max_height: int = 640,
    min_width: int = 360,
    max_width: int = 640,
) -> dict:
    """Generate metadata for an image"""
    height = random.randrange(min_height, max_height)
    width = random.randrange(min_width, max_width)

    return {
        "uid": unique_id,
        "height": height,
        "width": width,
    }


def _generate_ground_truth(
    unique_image_id: str,
    n_annotations: int,
    n_labels: int,
) -> GroundTruth:
    """Generate a GroundTruth for an image with the given number of annotations and labels"""

    image_metadata = _generate_image_metadata(unique_id=unique_image_id)
    image_datum = ImageMetadata(
        uid=image_metadata["uid"],
        height=image_metadata["height"],
        width=image_metadata["width"],
    ).to_datum()

    unique_label_ids = list(range(n_annotations * n_labels))

    annotations = [
        _generate_gt_annotation(
            height=image_metadata["height"],
            width=image_metadata["width"],
            n_labels=n_labels,
            unique_label_ids=unique_label_ids,
        )
        for _ in range(n_annotations)
    ]

    gt = GroundTruth(
        datum=image_datum,
        annotations=annotations,
    )

    return gt


def _generate_bounding_box(
    max_height: int, max_width: int, is_random: bool = False
):
    """Generate an arbitrary bounding box"""

    if is_random:
        x_min = int(random.uniform(0, max_width // 2))
        x_max = int(random.uniform(max_width // 2, max_width))
        y_min = int(random.uniform(0, max_height // 2))
        y_max = int(random.uniform(max_height // 2, max_height))
    else:
        # use the whole image as the bounding box to ensure that we have predictions overlap with groundtruths
        x_min = 0
        x_max = max_width
        y_min = 0
        y_max = max_height

    return BoundingBox.from_extrema(
        xmin=x_min, ymin=y_min, xmax=x_max, ymax=y_max
    )


def _generate_prediction_annotation(
    height: int, width: int, unique_label_ids: list, n_labels: int
):
    """Generate an arbitrary inference annotation"""
    box = _generate_bounding_box(max_height=height, max_width=width)
    labels = []
    for i in range(n_labels):
        unique_id = _sample_without_replacement(unique_label_ids, 1)[0]
        label = _generate_label(str(unique_id), add_score=True)
        labels.append(label)

    return Annotation(
        task_type=TaskType.OBJECT_DETECTION,
        labels=labels,
        bounding_box=box,
    )


def _generate_prediction(
    datum: Datum,
    height: int,
    width: int,
    n_annotations: int,
    n_labels: int,
):
    """Generate an arbitrary prediction based on some image"""

    # ensure that some labels are common
    n_label_ids = n_annotations * n_labels
    unique_label_ids = list(range(n_label_ids))

    annotations = [
        _generate_prediction_annotation(
            height=height,
            width=width,
            unique_label_ids=unique_label_ids,
            n_labels=n_labels,
        )
        for _ in range(n_annotations)
    ]

    return Prediction(datum=datum, annotations=annotations)


def generate_segmentation_data(
    client: Client,
    dataset_name: str,
    n_images: int = 10,
    n_annotations: int = 10,
    n_labels: int = 2,
) -> Dataset:
    """
    Generate a synthetic Valor dataset given a set of input images.

    Parameters
    ----------
    client : Session
        The Client object used to access your valor instance.
    dataset_name : str
        The name of the dataset you want to generate in Valor.
    n_images : int
        The number of images you'd like your dataset to contain.
    n_annotations : int
        The number of annotations per image you'd like your dataset to contain.
    n_labels : int
        The number of labels per annotation you'd like your dataset to contain.
    """
    dataset = Dataset.create(dataset_name)

    unique_image_ids = list(range(n_images))
    for _ in tqdm(range(n_images)):
        gt = _generate_ground_truth(
            unique_image_id=str(
                _sample_without_replacement(unique_image_ids, 1)[0]
            ),
            n_annotations=n_annotations,
            n_labels=n_labels,
        )
        dataset.add_groundtruth(gt)

    dataset.finalize()

    return dataset


def generate_prediction_data(
    client: Client,
    dataset: Dataset,
    model_name: str,
    n_annotations: int = 10,
    n_labels: int = 2,
):
    """
    Generate an arbitrary number of predictions for a previously generated dataset.

    Parameters
    ----------
    client : Session
        The Client object used to access your Valor instance.
    dataset : Dataset
        The dataset object to create predictions for.
    model_name : str
        The name of your model.
    n_annotations : int
        The number of annotations per prediction you'd like your dataset to contain.
    n_labels : int
        The number of labels per annotation you'd like your dataset to contain.
    """
    model = Model.create(model_name)

    datums = dataset.get_datums()

    for datum in datums:
        height = cast(int, datum.metadata["height"])
        width = cast(int, datum.metadata["width"])
        prediction = _generate_prediction(
            datum=datum,
            height=int(height),
            width=int(width),
            n_annotations=n_annotations,
            n_labels=n_labels,
        )
        model.add_prediction(dataset, prediction)

    model.finalize_inferences(dataset)
    return model


""" FastAPI Profiling Decorators """


def generate_tracemalloc_profile(filepath: str) -> any:
    """
    A decorator for generating a tracemalloc Snapshot and peak/size pkl at a given filepath

    Parameters
    ----------
    filepath
        The path where you want to store the yappi output (e.g., 'utils/profiles/foo.out')
    func
        The function you want to profile
    db
        The sqlalchemy session used by your back end
    args
        Positional args to pass to your function
    kwargs
        Keyword args to pass to your function
    """

    def decorator(func: callable):
        @functools.wraps(func)
        def wrap_func(*args, db: Session, **kwargs):
            try:
                tracemalloc.start()
                first_size, first_peak = tracemalloc.get_traced_memory()
                tracemalloc.reset_peak()

                result = func(*args, db=db, **kwargs)

                # save size and peak
                second_size, second_peak = tracemalloc.get_traced_memory()
                output = {
                    "first_size": first_size,
                    "second_size": second_size,
                    "first_peak": first_peak,
                    "second_peak": second_peak,
                }
                with open(
                    f"{filepath}.pkl",
                    "wb+",
                ) as f:
                    pickle.dump(output, f)

                # save snapshot
                snapshot = tracemalloc.take_snapshot()
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                snapshot.dump(filepath)
            except HTTPException as e:
                raise e
            return result

        return wrap_func

    return decorator


# NOTE: I don't recommend using the memory_profiler module since it doesn't correctly write to filepath. See README.md for alternative instructions
def generate_memory_profile(filepath: str) -> any:
    """
    A decorator for generating a memory_profiler report at a given filepath

    Parameters
    ----------
    filepath
        The path where you want to store the yappi output (e.g., 'utils/profiles/foo.out')
    func
        The function you want to profile
    db
        The sqlalchemy session used by your back end
    args
        Positional args to pass to your function
    kwargs
        Keyword args to pass to your function
    """

    def decorator(func: callable):
        @functools.wraps(func)
        def wrap_func(*args, db: Session, **kwargs):
            try:
                with open(
                    filepath,
                    "wb+",
                ) as f:
                    result = memory_profiler.profile(
                        func=func(*args, db=db, **kwargs), stream=f
                    )
            except HTTPException as e:
                raise e
            return result

        return wrap_func

    return decorator


def generate_yappi_profile(filepath: str) -> any:
    """
    A decorator for generating a yappi report at a given filepath. yappi is generally preferred over cprofile for multi-threaded applications.

    Parameters
    ----------
    filepath
        The path where you want to store the yappi output (e.g., 'utils/profiles/foo.out')
    func
        The function you want to profile
    db
        The sqlalchemy session used by your back end
    args
        Positional args to pass to your function
    kwargs
        Keyword args to pass to your function
    """

    def decorator(func: callable):
        @functools.wraps(func)
        def wrap_func(*args, db: Session, **kwargs):
            try:
                yappi.set_clock_type("wall")

                with yappi.run():
                    result = func(*args, db=db, **kwargs)

                func_stats = yappi.get_func_stats()
                func_stats.save(filepath)
                yappi.stop()
            except HTTPException as e:
                raise e
            return result

        return wrap_func

    return decorator


def generate_cprofile(filepath: str) -> any:
    """
    A decorator for generating a cprofile report at a given filepath. cprofile is the go-to profiler for most single-threaded applications

    Parameters
    ----------
    filepath
        The path where you want to store the yappi output (e.g., 'utils/profiles/foo.out')
    func
        The function you want to profile
    db
        The sqlalchemy session used by your back end
    args
        Positional args to pass to your function
    kwargs
        Keyword args to pass to your function
    """

    def decorator(func: callable):
        @functools.wraps(func)
        def wrap_func(*args, db: Session, **kwargs):
            try:
                # Creating profile object
                profiler = cProfile.Profile()
                profiler.enable()

                result = func(*args, db=db, **kwargs)

                profiler.disable()
                sec = io.StringIO()
                sortby = SortKey.CUMULATIVE
                ps = pstats.Stats(profiler, stream=sec).sort_stats(sortby)
                ps.dump_stats(filepath)

            except HTTPException as e:
                raise e
            return result

        return wrap_func

    return decorator


""" Client-Side Profilers """


def _profile_tracemalloc(
    fn: callable, output: dict, top_n_traces: int = 10, **kwargs
) -> None:
    """Use tracemalloc to identify the top 10 memory traces for the current directory (sorted by size)"""
    tracemalloc.start()
    intermediary_outout = fn(**kwargs)
    snapshot = (
        tracemalloc.take_snapshot()
        .filter_traces(
            (
                tracemalloc.Filter(True, f"{os.getcwd()}/*", all_frames=True),
                tracemalloc.Filter(False, "*/profiling.py"),
            )
        )
        .statistics("lineno")
    )
    tracemalloc_output = dict()
    for i, stat in enumerate(snapshot[:top_n_traces]):
        tracemalloc_output.update(
            {
                i: {
                    "filename": stat.traceback._frames[0][0],
                    "line": stat.traceback._frames[0][1],
                    "size": stat.size,
                    "count": stat.count,
                }
            }
        )

    output.update(tracemalloc_output | intermediary_outout)


def _profile_func(
    fn: callable,
    top_n_traces: int = 10,
    **kwargs,
) -> dict:
    """Wrapper to profile a specific function using timeit and tracemalloc"""
    print(f"Profiling {fn.__name__} with args {kwargs}")

    success = False
    exception = " "
    output = {}
    start = timeit.default_timer()

    try:
        _profile_tracemalloc(
            fn=fn, output=output, top_n_traces=top_n_traces, **kwargs
        )
        success = True
    except Exception as e:
        exception = str(e)

    stop = timeit.default_timer()
    timeit_output = {
        "start": start,
        "total_runtime_seconds": round(stop - start, 2),
    }
    if success:
        print(f"Succeeded in {stop-start} seconds")
    else:
        print(f"Failed in {stop-start} seconds with error {exception}")

    return kwargs | timeit_output | output | {"exception": exception}


def _get_docker_cpu_memory_stats() -> pd.DataFrame:
    tsv = os.popen(
        'docker stats --no-stream --format "table {{.Container}}     {{.CPUPerc}}     {{.MemPerc}}"'
    ).read()
    string_tsv = io.StringIO(tsv)
    return pd.read_csv(
        string_tsv, sep="    ", header=0, names=["id", "cpu_util", "mem_util"]
    )


def _get_docker_disk_stats() -> pd.DataFrame:
    tsv = os.popen(
        'docker ps --size  --format "table {{.ID}}    {{.Image}}    {{.Size}}"'
    ).read()
    string_tsv = io.StringIO(tsv)
    return pd.read_csv(
        string_tsv, sep="    ", header=0, names=["id", "image", "disk_space"]
    )


def _get_docker_pids() -> pd.DataFrame:
    string = ""
    for _ in range(4):
        string += os.popen(
            "for i in $(docker container ls --format '{{.ID}}'); do docker inspect -f '{{.State.Pid}}    {{.Name}}' $i; done"
        ).read()

    string_tsv = io.StringIO(string)
    df = pd.read_csv(
        string_tsv, sep="    ", header=0, names=["pid", "name"]
    ).drop_duplicates()
    df["name"] = df["name"].str[1:]
    return df


def _generate_docker_snapshot() -> dict:
    """
    Takes a snapshot of all running Docker containers, returning a list of nested dictionaries containing the memory utilization, CPU utilization, and disk usage
    """
    mem_stats = _get_docker_cpu_memory_stats()
    disk_stats = _get_docker_disk_stats()

    mem_stats["id"] = mem_stats["id"].astype(str)
    disk_stats["id"] = disk_stats["id"].astype(str)

    snapshot = pd.merge(disk_stats, mem_stats, on="id")

    records = snapshot.to_dict("records")
    output = {}
    for record in records:
        record_output = {}
        image = record["image"]
        for field in ["disk_space", "cpu_util", "mem_util"]:
            record_output.update({f"{image}_{field}": record[field]})
        output.update(record_output)

    return output


""" Valor-Specific Functions """


def _setup_dataset(
    client: Client,
    dataset_name: str,
    n_images: int,
    n_annotations: int,
    n_labels: int,
) -> Dataset:
    """Generate a valor dataset with a given number of images, annotations, and labels"""
    assert (
        min(n_images, n_annotations, n_labels) > 0
    ), "You must generate at least one image, annotation, and label"
    dataset = generate_segmentation_data(
        client=client,
        dataset_name=dataset_name,
        n_images=n_images,
        n_annotations=n_annotations,
        n_labels=n_labels,
    )

    dataset.finalize()

    return dataset


def _get_evaluation_metrics(
    client: Client,
    dataset: Dataset,
    model_name: str,
    n_predictions: int,
    n_annotations: int,
    n_labels: int,
) -> Tuple[Model, Evaluation]:
    """Create arbitrary evaluation metrics based on some dataset"""

    model = generate_prediction_data(
        client=client,
        dataset=dataset,
        model_name=model_name,
        n_annotations=n_annotations,
        n_labels=n_labels,
    )

    eval_job = model.evaluate_detection(
        datasets=dataset,
        iou_thresholds_to_compute=[0.1, 1],
        iou_thresholds_to_return=[0.1, 1],
        filter_by=[Label.key == "k1"],
    )
    if eval_job.wait_for_completion(timeout=30) != EvaluationStatus.DONE:
        raise TimeoutError

    return (model, eval_job)


def _run_valor_profiling_functions(
    client: Client,
    dataset_name: str,
    model_name: str,
    n_images: int,
    n_predictions: int,
    n_annotations: int,
    n_labels: int,
) -> dict:
    """Call the various functions that we want to use to profile valor. Returns a dict of intermediary times that we want to include in our output"""

    start = timeit.default_timer()

    timeit_output = {}
    dataset = _setup_dataset(
        client=client,
        dataset_name=dataset_name,
        n_images=n_images,
        n_annotations=n_annotations,
        n_labels=n_labels,
    )

    timeit_output.update(
        {"setup_runtime_seconds": round(timeit.default_timer() - start, 2)}
    )

    start = timeit.default_timer()

    model, eval_job = _get_evaluation_metrics(
        client=client,
        dataset=dataset,
        model_name=model_name,
        n_predictions=n_predictions,
        n_annotations=n_annotations,
        n_labels=n_labels,
    )

    timeit_output.update(
        {
            "evaluation_runtime_seconds": round(
                timeit.default_timer() - start, 2
            )
        }
    )

    dataset.delete(timeout=30)

    return timeit_output


def profile_valor(
    client: Client,
    dataset_name: str,
    n_image_grid: List[int],
    n_predictions_grid: List[int],
    n_annotation_grid: List[int],
    n_label_grid: List[int],
) -> List[dict]:
    """
    Profile valor while generating Datasets of various sizes

    Parameters
    ----------
    client
        The Client object used to access your valor instance
    dataset_name
        The name of the dataset you want to use for profiling
    n_image_grid
        A list of integers describing the various image sizes you want to test
    n_prediction_grid
        A list of integers describing the various prediction sizes you want to test
    n_annotation_grid
        A list of integers describing the various annotation sizes you want to test
    n_label_grid
        A list of integers describing the various label sizes you want to test

    Returns
    -------
    list
        A list of output records which are also saved to ./profiles/{dataset_name}.pkl (in case of system failures)
    """
    output = list()
    for n_images in n_image_grid:
        for n_predictions in n_predictions_grid:
            for n_annotations in n_annotation_grid:
                for n_labels in n_label_grid:
                    kwargs = {
                        "client": client,
                        "dataset_name": dataset_name,
                        "model_name": dataset_name + "_model",
                        "n_images": n_images,
                        "n_predictions": n_predictions,
                        "n_annotations": n_annotations,
                        "n_labels": n_labels,
                    }

                    results = _profile_func(
                        _run_valor_profiling_functions, **kwargs
                    )

                    snapshot = _generate_docker_snapshot()
                    results = results | snapshot

                    output.append(results)

                    # create checkpoint in case of system failure
                    filepath = f"{os.getcwd()}/profiles/{dataset_name}.pkl"
                    os.makedirs(os.path.dirname(filepath), exist_ok=True)
                    with open(
                        filepath,
                        "wb+",
                    ) as f:
                        pickle.dump(output, f)

    return output


""" I/O Helpers """


def load_pkl(
    filepath: str,
) -> List[dict]:
    """
    Helper function to retrieve a saved profile from disk

    Parameters
    ----------
    dataset_name
        The name of the dataset you want to use for profiling

    Returns
    -------
    list
        A list of output records from your profiling function
    """
    with open(filepath, "rb") as f:
        output = pickle.load(f)

    return output


if __name__ == "__main__":

    warnings.filterwarnings("ignore")
    sns.set_style("whitegrid")

    LOCAL_HOST = "http://localhost:8000"
    DATASET_NAME = "profiling_3"

    client = Client.connect(LOCAL_HOST)

    results = profile_valor(
        client=client,
        dataset_name=DATASET_NAME,
        n_image_grid=[10],
        n_predictions_grid=[2],
        n_annotation_grid=[2],
        n_label_grid=[2],
    )
