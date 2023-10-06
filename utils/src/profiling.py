import cProfile
import functools
import io
import os
import pickle
import pstats
import time
import timeit
import tracemalloc
from pstats import SortKey
from typing import List, Tuple

import memory_profiler
import pandas as pd
import yappi
from fastapi import HTTPException
from sqlalchemy.orm import Session

from velour.client import Client
from velour.client import Dataset as VelourDataset
from velour.client import Evaluation as VelourEvaluation
from velour.client import Model as VelourModel
from velour.data_generation import (
    generate_prediction_data,
    generate_segmentation_data,
)
from velour.enums import JobStatus

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
        The sqlalchemy session used by your backend
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
        The sqlalchemy session used by your backend
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
        The sqlalchemy session used by your backend
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
        The sqlalchemy session used by your backend
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
    fn(**kwargs)
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

    output.update(tracemalloc_output)


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


# TODO add type to container when available: https://github.com/docker/docker-py/issues/2796
def _get_container_stats(container, indicator: str = "") -> dict:
    stats = container.stats(stream=False)

    # cpu_perc calculation from https://stackoverflow.com/questions/30271942/get-docker-container-cpu-usage-as-percentage
    cpu_delta = (
        stats["cpu_stats"]["cpu_usage"]["total_usage"]
        - stats["precpu_stats"]["cpu_usage"]["total_usage"]
    )
    system_delta = (
        stats["cpu_stats"]["system_cpu_usage"]
        - stats["precpu_stats"]["system_cpu_usage"]
    )
    return {
        f"{indicator}memory_usage": stats["memory_stats"]["usage"],
        f"{indicator}memory_limit": stats["memory_stats"]["limit"],
        f"{indicator}cpu_usage": stats["cpu_stats"]["cpu_usage"][
            "total_usage"
        ],
        f"{indicator}cpu_usage_kernel": stats["cpu_stats"]["cpu_usage"][
            "usage_in_kernelmode"
        ],
        f"{indicator}cpu_usage_user": stats["cpu_stats"]["cpu_usage"][
            "usage_in_usermode"
        ],
        f"{indicator}cpu_usage_system": stats["cpu_stats"]["system_cpu_usage"],
        f"{indicator}cpu_throttled_time": stats["cpu_stats"][
            "throttling_data"
        ]["throttled_time"],
        f"{indicator}cpu_usage_perc": cpu_delta / system_delta,
    }


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


def _get_docker_pids():
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


def _generate_docker_snapshot():
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


""" Velour-Specific Functions """


def _setup_dataset(
    client: Client,
    dataset_name: str,
    n_images: int,
    n_annotations: int,
    n_labels: int,
) -> VelourDataset:
    """Generate a velour dataset with a given number of images, annotations, and labels"""
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
    dataset: VelourDataset,
    model_name: str,
    n_annotations: int,
    n_labels: int,
) -> Tuple[VelourModel, VelourEvaluation]:
    """Create arbitrary evaluation metrics based on some dataset"""

    model = generate_prediction_data(
        client=client,
        dataset=dataset,
        model_name=model_name,
        n_annotations=n_annotations,
        n_labels=n_labels,
    )

    eval_job = model.evaluate_ap(
        dataset=dataset,
        iou_thresholds=[0, 1],
        ious_to_keep=[0, 1],
        label_key="k1",
    )

    # sleep to give the backend time to compute
    time.sleep(1)
    assert eval_job.status == JobStatus.DONE
    return (model, eval_job)


def profile_velour(
    client: Client,
    dataset_name: str,
    n_image_grid: List[int],
    n_annotation_grid: List[int],
    n_label_grid: List[int],
) -> List[dict]:
    """
    Profile velour while generating VelourDatasets of various sizes

    Parameters
    ----------
    client
        The Client object used to access your velour instance
    dataset_name
        The name of the dataset you want to use for profiling
    n_image_grid
        A list of integers describing the various image sizes you want to test
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
        for n_annotations in n_annotation_grid:
            for n_labels in n_label_grid:
                kwargs = {
                    "client": client,
                    "dataset_name": dataset_name,
                    "n_images": n_images,
                    "n_annotations": n_annotations,
                    "n_labels": n_labels,
                }

                results = _profile_func(_setup_dataset, **kwargs)

                snapshot = _generate_docker_snapshot()
                results = results | snapshot

                output.append(results)

                # create checkpoint in case of system failure
                filepath = f"{os.getcwd()}/utils/profiles/{dataset_name}.pkl"
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
