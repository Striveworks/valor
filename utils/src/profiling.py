import cProfile
import os
import pickle
import timeit
import tracemalloc
from typing import List

import docker

from velour.client import Client
from velour.client import Dataset as VelourDataset
from velour.data_generation import generate_segmentation_data


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


def _generate_cprofile(fn: callable, filename: str, **kwargs) -> None:
    """Wrapper to generate a cprofile report and save it in a given filename"""
    pr = cProfile.Profile()
    pr.enable()

    fn(**kwargs)

    pr.disable()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    pr.dump_stats(filename)


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


def profile_velour(
    client: Client,
    dataset_name: str,
    n_image_grid: List[int],
    n_annotation_grid: List[int],
    n_label_grid: List[int],
    using_docker: bool = True,
    db_container_name: str = "velour-db-1",
    service_container_name: str = "velour-service-1",
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
    using_docker
        A boolean describing whether or not you're using Docker. If True, various Docker memory and CPU stats will be added to the output.
    db_container_name
        The name of your database container on Docker
    service_container_name
        The name of your service container on Docker

    Returns
    -------
    list
        A list of output records which are also saved to ./profiles/{dataset_name}.pkl (in case of system failures)
    """
    if using_docker:
        docker_client = docker.from_env()
        db_container = docker_client.containers.get(db_container_name)
        service_container = docker_client.containers.get(
            service_container_name
        )

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

                if using_docker:
                    service_stats = _get_container_stats(
                        container=service_container,
                        indicator="service_",
                    )
                    db_stats = _get_container_stats(
                        container=db_container,
                        indicator="db_",
                    )
                    results = results | service_stats | db_stats

                output.append(results)

                # create checkpoint in case of system failure
                filename = f"{os.getcwd()}/profiles/{dataset_name}.pkl"
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                with open(
                    filename,
                    "wb+",
                ) as f:
                    pickle.dump(output, f)

    return output


def load_profile_from_disk(
    dataset_name: str,
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
    with open(f"{os.getcwd()}/profiles/{dataset_name}.pkl", "rb") as f:
        output = pickle.load(f)

    return output
