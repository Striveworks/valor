import json
import math
import multiprocessing as mp
import resource
import time
from collections import deque
from multiprocessing import Queue
from typing import Any

from tqdm import tqdm


class BenchmarkError(Exception):
    def __init__(
        self, benchmark: str, error_type: str, error_message: str
    ) -> None:
        super().__init__(
            f"'{benchmark}' raised '{error_type}' with the following message: {error_message}"
        )


def _timeit_subprocess(*args, __fn, __queue: Queue, **kwargs):
    """
    Multiprocessing subprocess that reports either runtime or errors.

    This is handled within a subprocess to protect the benchmark against OOM errors.
    """
    try:
        timer_start = time.perf_counter()
        __fn(*args, **kwargs)
        timer_end = time.perf_counter()
        __queue.put(timer_end - timer_start)
    except Exception as e:
        __queue.put(e)


def create_runtime_profiler(
    time_limit: float | None,
    repeat: int = 1,
):
    """
    Creates a runtime profiler as a decorating function.

    The profiler reports runtime of the wrapped function from a subprocess to protect against OOM errors.

    Parameters
    ----------
    time_limit : float, optional
        An optional time limit to constrain the benchmark.
    repeat : int, default=1
        The number of times to repeat the benchmark to produce an average runtime.
    """
    ctx = mp.get_context("spawn")

    def decorator(fn):
        def wrapper(*args, **kwargs):
            # Record average runtime over repeated runs.
            elapsed = 0
            for _ in range(repeat):
                q = ctx.Queue()
                p = ctx.Process(
                    target=_timeit_subprocess,
                    args=args,
                    kwargs={"__fn": fn, "__queue": q, **kwargs},
                )
                p.start()
                p.join(timeout=time_limit)

                # Check if computation finishes within the timeout
                if p.is_alive():
                    p.terminate()
                    p.join()
                    q.close()
                    q.join_thread()
                    raise TimeoutError(
                        f"Function '{fn.__name__}' did not complete within {time_limit} seconds."
                    )

                # Retrieve the result
                result = q.get(timeout=1)
                if isinstance(result, Exception):
                    raise result
                elif isinstance(result, float):
                    elapsed += result
                else:
                    raise TypeError(type(result).__name__)

            return elapsed / repeat

        return wrapper

    return decorator


def pretty_print_results(results: tuple):
    valid, invalid, permutations = results

    print(
        "====================================================================="
    )
    print("Details")
    print(json.dumps(permutations, indent=4))

    if len(valid) > 0:
        print()
        print("Passed")
        keys = ["complexity", "runtime", *valid[0]["details"].keys()]
        header = " | ".join(f"{header:^15}" for header in keys)
        print(header)
        print("-" * len(header))
        for entry in valid:
            values = [
                entry["complexity"],
                round(entry["runtime"], 4),
                *entry["details"].values(),
            ]
            row = " | ".join(f"{str(value):^15}" for value in values)
            print(row)

    if len(invalid) > 0:
        print()
        print("Failed")
        keys = ["complexity", "error", *invalid[0]["details"].keys(), "msg"]
        header = " | ".join(f"{header:^15}" for header in keys)
        print(header)
        print("-" * len(header))
        for entry in invalid:
            values = [
                entry["complexity"],
                entry["error"],
                *entry["details"].values(),
                entry["msg"],
            ]
            row = " | ".join(f"{str(value):^15}" for value in values)
            print(row)


def _calculate_complexity(params: list[int | tuple[int]]) -> int:
    """
    Basic metric of benchmark complexity.
    """
    flattened_params = [
        math.prod(p) if isinstance(p, tuple) else p for p in params
    ]
    return math.prod(flattened_params)


class Benchmark:
    def __init__(
        self,
        time_limit: float | None,
        memory_limit: int | None,
        *_,
        repeat: int | None = 1,
        verbose: bool = False,
    ):
        self.time_limit = time_limit
        self.memory_limit = memory_limit
        self.repeat = repeat
        self.verbose = verbose

    def get_limits(
        self,
        *_,
        readable: bool = True,
        memory_unit: str = "GB",
        time_unit: str = "seconds",
    ) -> dict[str, str | int | float | None]:
        """
        Returns a dictionary of benchmark limits.

        Parameters
        ----------
        readable : bool, default=True
            Toggles whether the output should be human readable.
        memory_unit : str, default="GB"
            Toggles what unit to display the memory limit with when 'readable=True'.
        time_unit : str, default="seconds"
            Toggles what unit to display the time limit with when 'readable=True'.

        Returns
        -------
        dict[str, str | int | float | None]
            The benchmark limits.
        """

        memory_value = self.memory_limit
        if readable and memory_value is not None:
            match memory_unit:
                case "TB":
                    memory_value /= 1024**4
                case "GB":
                    memory_value /= 1024**3
                case "MB":
                    memory_value /= 1024**2
                case "KB":
                    memory_value /= 1024
                case "B":
                    pass
                case _:
                    valid_set = {"TB", "GB", "MB", "KB", "B"}
                    raise ValueError(
                        f"Expected memory unit to be in the set {valid_set}, received '{memory_unit}'."
                    )
            memory_value = f"{memory_value} {memory_unit}"

        time_value = self.time_limit
        if readable and time_value is not None:
            match time_unit:
                case "minutes":
                    time_value /= 60
                case "seconds":
                    pass
                case "milliseconds":
                    time_value *= 1000
                case _:
                    valid_set = {"minutes", "seconds", "milliseconds"}
                    raise ValueError(
                        f"Expected time unit to be in the set {valid_set}, received '{time_unit}'."
                    )
            time_value = f"{time_value} {time_unit}"

        return {
            "memory_limit": memory_value,
            "time_limit": time_value,
            "repeat": self.repeat,
        }

    @property
    def memory_limit(self) -> int | None:
        """
        The memory limit in bytes (B).
        """
        return self._memory_limit

    @memory_limit.setter
    def memory_limit(self, limit: int | None):
        """
        Stores the memory limit and restricts resources.
        """
        self._memory_limit = limit
        if limit is not None:
            _, hard = resource.getrlimit(resource.RLIMIT_AS)
            resource.setrlimit(resource.RLIMIT_AS, (limit, hard))

    def run(
        self,
        benchmark,
        **kwargs: list[Any],
    ):
        """
        Runs a benchmark with ranges of parameters.

        Parameters
        ----------
        benchmark : Callable
            The benchmark function.
        **kwargs : list[Any]
            Keyword arguments passing lists of parameters to benchmark. The values should be sorted in
            decreasing complexity. For example, if the number of labels is a parameter then a higher
            number of unique labels would be considered "more" complex.

        Example
        -------
        >>> b = Benchmark(
        ...     time_limit=10.0,
        ...     memory_limit=8 * (1024**3),
        ...     repeat=1,
        ...     verbose=False,
        ... )
        >>> results = b.run(
        ...     benchmark=semseg_add_data,
        ...     n_labels=[
        ...         100,
        ...         10,
        ...     ],
        ...     shape=[
        ...         (1000, 1000),
        ...         (100, 100),
        ...     ],
        ... )
        """

        nvars = len(kwargs)
        keys = tuple(kwargs.keys())
        vars = tuple(kwargs[key] for key in keys)

        initial_indices = tuple(0 for _ in range(nvars))
        max_indices = tuple(len(v) for v in vars)
        permutations = math.prod(max_indices)

        # Initialize queue with the starting index (0, ...)
        queue = deque()
        queue.append(initial_indices)

        # Keep track of explored combinations to avoid duplicates
        explored = set()
        explored.add(initial_indices)

        # Store valid combinations that finish within the time limit
        valid_combinations = []
        invalid_combinations = []

        pbar = tqdm(total=math.prod(max_indices), disable=(not self.verbose))
        prev_count = 0
        while queue:

            current_indices = queue.popleft()
            parameters = {
                k: v[current_indices[idx]]
                for idx, (k, v) in enumerate(zip(keys, vars))
            }
            complexity = _calculate_complexity(list(parameters.values()))

            details: dict = {k: str(v) for k, v in parameters.items()}

            # update terminal with status
            count = len(valid_combinations) + len(invalid_combinations)
            pbar.update(count - prev_count)
            prev_count = count

            try:
                runtime = benchmark(
                    time_limit=self.time_limit,
                    repeat=self.repeat,
                    **parameters,
                )
                valid_combinations.append(
                    {
                        "complexity": complexity,
                        "runtime": runtime,
                        "details": details,
                    }
                )
                continue
            except Exception as e:
                invalid_combinations.append(
                    {
                        "complexity": complexity,
                        "error": type(e).__name__,
                        "msg": str(e),
                        "details": details,
                    }
                )

            for idx in range(nvars):
                new_indices = list(current_indices)
                if new_indices[idx] + 1 < max_indices[idx]:
                    new_indices[idx] += 1
                    new_indices_tuple = tuple(new_indices)
                    if new_indices_tuple not in explored:
                        queue.append(new_indices_tuple)
                        explored.add(new_indices_tuple)

        valid_combinations.sort(key=lambda x: -x["complexity"])
        invalid_combinations.sort(key=lambda x: -x["complexity"])

        # clear terminal and display results
        results = (
            valid_combinations,
            invalid_combinations,
            {
                "benchmark": benchmark.__name__,
                "limits": self.get_limits(readable=True),
                "passed": permutations - len(invalid_combinations),
                "failed": len(invalid_combinations),
                "total": permutations,
            },
        )
        pbar.close()
        if self.verbose:
            pretty_print_results(results)

        return results
