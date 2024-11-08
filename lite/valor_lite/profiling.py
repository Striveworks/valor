import json
import math
import multiprocessing as mp
import resource
import sys
import time
from collections import deque
from multiprocessing import Queue


def _timeit_subprocess(*args, __fn, __queue: Queue, **kwargs):
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
    This profiles the runtime of the wrapped function in a subprocess.
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


def calculate_complexity(params: list[int | tuple[int]]) -> int:
    flattened_params = [
        math.prod(p) if isinstance(p, tuple) else p for p in params
    ]
    return math.prod(flattened_params)


def pretty_print_results(results: tuple):
    valid, invalid, permutations = results

    print(
        "====================================================================="
    )
    print("Details")
    print(json.dumps(permutations, indent=4))

    print()
    print("Passed")
    if len(valid) > 0:
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

    print()
    print("Failed")
    if len(invalid) > 0:
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

        # printing
        self.line_count = 0

    def get_limits(
        self,
        *_,
        readable: bool = True,
        memory_unit: str = "GB",
        time_unit: str = "seconds",
    ) -> dict[str, str | int | float | None]:

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

    def clear_status(self):
        if not self.verbose:
            return
        for _ in range(self.line_count):
            sys.stdout.write("\033[F")
            sys.stdout.write("\033[K")
        self.line_count = 0

    def write_status(self, text: str):
        if not self.verbose:
            return
        self.clear_status()
        self.line_count = text.count("\n") + 1
        sys.stdout.write(text + "\n")
        sys.stdout.flush()

    def run(
        self,
        benchmark,
        **kwargs,
    ):
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

        while queue:

            current_indices = queue.popleft()
            parameters = {
                k: v[current_indices[idx]]
                for idx, (k, v) in enumerate(zip(keys, vars))
            }
            complexity = calculate_complexity(tuple(parameters.values()))

            details: dict = {k: str(v) for k, v in parameters.items()}

            # update terminal with status
            self.write_status(
                f"Running '{benchmark.__name__}'\n"
                + json.dumps(
                    {
                        **details,
                        **self.get_limits(
                            readable=True,
                            memory_unit="GB",
                            time_unit="seconds",
                        ),
                    },
                    indent=4,
                )
            )

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
        self.clear_status()
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
        if self.verbose:
            pretty_print_results(results)

        return results
