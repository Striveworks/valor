# velour profiling utility

This directory contains decorators and helper functions that can be used to profile velour

## Usage

From `velour`, follow these instructions to generate profiling data.

### 1. Install python packages

```python

python -m pip install -r requirements.text

```

### 2. Setup profilers

Start by adding decorators to the backend endpoints that you want to profile. If you wanted to profile `/groundtruths`, for example, you'd add the following to lines to [`api/velour_api/main.py`](https://github.com/Striveworks/velour/blob/main/api/velour_api/main.py#L57):

```python

from utils.src.profiling import (
    generate_cprofile,
    generate_tracemalloc_profile,
    generate_yappi_profile,
)

...

@app.post(
    "/groundtruths",
    status_code=200,
    dependencies=[Depends(token_auth_scheme)],
    tags=["GroundTruths"],
)
@generate_yappi_profile(filepath="utils/profiles/create_groundtruths.yappi")
@generate_cprofile(filepath="utils/profiles/create_groundtruths.cprofile")
@generate_tracemalloc_profile(
    filepath="utils/profiles/create_groundtruths.tracemalloc"
)
def create_groundtruths(
    gt: schemas.GroundTruth, db: Session = Depends(get_db)
):
    try:
        ...

```

Finally, review the profiling settings in `utils/run_profiling.py` to adjust the number of images, annotations, labes, and predictions to generate during profiling.


### 3. Run profilers

To run these profiles while measuring client-side memory usage, we'd recommend using the following command:

```

mprof run utils/run_profiling.py --multiprocess

```

Visualize client-side memory usage using:

```

mprof plot -s

```

To delete all memory_profiler `.dat` outputs created by the memory profiler, use:

```

mprof clean

```