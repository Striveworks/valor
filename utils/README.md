# valor profiling utility

This directory contains decorators and helper functions that can be used to profile valor

## Usage

From `valor`, follow these instructions to generate profiling data.

### 1. Install python packages

```python

python -m pip install -r requirements.text

```

### 2. Setup profilers

Start by adding decorators to the backend endpoints that you want to profile. If you wanted to profile `/groundtruths`, for example, you'd add the following to lines to [`api/valor_api/main.py`](https://github.com/Striveworks/valor/blob/main/api/valor_api/main.py#L57):

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

python -m mprof run utils/run_profiling.py --multiprocess

```

Visualize client-side memory usage using:

```

python -m mprof plot -s

```

To delete all memory_profiler `.dat` outputs created by the memory profiler, use:

```

python -m mprof clean

```

### 4. Analyze results

See `utils/analyze_profiling_data.ipynb` for examples of how you can use these various profilers.