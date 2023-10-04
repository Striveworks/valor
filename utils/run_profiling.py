# %%[markdown]
# Setup
# %%
import pandas as pd
import seaborn as sns
from src.profiling import load_profile_from_disk, profile_velour

from velour.client import Client

sns.set_style("whitegrid")

LOCAL_HOST = "http://localhost:8000"
DATASET_NAME = "profiling5"

# %%[markdown]
# Setup Profiler
client = Client(LOCAL_HOST)

results = profile_velour(
    client=client,
    dataset_name=DATASET_NAME,
    n_image_grid=[100000],
    n_annotation_grid=[2],
    n_label_grid=[2],
    db_container_name="velour-db-1",
    service_container_name="velour-service-1",
)
# %%[markdown]
# Analyze Results

# %%
if "results" not in locals() and results not in globals():
    results = load_profile_from_disk()

df = pd.DataFrame.from_records(results)

df.columns
