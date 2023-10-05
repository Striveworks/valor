# %%
import warnings

import seaborn as sns
from src.profiling import profile_velour

from velour.client import Client

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

LOCAL_HOST = "http://localhost:8000"
DATASET_NAME = "profiling15"


client = Client(LOCAL_HOST)

results = profile_velour(
    client=client,
    dataset_name=DATASET_NAME,
    n_image_grid=[2],
    n_annotation_grid=[2],
    n_label_grid=[2],
    using_docker=False,
    db_container_name="velour-db-1",
    service_container_name="velour-service-1",
)

# %%
results

# %%
# import yappi

# stats = yappi.get_func_stats()
# stats.add("profiles/func_stats.out")

# stats.sort("tsub", "desc").print_all()

# %%
