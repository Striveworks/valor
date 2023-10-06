# A script to generate profiling reports for Velour
# See README.md for setup instructions

import warnings

import seaborn as sns
from src.profiling import profile_velour

from velour.client import Client

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

LOCAL_HOST = "http://localhost:8000"
DATASET_NAME = "profiling16"


client = Client(LOCAL_HOST)


results = profile_velour(
    client=client,
    dataset_name=DATASET_NAME,
    n_image_grid=[2],
    n_annotation_grid=[2],
    n_label_grid=[2],
)
