# A script to generate profiling reports for Valor
# See README.md for setup instructions

import warnings

import seaborn as sns
from src.profiling import profile_valor

from valor import Client

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

LOCAL_HOST = "http://localhost:8000"
DATASET_NAME = "profiling_3"


client = Client(LOCAL_HOST)


results = profile_valor(
    client=client,
    dataset_name=DATASET_NAME,
    n_image_grid=[10],
    n_predictions_grid=[2],
    n_annotation_grid=[2],
    n_label_grid=[2],
)
