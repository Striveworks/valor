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
# import os


# from memory_profiler import LogFile
# import sys
# import logging

# # create loggerf
# logger = logging.getLogger("profiles/memory_profile_log")
# logger.setLevel(logging.DEBUG)

# # create file handler which logs even debug messages
# fh = logging.FileHandler("profiles/memory_profile.log")
# fh.setLevel(logging.DEBUG)

# # create formatter
# formatter = logging.Formatter(
#     "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
# )
# fh.setFormatter(formatter)

# # add the handlers to the logger
# logger.addHandler(fh)

# sys.stdout = LogFile("memory_profile_log", reportIncrementFlag=True)


# @profile
# def test(a, b):
#     a = 2
#     b = 4
#     return a * b


# # os.makedirs(os.path.dirname(test_filepath), exist_ok=True)

# # with open(
# #     test_filepath,
# #     "wb+",
# # ) as f:
# #     result = profile(func=test(1, 2), stream=f)

# test(2, 4)


# # %%
# # %%
# # import yappi

# # stats = yappi.get_func_stats()
# # stats.add("profiles/func_stats.out")

# # stats.sort("tsub", "desc").print_all()

# # %%
