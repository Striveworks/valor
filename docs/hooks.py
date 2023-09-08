# import os
# import re
# import shutil


# def copy_get(config, **kwargs):
#     site_dir = config["site_dir"]
#     url = "https://localhost:8000"
#     chart = config['chart']
#     print(chart)
#     os.system(f"helm package helm/velour-chart -d {site_dir}/helm")
#     os.system(f"helm repo index {site_dir}/helm --url {url}")
