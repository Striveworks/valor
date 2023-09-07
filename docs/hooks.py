import os
import shutil


def copy_get(config, **kwargs):
    site_dir = config["site_dir"]
    shutil.make_archive(
        base_name="helm/velour-chart-latest",
        format="gztar",
        root_dir=".",
        base_dir="helm/velour-chart",
    )
    shutil.copy(
        "helm/velour-chart-latest.tar.gz",
        os.path.join(site_dir, "velour-chart-latest.tgz"),
    )
