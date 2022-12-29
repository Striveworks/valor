from typing import List
from urllib.parse import urljoin

import requests
from velour.data_types import BoundingPolygon, GroundTruthDetection


def _payload_for_bounding_polygon(poly: BoundingPolygon) -> List[List[int]]:
    return [[pt.x, pt.y] for pt in poly.points]


class Client:
    """Client for interacting with the velour backend

    TODO: create a client for "local mode" where a sqlite-type db
    replaces postgres
    """

    def __init__(self, host: str):
        if not (host.startswith("http://") or host.startswith("https://")):
            raise ValueError(
                f"host must stat with 'http://' or 'https://' but got {host}"
            )
        self.host = host

    def upload_gt_detection(
        self, det: GroundTruthDetection
    ) -> requests.Response:
        payload = {
            "boundary": _payload_for_bounding_polygon(det.boundary),
            "class_label": det.class_label,
        }

        url = urljoin(self.host, "groundtruth-detections")
        return requests.post(url, json=payload)
