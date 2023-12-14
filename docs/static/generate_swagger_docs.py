# Generates the swagger docs shown in `docs/endpoints.md`
import json

from fastapi.openapi.utils import get_openapi

from velour_api.main import app

with open("docs/static/openapi.json", "w") as f:
    json.dump(
        get_openapi(
            title=app.title,
            version=app.version,
            openapi_version=app.openapi_version,
            description=app.description,
            routes=app.routes,
        ),
        f,
    )
