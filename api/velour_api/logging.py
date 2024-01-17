from typing import Callable

import structlog
from fastapi import Request, Response
from fastapi.routing import APIRoute
from starlette.background import BackgroundTask

logger = structlog.get_logger()


def log_request(request: Request):
    logger.info(
        "Velour API Call",
        method=request.method,
        path=request.url.path,
        hostname=request.url.hostname,
    )


class LoggingRoute(APIRoute):
    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request) -> Response:
            response = await original_route_handler(request)
            response.background = BackgroundTask(log_request, request)
            return response

        return custom_route_handler
