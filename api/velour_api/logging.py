from typing import Callable

import structlog
from fastapi import BackgroundTasks, Request, Response
from fastapi.routing import APIRoute
from starlette.background import BackgroundTask

logger = structlog.get_logger()


def log_request(
    request: Request, ignore_paths=frozenset(["/health", "/ready"])
):
    if request.url.path in ignore_paths:
        return
    logger.info(
        "Velour API Call",
        method=request.method,
        path=request.url.path,
        hostname=request.url.hostname,
    )


def add_task(response: Response, task: BackgroundTask):
    if not response.background:
        response.background = BackgroundTasks([task])
    elif isinstance(response.background, BackgroundTasks):
        response.background.add_task(task)
    else:  # Empirically this doesn't happen but let's handle it anyway
        if not isinstance(response.background, BackgroundTask):
            logger.error(
                "Unexpected response.background",
                background_type=str(type(response.background)),
            )
        old_task = response.background
        response.background = BackgroundTasks([old_task, task])

    return response


class LoggingRoute(APIRoute):
    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request) -> Response:
            response = await original_route_handler(request)
            task = BackgroundTask(log_request, request)
            return add_task(response, task)

        return custom_route_handler
