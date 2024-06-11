import time
from typing import Awaitable, Callable, Union

import structlog
from fastapi import HTTPException, Request, Response
from fastapi.exception_handlers import (
    request_validation_exception_handler as fastapi_request_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = structlog.get_logger()


async def log_endpoint_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Union[Response, JSONResponse]]],
) -> Union[Response, JSONResponse]:
    start_time = time.monotonic()
    response = await call_next(request)
    duration_seconds = time.monotonic() - start_time
    logger.info(
        "Valor API Call",
        method=request.method,
        path=request.url.path,
        hostname=request.url.hostname,
        status=response.status_code,
        duration_ms=duration_seconds * 1000
    )
    return response


async def handle_request_validation_exception(
    request: Request, exc: RequestValidationError
) -> Union[JSONResponse, Response]:
    response = await fastapi_request_validation_exception_handler(request, exc)
    logger.warn("Valor request validation exception", errors=exc.errors())
    return response


async def handle_http_exception(
    request: Request, exc: HTTPException
) -> Union[JSONResponse, Response]:
    if exc.status_code >= 500:
        logger.error(
            "Valor HTTP exception",
            method=request.method,
            path=request.url.path,
            hostname=request.url.hostname,
            exc_info=exc,
        )
    return JSONResponse(
        content={"status": exc.status_code, "detail": exc.detail},
        status_code=exc.status_code,
    )


async def handle_unhandled_exception(
    request: Request, exc: Exception
) -> Union[JSONResponse, Response]:
    logger.error(
        "Valor unhandled exception",
        method=request.method,
        path=request.url.path,
        hostname=request.url.hostname,
        exc_info=exc,
    )
    return JSONResponse(
        content={"status": 500, "detail": "Internal Server Error"},
        status_code=500,
    )
