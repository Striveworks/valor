from typing import Awaitable, Callable, Union

import structlog
from fastapi import Request, Response
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
    response = await call_next(request)
    logger.info(
        "Valor API Call",
        method=request.method,
        path=request.url.path,
        hostname=request.url.hostname,
        status=response.status_code,
    )
    return response


async def handle_request_validation_exception(
    request: Request, exc: RequestValidationError
) -> Union[JSONResponse, Response]:
    response = await fastapi_request_validation_exception_handler(request, exc)
    logger.warn("Valor request validation exception", errors=exc.errors())
    return response


async def handle_unhandled_exception(
    request: Request, exc: RequestValidationError
) -> Union[JSONResponse, Response]:
    logger.error(
        "Valor unhandled exception",
        method=request.method,
        path=request.url.path,
        hostname=request.url.hostname,
        exception=str(exc),
    )
    return JSONResponse(
        content={"status": 500, "detail": "Internal Server Error"},
        status_code=500,
    )
