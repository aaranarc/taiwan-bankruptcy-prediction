import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from app.utils.logger import get_logger

logger = get_logger("logging_middleware")

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Log request basic info
        method = request.method
        url = request.url.path
        client_host = request.client.host if request.client else "unknown"
        
        logger.info(f"Incoming request: {method} {url} from {client_host}")
        
        response = await call_next(request)
        
        process_time = (time.time() - start_time) * 1000
        formatted_process_time = f"{process_time:.2f}ms"
        
        logger.info(
            f"Response: {method} {url} - Status Code: {response.status_code} "
            f"completed in {formatted_process_time}"
        )
        
        return response
