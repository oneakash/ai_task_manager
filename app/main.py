import logging

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler

from app.logging_config import setup_logging
from app.rate_limit import limiter
from app.features.tasks.router import router as tasks_router
from app.features.tasks.seed import seed_database

setup_logging()
logger = logging.getLogger("TaskFlowAgent")
logger.info("==========================================")
logger.info("Initializing TASKFLOW AI Agent Task Manager")
logger.info("==========================================")


@asynccontextmanager
async def lifespan(app: FastAPI):
    seed_database()
    yield


app = FastAPI(title="TASKFLOW - AI Agent Task Manager", lifespan=lifespan)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.middleware("http")
async def log_http_requests(request: Request, call_next):
    import time

    start_t = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start_t) * 1000
    logger.info(
        f"[HTTP] {request.method} {request.url.path} -> Status {response.status_code} ({duration_ms:.1f}ms)"
    )
    return response


app.include_router(tasks_router, prefix="/api", tags=["tasks"])

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def serve_spa():
    return FileResponse("static/index.html")