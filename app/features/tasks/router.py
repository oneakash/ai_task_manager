from typing import List

from fastapi import APIRouter, Depends, Request
from fastapi.responses import PlainTextResponse
from sqlalchemy.orm import Session

from app.dependencies import get_db
from app.rate_limit import limiter
from app.features.tasks import service
from app.features.tasks.schemas import TaskSchema, ChatRequest, ChatResponse

router = APIRouter()


@router.get("/tasks", response_model=List[TaskSchema])
def get_tasks(db: Session = Depends(get_db)):
    return service.list_tasks(db)


@router.post("/reset", response_model=List[TaskSchema])
def reset_demo_tasks(db: Session = Depends(get_db)):
    return service.reset_demo_tasks(db)


@router.get("/logs", response_class=PlainTextResponse)
def get_application_logs(lines: int = 100):
    return service.read_log_tail(lines)


@router.post("/chat", response_model=ChatResponse)
@limiter.limit("10/minute")
def process_chat(request: Request, req: ChatRequest, db: Session = Depends(get_db)):
    # UPDATED: Pass req.thread_id to the service
    return service.handle_chat(db, req.message, thread_id=req.thread_id)