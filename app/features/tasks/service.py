import logging
from sqlalchemy.orm import Session
from langchain_core.messages import HumanMessage
from agent_graph import app_graph

from app.core.chroma import sync_task_to_chroma, remove_task_from_chroma
from app.features.tasks.models import TaskModel
from app.features.tasks.parser import parse_with_llm_or_fallback, deterministic_nlp_parser
from app.features.tasks.schemas import (
    AgentAction,
    AgentPlan,
    ChatResponse,
    OUT_OF_SCOPE_REPLY,
)
from app.features.tasks.search import find_matching_tasks, generate_deterministic_id
from app.features.tasks.validation import safe_clarify_message, validate_plan, is_task_related

logger = logging.getLogger("TaskFlowAgent")


def handle_chat(db: Session, message: str, thread_id: str = "default_user") -> ChatResponse:
    # ONLY pass the new message. 
    # Let LangGraph fetch the rest of the state (like pending_action) from its memory.
    initial_state = {
        "messages": [HumanMessage(content=message)]
    }
    
    config = {"configurable": {"thread_id": thread_id, "db": db}}
    
    try:
        # The checkpointer resumes the state automatically!
        result = app_graph.invoke(initial_state, config=config)
    except Exception as e:
        logger.error(f"[GRAPH_ERROR] {e}")
        return ChatResponse(
            reply="Sorry, I encountered an internal error processing that.",
            tasks=db.query(TaskModel).all(),
            highlighted_ids=[],
            board_updated=False
        )

    return ChatResponse(
        reply=result["final_reply"],
        tasks=db.query(TaskModel).all(),
        highlighted_ids=result.get("highlighted", []),
        board_updated=result.get("board_updated", False)
    )

def list_tasks(db: Session) -> list[TaskModel]:
    return db.query(TaskModel).all()

def reset_demo_tasks(db: Session) -> list[TaskModel]:
    from app.features.tasks.seed import INITIAL_SEED_TASKS
    for t in db.query(TaskModel).all():
        remove_task_from_chroma(t.id)
    db.query(TaskModel).delete()
    db.commit()
    for item in INITIAL_SEED_TASKS:
        task = TaskModel(**item)
        db.add(task)
        db.commit()
        sync_task_to_chroma(task)
    return db.query(TaskModel).all()

def read_log_tail(lines: int) -> str:
    import os
    from app.config import LOG_FILE
    if not os.path.exists(LOG_FILE): return "Log file not created yet."
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        return "".join(f.readlines()[-lines:])