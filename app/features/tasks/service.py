import logging
from sqlalchemy.orm import Session

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

def handle_chat(db: Session, message: str) -> ChatResponse:
    # LAYER 1: Block off-topic messages before the LLM is called
    known_titles = [t.title for t in db.query(TaskModel).all()]
    if not is_task_related(message, known_titles):
        logger.info(f"[OUT_OF_SCOPE] Blocked before LLM: {message!r}")
        return ChatResponse(
            reply=OUT_OF_SCOPE_REPLY,
            tasks=db.query(TaskModel).all(),
            highlighted_ids=[],
            board_updated=False
        )

    # Parse + validate plan
    try:
        plan = parse_with_llm_or_fallback(message)
        validate_plan(plan)
    except ValueError as e:
        logger.error(f"[PLAN_VALIDATION_FAILED] {e} - falling back to regex parser")
        try:
            plan = deterministic_nlp_parser(message)
            validate_plan(plan)
        except ValueError as e2:
            logger.error(f"[PLAN_VALIDATION_FAILED_2] {e2} - returning fixed reply")
            plan = AgentPlan(actions=[AgentAction(action="OUT_OF_SCOPE")])

    replies: list[str] = []
    highlighted: list[str] = []
    board_updated = False

    for idx, act in enumerate(plan.actions, start=1):
        logger.info(f"[ACTION_EXEC_{idx}] action={act.action} | payload={act.model_dump()}")

        if act.action == "CREATE_TASK":
            new_id = generate_deterministic_id(db)
            safe_title = (act.title or "Untitled Task")[:200] # Hard truncation
            new_task = TaskModel(
                id=new_id, title=safe_title, description=(act.description or "")[:1000],
                status=act.status or "To do", priority=act.priority or "Medium", due_date=act.due_date or "Tomorrow"
            )
            db.add(new_task)
            db.commit()
            db.refresh(new_task)
            sync_task_to_chroma(new_task)
            msg = f'Created task "{new_task.title}" ({new_task.priority} priority, due {new_task.due_date}).'
            logger.info(f"[SQLITE_CREATED] id={new_task.id} | {msg}")
            replies.append(msg)
            highlighted.append(new_task.id)
            board_updated = True

        elif act.action == "UPDATE_TASK":
            res = find_matching_tasks(db, act.target_query or "", for_mutation=True)
            if not res["matches"]:
                replies.append(f'Could not find any task matching "{act.target_query}".')
            elif res["ambiguous"]:
                titles = ", ".join(f'"{m.title}"' for m in res["matches"])
                replies.append(f'Clarification needed: Multiple tasks match "{act.target_query}" ({titles}).')
            else:
                target = res["matches"][0]
                changes = []
                if act.status:
                    target.status = act.status
                    if act.status == "Done": target.due_date = "Completed"
                    changes.append(f"to {act.status}")
                if act.priority: target.priority = act.priority; changes.append(f"priority to {act.priority}")
                if act.due_date: target.due_date = act.due_date; changes.append(f"due date to {act.due_date}")
                if act.title: target.title = act.title[:200]; changes.append(f'title to "{target.title}"')
                
                db.commit()
                db.refresh(target)
                sync_task_to_chroma(target)
                replies.append(f'Updated "{target.title}" ({" and ".join(changes) or "updated"}).')
                highlighted.append(target.id)
                board_updated = True

        elif act.action == "DELETE_TASK":
            res = find_matching_tasks(db, act.target_query or "", for_mutation=True)
            if not res["matches"]:
                replies.append(f'Could not find any task matching "{act.target_query}" to delete.')
            elif res["ambiguous"]:
                replies.append(f'Clarification needed before deleting: Multiple tasks match "{act.target_query}".')
            else:
                target = res["matches"][0]
                remove_task_from_chroma(target.id)
                db.delete(target)
                db.commit()
                replies.append(f'Deleted task "{target.title}".')
                board_updated = True

        elif act.action in ("SEARCH_TASKS", "LIST_TASKS"):
            status_filter = act.message if act.message == "UNFINISHED" else act.status
            res = find_matching_tasks(db, act.target_query or "", status_filter=status_filter, priority_filter=act.priority, for_mutation=False)
            if not res["matches"]:
                replies.append(f'No matching tasks found for "{act.target_query or "your filter"}".')
            else:
                found = ", ".join(m.title for m in res["matches"])
                replies.append(f"Found: {found}")
                highlighted.extend([m.id for m in res["matches"]])

        elif act.action == "CLARIFY":
            replies.append(safe_clarify_message(act.message))

        else:  
            # LAYER 2: OUT_OF_SCOPE kill switch. LLM text is completely ignored.
            logger.warning(f"[OUT_OF_SCOPE] Blocked free-text response: action={act.action} message='{act.message}'")
            replies.append(OUT_OF_SCOPE_REPLY)

    final_reply = " ".join(replies)
    if board_updated and not final_reply.startswith("Done."):
        final_reply = "Done. " + final_reply

    return ChatResponse(
        reply=final_reply,
        tasks=db.query(TaskModel).all(),
        highlighted_ids=highlighted,
        board_updated=board_updated
    )

# (Keep list_tasks, reset_demo_tasks, and read_log_tail here as they were previously)
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