import re
import logging
from typing import List, Optional
from app.features.tasks.schemas import AgentPlan

logger = logging.getLogger("TaskFlowAgent")

MAX_ACTIONS_PER_REQUEST = 5

ALLOWED_ACTIONS = {
    "CREATE_TASK", "UPDATE_TASK", "DELETE_TASK", "SEARCH_TASKS",
    "LIST_TASKS", "CLARIFY", "OUT_OF_SCOPE",
}

# Robust Regex Filters
TASK_WORDS = re.compile(
    r"\b(task|tasks|todo|to-do|board|backlog|in progress|to do|done|status|priority|"
    r"due|deadline|high|medium|low|complete|finish|mark|move|create|add|new|delete|"
    r"remove|rename|edit|update|change|set|assign|reopen|list|show|find|search|"
    r"unfinished|pending|overdue)\b", re.I)

OFF_TOPIC = re.compile(
    r"\b(joke|poem|story|essay|song|riddle|python|javascript|java|golang|\bgo\b|"
    r"c\+\+|rust|sql|html|css|code|program|script|function|algorithm|translate|"
    r"recipe|weather|capital of|who is|who was|what is|explain|meaning of|"
    r"ignore (all |previous )?instructions|system prompt|pretend|roleplay)\b", re.I)

def is_task_related(text: str, known_titles: List[str]) -> bool:
    t = text.strip()
    if not t or len(t) > 500:
        return False
    
    low = t.lower()
    if any(title.lower() in low for title in known_titles):
        return True
        
    if OFF_TOPIC.search(t) and not re.search(r"\b(task|tasks|board)\b", t, re.I):
        return False
        
    return bool(TASK_WORDS.search(t))

def safe_clarify_message(msg: Optional[str]) -> str:
    if not msg:
        return "Which task did you mean? Please be more specific."
    msg = msg.strip()
    if len(msg) > 200 or "```" in msg or "\n" in msg:
        return "Which task did you mean? Please be more specific."
    return msg

def validate_plan(plan: AgentPlan) -> None:
    if not plan.actions:
        raise ValueError("Empty action list")
    if len(plan.actions) > MAX_ACTIONS_PER_REQUEST:
        raise ValueError(f"Too many actions ({len(plan.actions)})")
    for a in plan.actions:
        if a.action not in ALLOWED_ACTIONS:
            raise ValueError(f"Disallowed action: {a.action}")