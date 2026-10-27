from typing import List, Optional, Literal
from pydantic import BaseModel, Field, field_validator

StatusType = Literal["To do", "In progress", "Done"]
PriorityType = Literal["Low", "Medium", "High"]

# GENERAL_RESPONSE is REMOVED.
ActionType = Literal[
    "CREATE_TASK",
    "UPDATE_TASK",
    "DELETE_TASK",
    "SEARCH_TASKS",
    "LIST_TASKS",
    "CLARIFY",
    "OUT_OF_SCOPE",
]

TITLE_MAX = 200
DESCRIPTION_MAX = 1000
TARGET_QUERY_MAX = 200
MESSAGE_MAX = 200
DUE_DATE_MAX = 50
USER_MESSAGE_MAX = 500

class TaskSchema(BaseModel):
    id: str
    title: str = Field(..., min_length=1, max_length=TITLE_MAX)
    description: Optional[str] = Field("", max_length=DESCRIPTION_MAX)
    status: StatusType = "To do"
    priority: PriorityType = "Medium"
    due_date: Optional[str] = Field("", max_length=DUE_DATE_MAX)

    class Config:
        from_attributes = True

class AgentAction(BaseModel):
    action: ActionType
    target_query: Optional[str] = Field(None, max_length=TARGET_QUERY_MAX)
    title: Optional[str] = Field(None, min_length=1, max_length=TITLE_MAX)
    description: Optional[str] = Field("", max_length=DESCRIPTION_MAX)
    status: Optional[StatusType] = None
    priority: Optional[PriorityType] = None
    due_date: Optional[str] = Field(None, max_length=DUE_DATE_MAX)
    message: Optional[str] = Field(None, max_length=MESSAGE_MAX)

    @field_validator("title", "target_query")
    @classmethod
    def strip_and_validate(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = v.strip()
        if not v:
            raise ValueError("must contain non-whitespace characters")
        return v

class AgentPlan(BaseModel):
    actions: List[AgentAction]

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=USER_MESSAGE_MAX)

class ChatResponse(BaseModel):
    reply: str
    tasks: List[TaskSchema]
    highlighted_ids: List[str] = []
    board_updated: bool = False

# Fixed server-side reply
OUT_OF_SCOPE_REPLY = (
    "I can only help manage your tasks. Try: 'Create a high-priority task to finish "
    "the demo tomorrow', 'Move the docs task to In Progress', or "
    "'Show unfinished tasks about documentation'."
)