import os
import logging
import operator
from typing import TypedDict, List, Optional, Literal, Annotated
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig # <--- IMPORT THIS
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from sqlalchemy.orm import Session
from app.features.tasks.schemas import AgentPlan, AgentAction, OUT_OF_SCOPE_REPLY
from app.features.tasks.search import find_matching_tasks, generate_deterministic_id
from app.features.tasks.validation import is_task_related, validate_plan
from app.core.chroma import sync_task_to_chroma, remove_task_from_chroma
from app.features.tasks.models import TaskModel

logger = logging.getLogger("TaskFlowAgent")

# --- 1. Define the Graph State ---
class AgentState(TypedDict):
    # Annotated with operator.add tells LangGraph to append to the list, not overwrite it
    messages: Annotated[List[BaseMessage], operator.add] 
    plan: Optional[AgentPlan]
    pending_action: Optional[dict]
    replies: List[str]
    highlighted: List[str]
    board_updated: bool
    final_reply: str

# --- 2. Define the LLM Structured Output ---
class LLMActionSchema(BaseModel):
    action: Literal["CREATE_TASK", "UPDATE_TASK", "DELETE_TASK", "SEARCH_TASKS", "LIST_TASKS", "OUT_OF_SCOPE"]
    target_query: Optional[str] = Field(None, description="The task name to update/delete/search. Leave null if unknown.")
    title: Optional[str] = None
    priority: Optional[Literal["Low", "Medium", "High"]] = None
    status: Optional[Literal["To do", "In progress", "Done"]] = None
    due_date: Optional[str] = None
    message: Optional[str] = None

class LLMPlanSchema(BaseModel):
    actions: List[LLMActionSchema]

# Initialize Groq
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="openai/gpt-oss-20b",
    temperature=0
).with_structured_output(LLMPlanSchema)


# --- 3. Define the Nodes (UPDATED WITH RunnableConfig) ---

def guardrail_node(state: AgentState, config: RunnableConfig):
    """Blocks off-topic messages before doing anything else."""
    if state.get("pending_action"):
        return {}

    db: Session = config["configurable"]["db"]
    message = state["messages"][-1].content
    message_lower = message.lower()
    
    # BYPASS: If the user uses obvious task verbs, let the LLM handle it.
    task_verbs = ["create", "add", "new task", "update", "delete", "remove", "move", "mark", "complete", "show", "find", "search", "list"]
    if any(verb in message_lower for verb in task_verbs):
        return {}

    known_titles = [t.title for t in db.query(TaskModel).all()]
    if not is_task_related(message, known_titles):
        logger.info(f"[OUT_OF_SCOPE] Blocked before LLM: {message!r}")
        return {"final_reply": OUT_OF_SCOPE_REPLY, "board_updated": False}
    return {}

def parse_node(state: AgentState, config: RunnableConfig):
    """Parses the user intent, OR merges it with a pending clarification."""
    user_message = state["messages"][-1].content
    pending = state.get("pending_action")

    # If we are waiting for a task name, merge it and skip the LLM
    if pending:
        logger.info(f"[CLARIFICATION_RESUME] Applying target '{user_message}' to pending action.")
        updated_action = dict(pending) # Copy to avoid mutating state directly
        updated_action["target_query"] = user_message
        merged_plan = AgentPlan(actions=[AgentAction(**updated_action)])
        return {"plan": merged_plan, "pending_action": None} # Clear pending state!

    try:
        # UPDATED SYSTEM PROMPT
        system_msg = SystemMessage(content="""You are TaskFlow AI. Parse the user's instruction into a structured JSON plan.
        
        Examples:
        - "Create a high priority task to finish the demo tomorrow" -> {"actions": [{"action": "CREATE_TASK", "title": "finish the demo", "priority": "High", "status": "To do", "due_date": "Tomorrow"}]}
        - "create new task titled meet with rasel bhai" -> {"actions": [{"action": "CREATE_TASK", "title": "meet with rasel bhai", "priority": "Medium", "status": "To do", "due_date": "Tomorrow"}]}
        - "Update the priority of the API task to Low" -> {"actions": [{"action": "UPDATE_TASK", "target_query": "API task", "priority": "Low"}]}
        - "Move the demo to In progress" -> {"actions": [{"action": "UPDATE_TASK", "target_query": "demo", "status": "In progress"}]}
        - "Show unfinished tasks" -> {"actions": [{"action": "SEARCH_TASKS", "status": "To do", "message": "UNFINISHED"}]}
        
        Rules: 
        - If they want to update/delete but don't specify which task, set target_query to null.
        - Valid priorities: Low, Medium, High. Valid statuses: To do, In progress, Done.
        - Always output valid JSON only.""")
        
        result = llm.invoke([system_msg] + state["messages"])
        # Convert Pydantic LLM output to your internal AgentPlan schema
        plan = AgentPlan(actions=[AgentAction(**act.model_dump()) for act in result.actions])
        validate_plan(plan)
        return {"plan": plan}
    except Exception as e:
        logger.error(f"[LLM_ERROR] {e} - falling back to regex parser")
        from app.features.tasks.parser import deterministic_nlp_parser
        try:
            plan = deterministic_nlp_parser(user_message)
            return {"plan": plan}
        except Exception as e2:
            return {"plan": AgentPlan(actions=[AgentAction(action="OUT_OF_SCOPE")])}

def execute_node(state: AgentState, config: RunnableConfig):
    """Executes the plan against SQLite and ChromaDB."""
    db: Session = config["configurable"]["db"]
    plan = state["plan"]
    
    replies = []
    highlighted = []
    board_updated = False
    new_pending_action = None

    for act in plan.actions:
        logger.info(f"[ACTION_EXEC] action={act.action} | payload={act.model_dump()}")

        if act.action == "CREATE_TASK":
            new_id = generate_deterministic_id(db)
            new_task = TaskModel(
                id=new_id, title=(act.title or "Untitled")[:200], 
                description=(act.description or "")[:1000],
                status=act.status or "To do", priority=act.priority or "Medium", 
                due_date=act.due_date or "Tomorrow"
            )
            db.add(new_task)
            db.commit()
            sync_task_to_chroma(new_task)
            replies.append(f'Created task "{new_task.title}".')
            highlighted.append(new_task.id)
            board_updated = True

        elif act.action in ["UPDATE_TASK", "DELETE_TASK"]:
            target_q = (act.target_query or "").strip()
            
            # If missing target, pause and ask for clarification
            if not target_q:
                new_pending_action = act.model_dump()
                action_word = "update" if act.action == "UPDATE_TASK" else "delete"
                replies.append(f"Which task would you like me to {action_word}?")
                continue # Stop processing this action

            res = find_matching_tasks(db, target_q, for_mutation=True)
            
            if not res["matches"]:
                replies.append(f'Could not find any task matching "{target_q}".')
            elif res["ambiguous"]:
                # If ambiguous, save the action and ask for clarification
                new_pending_action = act.model_dump()
                titles = ", ".join(f'"{m.title}"' for m in res["matches"])
                replies.append(f'Clarification needed: Multiple tasks match "{target_q}" ({titles}).')
            else:
                # Exact match found - Execute!
                target = res["matches"][0]
                if act.action == "DELETE_TASK":
                    remove_task_from_chroma(target.id)
                    db.delete(target)
                    db.commit()
                    replies.append(f'Deleted task "{target.title}".')
                    board_updated = True
                else:
                    changes = []
                    if act.status: 
                        target.status = act.status
                        if act.status == "Done": target.due_date = "Completed"
                        changes.append(f"to {act.status}")
                    if act.priority: 
                        target.priority = act.priority
                        changes.append(f"priority to {act.priority}")
                    if act.due_date: 
                        target.due_date = act.due_date
                        changes.append(f"due date to {act.due_date}")
                    if act.title: 
                        target.title = act.title[:200]
                        changes.append(f'title to "{target.title}"')
                    
                    db.commit()
                    sync_task_to_chroma(target)
                    replies.append(f'Updated "{target.title}" ({" and ".join(changes) or "updated"}).')
                    highlighted.append(target.id)
                    board_updated = True

        elif act.action in ("SEARCH_TASKS", "LIST_TASKS"):
            target_q = (act.target_query or "").strip()
            status_filter = act.message if act.message == "UNFINISHED" else act.status
            res = find_matching_tasks(db, target_q, status_filter=status_filter, priority_filter=act.priority, for_mutation=False)
            
            if not res["matches"]:
                replies.append(f'No matching tasks found for "{target_q or "your filter"}".')
            else:
                found = ", ".join(m.title for m in res["matches"])
                replies.append(f"Found: {found}")
                highlighted.extend([m.id for m in res["matches"]])

        else:
            replies.append(OUT_OF_SCOPE_REPLY)

    final_reply = " ".join(replies)
    if board_updated and not final_reply.startswith("Done."):
        final_reply = "Done. " + final_reply

    return {
        "final_reply": final_reply,
        "replies": replies,
        "highlighted": highlighted,
        "board_updated": board_updated,
        "pending_action": new_pending_action,
        "messages": [AIMessage(content=final_reply)]
    }

# --- 4. Compile the Graph ---
def route_guardrail(state: AgentState):
    if state.get("final_reply") == OUT_OF_SCOPE_REPLY:
        return "end"
    return "parse"

workflow = StateGraph(AgentState)

workflow.add_node("guardrail", guardrail_node)
workflow.add_node("parse", parse_node)
workflow.add_node("execute", execute_node)

workflow.set_entry_point("guardrail")
workflow.add_conditional_edges("guardrail", route_guardrail, {"end": END, "parse": "parse"})
workflow.add_edge("parse", "execute")
workflow.add_edge("execute", END)

memory = MemorySaver()
app_graph = workflow.compile(checkpointer=memory)