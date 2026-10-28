# TASKFLOW — AI Agent Task Manager

A single-page AI Task Manager built for the **AI & Vibe Coding 2-Day Challenge**. Users manage their entire task board (create, list, edit, move, complete, delete, and semantically search tasks) through a natural language AI chat interface.

---

## 1. Tech Stack & Architectural Choices

- **Vector Store (Mandatory):** **ChromaDB** (`PersistentClient` stored in `./chroma_db`) using cosine similarity for semantic task embeddings and retrieval.
- **Primary Source of Truth:** **SQLite + SQLAlchemy** (`tasks.db`) with deterministic task IDs (`task-1`, `task-2`, etc.) so all task states persist across browser refreshes and server restarts.
- **AI & Action Execution Engine:** **LangGraph** + **FastAPI** + **Pydantic**.
  - **Stateful Orchestration:** Uses LangGraph `StateGraph` and `MemorySaver` to maintain conversation context and `pending_action` states across stateless HTTP requests. This enables multi-turn clarification without losing the user's original intent.
  - **LLM:** **Groq** (`openai/gpt-oss-20b`) via the OpenAI-compatible API, configured using `GROQ_API_KEY` and `GROQ_MODEL`.
  - **Structured Actions:** The LLM outputs strictly formatted JSON matching Pydantic `AgentPlan` / `AgentAction` schemas. The backend validates and executes them.
  - **Fallback Parser:** Includes a built-in Deterministic NLP Fallback Parser so basic commands work even without an API key.
  - **Ambiguity Guardrail:** If a user asks to edit/delete an ambiguous task matching multiple records, the LangGraph state machine pauses execution, saves the intent, and asks for clarification before mutating the database.
- **Single-Page UI:** Responsive split-screen **TASKFLOW** interface (`static/index.html` with Tailwind CSS) featuring the live Task Board on the left and the AI Assistant conversation panel on the right.

---

## 2. Quick Setup & Run Instructions

```bash
# 1. Clone the repository
git clone https://github.com/oneakash/ai_task_manager.git
cd ai_task_manager

# 2. Setup Virtual Environment & Install dependencies
python -m venv .venv
# or
python3 -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Configure Groq API Key
cp .env.example .env
# Edit .env to add your Groq credentials:
# GROQ_API_KEY=gsk_...
# GROQ_MODEL=openai/gpt-oss-20b

# 4. Start the server
uvicorn app.main:app --reload --port 8000
```

Open **http://localhost:8000** in your browser.

---

## 3. Example Instructions to Test

1. `"Create a high-priority task to finish the demo tomorrow."`
2. `"Show all unfinished tasks related to documentation."`
3. `"Move the demo task to In progress and mark the weekly report Done."`
4. `"Change the priority of the API task to Low."`
5. **Multi-turn Clarification Test:** Type `"Update priority to medium"` (without a task name). The agent will ask *"Which task?"*. Reply with `"call to my mother"` — it will remember the pending action and apply the Medium priority to that specific task.
6. **Ambiguity Test:** Create two tasks with the word "presentation", then type `"Delete the presentation task"`. The agent will ask you to clarify which one before deleting.

---

## 4. Meaningful Prompts, Failures Encountered & How They Were Corrected

### Prompt 1: Persistent Storage & ChromaDB Synchronization
> *"Build a FastAPI backend using SQLite as the primary source of truth and persistent ChromaDB for semantic search. Define a Task schema with deterministic IDs, title, description, status, priority, and due_date. Synchronize every task creation, update, and deletion with ChromaDB."*

- **Failure Encountered:** Updating an existing task caused duplicate ID errors when calling `.add()` in ChromaDB, and optional `None` fields in metadata caused ChromaDB validation crashes.
- **How I Corrected It:** Switched `.add()` to idempotent `.upsert()` using the deterministic task ID, sanitized `None` metadata fields to empty strings `""`, and added `collection.delete(ids=[task_id])` on task deletion so stale vectors never appear.

### Prompt 2: Multi-Action Agent & Deterministic ID Lookup
> *"Create an AI agent orchestration layer that parses natural language instructions into a Pydantic `AgentPlan` containing `actions: List[AgentAction]`. Support compound commands executing sequentially in a single turn."*

- **Failure Encountered:** Compound instructions like *"Move the demo task to In progress and mark the weekly report Done"* only executed the first clause. Additionally, the LLM tried to guess `task_id` values directly.
- **How I Corrected It:** Wrapped the output schema in `actions: List[AgentAction]` and replaced direct LLM ID guessing with a `target_query` string that the backend resolves against SQLite + ChromaDB.

### Prompt 3: Semantic Search & Ambiguity Clarification Guardrail
> *"Implement semantic task search with ChromaDB and add a disambiguation check before `UPDATE_TASK` or `DELETE_TASK`. If multiple tasks match closely, do not mutate; return a clarification prompt."*

- **Failure Encountered:** ChromaDB `.query(n_results=5)` returned top-K tasks even if unrelated, and ambiguous delete commands immediately deleted the top match.
- **How I Corrected It:** Added a cosine distance threshold filter (`dist < 0.72`), combined with metadata filtering, and implemented `for_mutation=True` ambiguity detection that halts updates/deletes when multiple tasks match.

### Prompt 4: Stateful Multi-Turn Clarification via LangGraph
> *"Implement a LangGraph state machine with a `MemorySaver` checkpointer. If the AI detects a mutation intent missing a target task (e.g., 'Change priority to Medium'), the graph must pause, save the `pending_action` to memory, and ask the user for the task name. Resume the exact action on the next message."*

- **Failure Encountered:** When asked *"update priority to medium"*, the agent asked for a task name. But when the user replied *"call to my mother"*, the backend created a **new task** instead of resuming the pending action. The `guardrail_node` also blocked the reply because the phrase wasn't recognized as a valid task command. Furthermore, stateless FastAPI requests were wiping the LangGraph memory.
- **How I Corrected It:** 
  1. Removed explicit state initialization in the FastAPI service, allowing LangGraph's `MemorySaver` to resume state using `thread_id`.
  2. Used `RunnableConfig` in all graph nodes to fix dependency injection errors.
  3. Added an "Action Verb Bypass" to the guardrail: if the user is answering a clarification, the guardrail skips the semantic block and passes the text directly to the `parse_node`.

### Prompt 5: Strict LLM JSON Validation & Guardrail False Positives
> *"Ensure the `openai/gpt-oss-20b` model strictly outputs valid JSON matching the Pydantic schema, and prevent the guardrail from blocking basic task creation commands like 'create new task titled meet with rasel bhai'."*

- **Failure Encountered:** The Groq model frequently failed JSON validation, causing fallback to the regex parser which couldn't understand natural phrasing. The guardrail blocked new task titles because they didn't contain standard keywords.
- **How I Corrected It:** Upgraded to LangChain's `with_structured_output()` to guarantee valid JSON generation from Groq. Added few-shot examples directly into the `SystemMessage` prompt. Implemented an `any(verb in message for verb in task_verbs)` bypass in the guardrail node so users can use natural phrasing without being blocked.

### Prompt 6: Single-Page Split-Screen UI
> *"Build a responsive single-page frontend matching the TASKFLOW UI design with the Task Board on the left and the AI Assistant chat on the right. Ensure every chat instruction immediately updates the Task Board."*

- **Failure Encountered:** The task board required a manual page refresh, and long chat threads pushed the input box off-screen.
- **How I Corrected It:** Returned the updated `tasks` array and `highlighted_ids` directly inside the `POST /api/chat` response to re-render instantly, and constrained the chat card to a fixed flex column (`h-[640px]`) with `overflow-y-auto`.