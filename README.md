# TASKFLOW — AI Agent Task Manager

A single-page AI Task Manager built for the **AI & Vibe Coding 2-Day Challenge**. Users manage their entire task board (create, list, edit, move, complete, delete, and semantically search tasks) through a natural language AI chat interface.

---

## 1. Tech Stack & Architectural Choices

- **Vector Store (Mandatory):** **ChromaDB** (`PersistentClient` stored in `./chroma_db`) using cosine similarity for semantic task embeddings and retrieval.
- **Primary Source of Truth:** **SQLite + SQLAlchemy** (`tasks.db`) with deterministic task IDs (`task-1`, `task-2`, etc.) so all task states persist across browser refreshes and server restarts.
- **AI & Action Execution Engine:** **FastAPI + Pydantic Structured Actions** (`AgentPlan` / `AgentAction`).
  - Supports **OpenAI Structured Outputs** (`gpt-4o-mini` via `OPENAI_API_KEY`) *and* includes a built-in **Deterministic NLP Fallback Parser** so every assignment command works out-of-the-box even without an API key.
  - Supports **compound multi-step instructions** in a single message (e.g., *"Move the demo task to In progress and mark the weekly report Done"*).
  - Includes an **Ambiguity Guardrail**: if a user asks to edit or delete an ambiguous task that matches multiple records closely, the agent pauses and asks for clarification before mutating the database.
- **Single-Page UI:** Responsive split-screen **TASKFLOW** interface (`static/index.html` with Tailwind CSS) featuring the live Task Board on the left and the AI Assistant conversation panel on the right.

---

## 2. Quick Setup & Run Instructions

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Configure OpenAI API Key
cp .env.example .env
# Edit .env to add OPENAI_API_KEY=sk-... (or leave blank to use the built-in NLP engine)

# 3. Start the server
uvicorn main:app --reload --port 8000
```

Open **http://localhost:8000** in your browser.

---

## 3. Example Instructions to Test

1. `"Create a high-priority task to finish the demo tomorrow."`
2. `"Show all unfinished tasks related to documentation."`
3. `"Move the demo task to In progress and mark the weekly report Done."`
4. `"Change the priority of the API task to Low."`
5. **Ambiguity Test:** Create another task named `"Prepare client presentation"` and then type `"Delete the presentation task"` — the agent will ask you to clarify which presentation task you mean before deleting.

---

## 4. Meaningful Prompts, Failures Encountered & How They Were Corrected

### Prompt 1: Persistent Storage & ChromaDB Synchronization
> *"Build a FastAPI backend using SQLite as the primary source of truth and persistent ChromaDB for semantic search. Define a Task schema with deterministic IDs (`task-1`, `task-2`), `title`, `description`, `status` ('To do', 'In progress', 'Done'), `priority` ('Low', 'Medium', 'High'), and `due_date`. Synchronize every task creation, update, and deletion with ChromaDB."*

- **Failure Encountered:** Updating an existing task caused duplicate ID errors when calling `.add()` in ChromaDB, and optional `None` fields in metadata caused ChromaDB validation crashes.
- **How I Corrected It:** Switched `.add()` to idempotent `.upsert()` using the deterministic task ID, sanitized `None` metadata fields to empty strings `""`, and added `collection.delete(ids=[task_id])` on task deletion so stale vectors never appear in search results.

### Prompt 2: Structured Multi-Action Agent & Deterministic ID Lookup
> *"Create an AI agent orchestration layer that parses natural language instructions into a Pydantic `AgentPlan` containing `actions: List[AgentAction]`. Support `CREATE_TASK`, `UPDATE_TASK`, `DELETE_TASK`, and `SEARCH_TASKS` so compound commands execute sequentially in a single turn."*

- **Failure Encountered:** Initially, compound instructions like *"Move the demo task to In progress and mark the weekly report Done"* only executed the first clause because the schema returned a single action object. Additionally, the LLM tried to guess `task_id` values directly.
- **How I Corrected It:** Wrapped the output schema in `actions: List[AgentAction]` and replaced direct LLM ID guessing with a `target_query` string that the backend resolves against SQLite + ChromaDB.

### Prompt 3: Semantic Search & Ambiguity Clarification Guardrail
> *"Implement semantic task search with ChromaDB and add a disambiguation check before `UPDATE_TASK` or `DELETE_TASK`. If multiple tasks match a user's query closely, do not mutate the database; return a clarification prompt listing the matching task titles."*

- **Failure Encountered:** ChromaDB's `.query(n_results=5)` always returned top-K tasks even when they were unrelated to *"documentation"*, and ambiguous delete commands immediately deleted the top match without asking.
- **How I Corrected It:** Added a cosine distance threshold filter (`dist < 0.72`) combined with status/priority metadata filtering, and implemented `for_mutation=True` ambiguity detection that halts updates/deletes when multiple tasks match the query keyword or have near-identical vector distances.

### Prompt 4: Single-Page Split-Screen UI (TASKFLOW Reference)
> *"Build a responsive single-page frontend matching the TASKFLOW UI design with the Task Board on the left and the AI Assistant chat on the right. Ensure every chat instruction immediately updates the Task Board without a page reload."*

- **Failure Encountered:** The task board originally required a manual page refresh after chat commands, and long chat threads pushed the instruction input box off the bottom of the viewport.
- **How I Corrected It:** Returned the updated `tasks` array and `highlighted_ids` directly inside the `POST /api/chat` response payload to re-render the task list instantaneously, and constrained the right-hand chat card to a fixed flex column (`h-[640px]`) with `overflow-y-auto`.
