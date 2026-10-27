import logging
from typing import Any, Dict, Optional
from sqlalchemy.orm import Session

from app.features.tasks.models import TaskModel
from app.core.chroma import collection

logger = logging.getLogger("TaskFlowAgent")


def generate_deterministic_id(db: Session) -> str:
    """Generates sequential deterministic IDs (task-1, task-2, ...)."""
    existing = db.query(TaskModel).all()
    max_num = 0
    for t in existing:
        if t.id.startswith("task-"):
            try:
                num = int(t.id.split("-")[1])
                if num > max_num:
                    max_num = num
            except ValueError:
                pass
    new_id = f"task-{max_num + 1}"
    logger.info(f"[ID_GENERATOR] Generated deterministic ID: {new_id}")
    return new_id


def find_matching_tasks(
    db: Session,
    query_text: str,
    status_filter: Optional[str] = None,
    priority_filter: Optional[str] = None,
    for_mutation: bool = False,
) -> Dict[str, Any]:
    logger.info(
        f"[SEARCH_START] query='{query_text}' | status_filter={status_filter} "
        f"| priority_filter={priority_filter} | for_mutation={for_mutation}"
    )
    all_tasks = db.query(TaskModel).all()
    if not all_tasks:
        logger.warning("[SEARCH_EMPTY] No tasks exist in SQLite database.")
        return {"matches": [], "ambiguous": False}

    task_map = {t.id: t for t in all_tasks}
    q_clean = (query_text or "").strip().lower()
    if not q_clean or q_clean in ["all", "all tasks", "overview", "everything", "my tasks"]:
        filtered = []
        for t in all_tasks:
            if status_filter == "UNFINISHED" and t.status == "Done": continue
            elif status_filter and status_filter != "UNFINISHED" and t.status.lower() != status_filter.lower(): continue
            if priority_filter and t.priority.lower() != priority_filter.lower(): continue
            filtered.append(t)
        return {"matches": filtered, "ambiguous": False}
    if q_clean in task_map:
        logger.info(f"[SEARCH_DIRECT_ID] Matched exact task ID: {q_clean}")
        return {"matches": [task_map[q_clean]], "ambiguous": False}

    exact_matches = []
    substring_matches = []
    for t in all_tasks:
        t_title = t.title.lower()
        if q_clean and q_clean == t_title:
            exact_matches.append(t)
        elif q_clean and (q_clean in t_title or any(w in t_title for w in q_clean.split() if len(w) > 2)):
            substring_matches.append(t)

    if len(exact_matches) == 1:
        logger.info(
            f"[SEARCH_EXACT_TITLE] Matched single exact title: "
            f"'{exact_matches[0].title}' ({exact_matches[0].id})"
        )
        return {"matches": exact_matches, "ambiguous": False}

    n_results = min(len(all_tasks), 6)
    chroma_res = collection.query(
        query_texts=[query_text if query_text else "task"], n_results=n_results
    )
    ids = chroma_res.get("ids", [[]])[0]
    distances = chroma_res.get("distances", [[]])[0]

    distance_log = [f"{tid}:{dist:.4f}" for tid, dist in zip(ids, distances)]
    logger.info(f"[CHROMA_QUERY_SCORES] query='{query_text}' -> {distance_log}")

    scored_tasks = []
    for tid, dist in zip(ids, distances):
        if tid in task_map:
            t = task_map[tid]
            if status_filter == "UNFINISHED" and t.status == "Done":
                continue
            elif status_filter and status_filter != "UNFINISHED" and t.status.lower() != status_filter.lower():
                continue
            if priority_filter and t.priority.lower() != priority_filter.lower():
                continue
            scored_tasks.append((t, dist))

    if for_mutation:
        if len(substring_matches) > 1:
            matched_names = [m.title for m in substring_matches]
            logger.warning(f"[AMBIGUITY_DETECTED] Multiple substring matches for '{query_text}': {matched_names}")
            return {"matches": substring_matches, "ambiguous": True}
        if len(substring_matches) == 1:
            logger.info(
                f"[MUTATION_TARGET_RESOLVED] Substring match: "
                f"'{substring_matches[0].title}' ({substring_matches[0].id})"
            )
            return {"matches": substring_matches, "ambiguous": False}

        close_vectors = [t for t, d in scored_tasks if d < 0.60]
        if len(close_vectors) > 1 and abs(scored_tasks[0][1] - scored_tasks[1][1]) < 0.08:
            matched_names = [m.title for m in close_vectors[:3]]
            logger.warning(f"[AMBIGUITY_DETECTED] Close vector distances for '{query_text}': {matched_names}")
            return {"matches": close_vectors[:3], "ambiguous": True}
        elif len(close_vectors) >= 1:
            logger.info(
                f"[MUTATION_TARGET_RESOLVED] Vector match: "
                f"'{close_vectors[0].title}' ({close_vectors[0].id})"
            )
            return {"matches": [close_vectors[0]], "ambiguous": False}

        logger.warning(f"[MUTATION_TARGET_NOT_FOUND] No task matched '{query_text}'")
        return {"matches": [], "ambiguous": False}

    results = []
    seen_ids = set()
    for t in substring_matches:
        if status_filter == "UNFINISHED" and t.status == "Done":
            continue
        elif status_filter and status_filter != "UNFINISHED" and t.status.lower() != status_filter.lower():
            continue
        if priority_filter and t.priority.lower() != priority_filter.lower():
            continue
        results.append(t)
        seen_ids.add(t.id)

    for t, dist in scored_tasks:
        if t.id not in seen_ids and dist < 0.72:
            results.append(t)
            seen_ids.add(t.id)

    logger.info(f"[SEARCH_RESULTS] Returned {len(results)} tasks: {[r.id for r in results]}")
    return {"matches": results, "ambiguous": False}